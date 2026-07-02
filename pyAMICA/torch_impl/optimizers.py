"""
Optimization methods for PyTorch AMICA.
"""

import torch
import torch.nn as nn
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class NaturalGradientOptimizer(torch.optim.Optimizer):
    """
    Natural Gradient Descent optimizer.

    Uses the Fisher Information Matrix to precondition gradients,
    following the natural geometry of the parameter space.

    Parameters
    ----------
    params : iterable
        Iterable of parameters to optimize
    lr : float, default=0.1
        Learning rate
    eps : float, default=1e-8
        Small value for numerical stability
    fisher_type : str, default='diagonal'
        Type of Fisher matrix approximation ('diagonal', 'kfac', 'full')
    """

    def __init__(
        self, params, lr: float = 0.1, eps: float = 1e-8, fisher_type: str = "diagonal"
    ):
        defaults = dict(lr=lr, eps=eps, fisher_type=fisher_type)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                # Initialize state
                if len(state) == 0:
                    state["step"] = 0
                    state["fisher_ema"] = torch.zeros_like(grad)

                fisher_ema = state["fisher_ema"]
                state["step"] += 1

                # Update Fisher estimate (diagonal approximation)
                if group["fisher_type"] == "diagonal":
                    # Use gradient squared as Fisher diagonal estimate
                    fisher_ema.mul_(0.9).add_(grad**2, alpha=0.1)

                    # Natural gradient step
                    step_size = group["lr"]
                    p.add_(grad / (fisher_ema.sqrt() + group["eps"]), alpha=-step_size)

                elif group["fisher_type"] == "full":
                    # Full Fisher (expensive, not recommended for large models)
                    # This would require computing Hessian
                    raise NotImplementedError("Full Fisher matrix not yet implemented")

                elif group["fisher_type"] == "kfac":
                    # KFAC approximation (for layer-wise parameters)
                    raise NotImplementedError("KFAC approximation not yet implemented")

        return loss


class NewtonOptimizer:
    """
    Newton optimization for AMICA using PyTorch.

    This optimizer computes the Newton direction using the Hessian
    or its approximations for faster convergence.

    Parameters
    ----------
    model : nn.Module
        The AMICA model to optimize
    method : str, default='cg'
        Newton method to use ('exact', 'cg', 'lbfgs')
    max_iter : int, default=10
        Maximum iterations per Newton step
    tol : float, default=1e-6
        Convergence tolerance
    """

    def __init__(
        self,
        model: nn.Module,
        method: str = "cg",
        max_iter: int = 10,
        tol: float = 1e-6,
    ):
        self.model = model
        self.method = method
        self.max_iter = max_iter
        self.tol = tol

        # Check if pytorch-minimize is available
        self.use_pytorch_minimize = False
        try:
            from pytorch_minimize import minimize

            self.minimize = minimize
            self.use_pytorch_minimize = True
            logger.info("Using pytorch-minimize for Newton optimization")
        except ImportError:
            logger.warning("pytorch-minimize not found, using basic Newton-CG")

    def step(self, X: torch.Tensor):
        """
        Perform a Newton optimization step.

        Parameters
        ----------
        X : torch.Tensor
            Input data
        """
        if self.use_pytorch_minimize:
            self._step_with_minimize(X)
        else:
            self._step_basic(X)

    def _step_with_minimize(self, X: torch.Tensor):
        """Newton step using pytorch-minimize."""

        def objective(params_flat):
            # Unflatten parameters
            self._unflatten_params(params_flat)

            # Compute negative log-likelihood
            neg_ll = -self.model.compute_log_likelihood(X)

            # Compute gradient if needed
            if params_flat.requires_grad:
                neg_ll.backward()

            return neg_ll

        # Flatten parameters
        params_flat = self._flatten_params()
        params_flat.requires_grad_(True)

        # Minimize using selected method
        if self.method == "exact":
            method = "newton-exact"
        elif self.method == "cg":
            method = "newton-cg"
        else:
            method = "l-bfgs"

        result = self.minimize(
            objective,
            params_flat,
            method=method,
            options={"maxiter": self.max_iter, "tol": self.tol},
        )

        # Update model parameters
        self._unflatten_params(result.x)

    def _step_basic(self, X: torch.Tensor):
        """Basic Newton-CG step without pytorch-minimize."""

        # This is a simplified Newton-CG implementation
        # For production, use pytorch-minimize

        with torch.enable_grad():
            # Compute gradient
            self.model.zero_grad()
            neg_ll = -self.model.compute_log_likelihood(X)
            neg_ll.backward()

            # Get gradient vector
            grad = self._get_gradient_vector()

            # Compute Newton direction using CG
            # H * p = -g, where H is Hessian, p is Newton direction, g is gradient
            newton_dir = self._conjugate_gradient(X, grad)

            # Line search for step size
            step_size = self._line_search(X, newton_dir)

            # Update parameters
            self._update_params(newton_dir, step_size)

    def _conjugate_gradient(
        self, X: torch.Tensor, grad: torch.Tensor, max_iter: Optional[int] = None
    ) -> torch.Tensor:
        """
        Solve H * p = -g using Conjugate Gradient.

        Parameters
        ----------
        X : torch.Tensor
            Input data
        grad : torch.Tensor
            Gradient vector
        max_iter : int, optional
            Maximum CG iterations

        Returns
        -------
        p : torch.Tensor
            Newton direction
        """
        max_iter = max_iter or min(self.max_iter, len(grad))

        # Initialize
        p = torch.zeros_like(grad)
        r = -grad.clone()  # residual
        d = r.clone()  # direction
        r_norm_sq = torch.dot(r, r)

        for i in range(max_iter):
            if r_norm_sq < self.tol**2:
                break

            # Compute Hessian-vector product: H * d
            Hd = self._hessian_vector_product(X, d)

            # Update solution
            alpha = r_norm_sq / torch.dot(d, Hd)
            p.add_(d, alpha=alpha)
            r.sub_(Hd, alpha=alpha)

            # Update direction
            r_norm_sq_new = torch.dot(r, r)
            beta = r_norm_sq_new / r_norm_sq
            d = r + beta * d
            r_norm_sq = r_norm_sq_new

        return p

    def _hessian_vector_product(self, X: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """
        Compute Hessian-vector product using automatic differentiation.

        Parameters
        ----------
        X : torch.Tensor
            Input data
        v : torch.Tensor
            Vector to multiply with Hessian

        Returns
        -------
        Hv : torch.Tensor
            Hessian-vector product
        """
        # This computes H*v where H is the Hessian of the loss

        # First compute gradient
        self.model.zero_grad()
        neg_ll = -self.model.compute_log_likelihood(X)
        grads = torch.autograd.grad(neg_ll, self.model.parameters(), create_graph=True)

        # Flatten gradients
        grad_flat = torch.cat([g.reshape(-1) for g in grads])

        # Compute gradient-vector dot product
        gv = torch.dot(grad_flat, v)

        # Second derivative gives Hessian-vector product
        Hv = torch.autograd.grad(gv, self.model.parameters())
        Hv_flat = torch.cat([h.reshape(-1) for h in Hv])

        return Hv_flat

    def _line_search(
        self, X: torch.Tensor, direction: torch.Tensor, max_step: float = 1.0
    ) -> float:
        """
        Backtracking line search for step size.

        Parameters
        ----------
        X : torch.Tensor
            Input data
        direction : torch.Tensor
            Search direction
        max_step : float
            Maximum step size

        Returns
        -------
        step_size : float
            Optimal step size
        """
        # Simple backtracking line search
        step_size = max_step
        rho = 0.5  # Backtracking factor

        # Current loss
        with torch.no_grad():
            current_loss = -self.model.compute_log_likelihood(X)

        # Try decreasing step sizes
        for _ in range(20):
            # Update parameters tentatively
            self._update_params(direction, step_size)

            # Compute new loss
            with torch.no_grad():
                new_loss = -self.model.compute_log_likelihood(X)

            # Check Armijo condition
            if new_loss < current_loss:
                break

            # Revert parameters and decrease step size
            self._update_params(direction, -step_size)
            step_size *= rho

        return step_size

    def _flatten_params(self) -> torch.Tensor:
        """Flatten all model parameters into a single vector."""
        params = []
        for p in self.model.parameters():
            params.append(p.data.reshape(-1))
        return torch.cat(params)

    def _unflatten_params(self, params_flat: torch.Tensor):
        """Unflatten parameter vector back to model parameters."""
        idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data = params_flat[idx : idx + numel].view_as(p).clone()
            idx += numel

    def _get_gradient_vector(self) -> torch.Tensor:
        """Get flattened gradient vector."""
        grads = []
        for p in self.model.parameters():
            if p.grad is not None:
                grads.append(p.grad.reshape(-1))
            else:
                grads.append(torch.zeros_like(p).reshape(-1))
        return torch.cat(grads)

    def _update_params(self, direction: torch.Tensor, step_size: float):
        """Update model parameters along direction."""
        idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            update = direction[idx : idx + numel].view_as(p)
            p.data.add_(update, alpha=step_size)
            idx += numel
