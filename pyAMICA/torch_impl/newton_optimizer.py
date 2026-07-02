"""
Newton optimization for PyTorch AMICA matching Fortran implementation.

This module implements the Newton optimization method as used in the Fortran
AMICA code, with automatic ramping and proper convergence behavior.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class AMICANewtonOptimizer:
    """
    Newton optimizer specifically for AMICA that matches Fortran behavior.
    
    The Fortran implementation gradually transitions to Newton method:
    - Starts with natural gradient (lrate = 0.05)
    - Ramps up to Newton (lrate → 1.0) after newt_start iterations
    - Uses approximate Hessian based on source statistics
    
    Parameters
    ----------
    model : nn.Module
        AMICA model
    newt_start : int, default=50
        Iteration to start Newton method
    newt_ramp : int, default=10
        Number of iterations to ramp to full Newton
    use_pytorch_minimize : bool, default=False
        Whether to use pytorch-minimize (if available)
    """
    
    def __init__(
        self,
        model: nn.Module,
        newt_start: int = 50,
        newt_ramp: int = 10,
        use_pytorch_minimize: bool = False
    ):
        self.model = model
        self.newt_start = newt_start
        self.newt_ramp = newt_ramp
        self.current_iter = 0
        
        # Try to load pytorch-minimize if requested
        self.use_pytorch_minimize = False
        if use_pytorch_minimize:
            try:
                from pytorch_minimize import minimize
                self.minimize_fn = minimize
                self.use_pytorch_minimize = True
                logger.info("Using pytorch-minimize for Newton optimization")
            except ImportError:
                logger.warning("pytorch-minimize not available, using custom implementation")
    
    def get_learning_rate(self, base_lrate: float = 0.05, newtrate: float = 0.5) -> float:
        """
        Get learning rate with Newton ramping like Fortran.
        
        Fortran behavior:
        - Before newt_start: use base_lrate  
        - During ramping: double each iteration, cap at newtrate
        - After ramping: use newtrate
        
        The Fortran code does:
        lrate = min(newtrate, lrate + min(1/newt_ramp, lrate))
        This effectively doubles lrate each iteration during ramping.
        
        Parameters
        ----------
        base_lrate : float
            Base learning rate for natural gradient
        newtrate : float
            Maximum Newton learning rate (default 0.5)
            
        Returns
        -------
        lrate : float
            Current learning rate
        """
        if self.current_iter < self.newt_start:
            # Natural gradient phase
            return base_lrate
        else:
            # Newton phase - compute how many iterations into Newton we are
            newton_iter = self.current_iter - self.newt_start
            
            # Start with base learning rate
            lrate = base_lrate
            
            # Double the learning rate for each Newton iteration (up to newt_ramp iterations)
            for _ in range(min(newton_iter + 1, self.newt_ramp)):
                # Fortran: lrate = min(newtrate, lrate + min(1/newt_ramp, lrate))
                increment = min(1.0 / self.newt_ramp, lrate)
                lrate = min(newtrate, lrate + increment)
            
            return lrate
    
    def compute_newton_direction(
        self,
        W: torch.Tensor,
        Y: torch.Tensor,
        grad_W: torch.Tensor,
        model_idx: int = 0
    ) -> torch.Tensor:
        """
        Compute Newton direction using AMICA-specific approximate Hessian.
        
        This matches the Fortran implementation which uses source statistics
        to approximate the Hessian for efficient computation.
        
        Parameters
        ----------
        W : torch.Tensor
            Current unmixing matrix
        Y : torch.Tensor
            Current sources (n_sources, n_samples)
        grad_W : torch.Tensor
            Natural gradient of W
        model_idx : int
            Model index for multi-model AMICA
            
        Returns
        -------
        H : torch.Tensor
            Newton direction
        """
        n_sources = W.shape[0]
        device = W.device
        dtype = W.dtype
        
        # Compute source statistics for Hessian approximation
        with torch.no_grad():
            # Second moments (diagonal of Hessian)
            sigma2 = torch.mean(Y ** 2, dim=1)  # (n_sources,)
            
            # Fourth moments (for off-diagonal)
            # This approximates the interaction between components
            Y_abs = torch.abs(Y)
            kappa = torch.mean(Y_abs ** 3, dim=1) / (torch.mean(Y_abs, dim=1) ** 3 + 1e-10)
            
            # Compute approximate inverse Hessian (Newton direction)
            H = torch.zeros_like(grad_W)
            
            # Diagonal scaling
            diag_scale = 1.0 / (sigma2 + 1e-10)
            H.diagonal().copy_(grad_W.diagonal() * diag_scale)
            
            # Off-diagonal elements with stability check
            for i in range(n_sources):
                for j in range(n_sources):
                    if i != j:
                        sk1 = sigma2[i] * kappa[j]
                        sk2 = sigma2[j] * kappa[i]
                        denom = sk1 * sk2 - 1.0
                        
                        if abs(denom) > 0.1:  # Stability threshold
                            H[i, j] = (sk1 * grad_W[i, j] - grad_W[j, i]) / denom
                        else:
                            # Fall back to scaled gradient
                            H[i, j] = grad_W[i, j] * diag_scale[i]
        
        return H
    
    def step(
        self,
        X: torch.Tensor,
        base_lrate: float = 0.05,
        use_newton: bool = True,
        current_iter: int = None
    ) -> Tuple[float, bool]:
        """
        Perform optimization step with automatic Newton ramping.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data (n_channels, n_samples)
        base_lrate : float
            Base learning rate
        use_newton : bool
            Whether to use Newton (will auto-enable based on iteration)
        current_iter : int, optional
            Current iteration number (if None, uses internal counter)
            
        Returns
        -------
        lrate : float
            Learning rate used
        is_newton : bool
            Whether Newton method was used
        """
        # Use provided iteration or increment internal counter
        if current_iter is not None:
            self.current_iter = current_iter
        else:
            self.current_iter += 1
        
        # Determine if we should use Newton based on iteration
        is_newton = use_newton and (self.current_iter >= self.newt_start)
        
        # Get ramped learning rate
        lrate = self.get_learning_rate(base_lrate) if is_newton else base_lrate
        
        if is_newton and self.use_pytorch_minimize:
            # Use pytorch-minimize for Newton step
            self._step_with_minimize(X, lrate)
        elif is_newton:
            # Use custom Newton implementation
            self._step_custom_newton(X, lrate)
        else:
            # Natural gradient step (handled by main optimizer)
            pass
        
        return lrate, is_newton
    
    def _step_custom_newton(self, X: torch.Tensor, lrate: float):
        """
        Custom Newton step matching Fortran implementation.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data
        lrate : float
            Learning rate (1.0 for full Newton)
        """
        model = self.model
        
        with torch.enable_grad():
            # Forward pass
            model.zero_grad()
            neg_ll = -model.compute_log_likelihood(X)
            neg_ll.backward()
            
            # For each model's mixing matrix
            for h in range(model.n_models):
                A = model.A[h]
                
                if A.grad is not None:
                    # Get unmixing matrix
                    W = torch.linalg.pinv(A.T).T
                    
                    # Get sources
                    Y = model._forward_single(X, h)
                    
                    # Natural gradient (W.T @ W @ A.grad)
                    nat_grad = W.T @ W @ A.grad
                    
                    # Compute Newton direction
                    newton_dir = self.compute_newton_direction(
                        W, Y, nat_grad, model_idx=h
                    )
                    
                    # Update with Newton direction
                    with torch.no_grad():
                        # Newton update: A += lrate * A @ newton_dir
                        A.data.add_(A @ newton_dir, alpha=lrate)
    
    def _step_with_minimize(self, X: torch.Tensor, lrate: float):
        """
        Newton step using pytorch-minimize library.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data
        lrate : float
            Learning rate
        """
        def objective(params_flat):
            # Unflatten parameters to model
            self._unflatten_params(params_flat)
            
            # Compute loss
            neg_ll = -self.model.compute_log_likelihood(X)
            
            return neg_ll
        
        # Flatten current parameters
        params_flat = self._flatten_params()
        params_flat.requires_grad_(True)
        
        # Run Newton-CG optimization
        result = self.minimize_fn(
            objective,
            params_flat,
            method='newton-cg',
            options={
                'maxiter': 5,  # Few iterations per step
                'lr': lrate,    # Use ramped learning rate
                'gtol': 1e-5
            }
        )
        
        # Update model with optimized parameters
        self._unflatten_params(result.x)
    
    def _flatten_params(self) -> torch.Tensor:
        """Flatten model parameters."""
        params = []
        for p in self.model.parameters():
            params.append(p.data.view(-1))
        return torch.cat(params)
    
    def _unflatten_params(self, params_flat: torch.Tensor):
        """Unflatten parameters back to model."""
        idx = 0
        for p in self.model.parameters():
            numel = p.numel()
            p.data.copy_(params_flat[idx:idx+numel].view_as(p))
            idx += numel
    
    def reset(self):
        """Reset iteration counter for new optimization run."""
        self.current_iter = 0