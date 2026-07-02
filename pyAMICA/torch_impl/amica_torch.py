"""
Fixed PyTorch AMICA implementation with proper gradient handling and output modes.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union
from pathlib import Path
from tqdm import tqdm
import logging

from .fortran_output import FortranStyleOutput

logger = logging.getLogger(__name__)


class AMICATorch(nn.Module):
    """
    PyTorch implementation of Adaptive Mixture ICA (AMICA).

    Features:
    - GPU/MPS/CPU support with automatic device selection
    - Two output modes: normal (tqdm) and debug (Fortran-style)
    - Automatic differentiation
    - Numerical stability safeguards
    """

    def __init__(
        self,
        n_channels: int,
        n_sources: Optional[int] = None,
        n_models: int = 1,
        n_mix: int = 3,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self.n_channels = n_channels
        self.n_sources = n_sources or n_channels
        self.n_models = n_models
        self.n_mix = n_mix
        self.dtype = dtype

        # Set up device with MPS handling
        if device is None:
            self.device = self._setup_device()
        else:
            self.device = torch.device(device) if isinstance(device, str) else device

        # Initialize parameters
        self._init_parameters()

        # Move to device
        self.to(self.device)

        # Numerical stability parameters
        self.eps = 1e-10
        self.min_cond = 1e-15
        self.min_log = -1500.0
        self.max_val = 1e32
        self.min_eig = 1e-15

        logger.info(f"Initialized AMICATorch on {self.device}")

    def _setup_device(self) -> torch.device:
        """Automatically select the best available device."""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            # MPS available but with limitations
            logger.warning("Using MPS device. Some operations may fall back to CPU.")
            return torch.device("mps")
        else:
            return torch.device("cpu")

    def _init_parameters(self):
        """Initialize model parameters."""
        # Mixing matrices (one per model)
        self.A = nn.ParameterList(
            [
                nn.Parameter(
                    torch.eye(self.n_channels, self.n_sources, dtype=self.dtype)
                )
                for _ in range(self.n_models)
            ]
        )

        # Bias terms (one per model)
        self.c = nn.ParameterList(
            [
                nn.Parameter(torch.zeros(self.n_channels, 1, dtype=self.dtype))
                for _ in range(self.n_models)
            ]
        )

        # Model weights (log-space for stability)
        self.log_gm = nn.Parameter(torch.zeros(self.n_models, dtype=self.dtype))

        # Mixture parameters (log-space for positivity)
        self.log_alpha = nn.Parameter(
            torch.zeros(self.n_mix, self.n_sources, dtype=self.dtype)
        )
        self.mu = nn.Parameter(
            torch.zeros(self.n_mix, self.n_sources, dtype=self.dtype)
        )
        self.log_beta = nn.Parameter(
            torch.zeros(self.n_mix, self.n_sources, dtype=self.dtype)
        )
        self.rho_logit = nn.Parameter(
            torch.zeros(self.n_mix, self.n_sources, dtype=self.dtype)
        )

        # Preprocessing parameters (not trainable)
        self.register_buffer("mean", torch.zeros(self.n_channels, 1, dtype=self.dtype))
        self.register_buffer("sphere", torch.eye(self.n_channels, dtype=self.dtype))

    @property
    def gm(self) -> torch.Tensor:
        """Get normalized model weights."""
        return torch.softmax(self.log_gm, dim=0)

    @property
    def alpha(self) -> torch.Tensor:
        """Get normalized mixture weights."""
        return torch.softmax(self.log_alpha, dim=0)

    @property
    def beta(self) -> torch.Tensor:
        """Get positive scale parameters."""
        return torch.exp(self.log_beta) + self.eps

    @property
    def rho(self) -> torch.Tensor:
        """Get shape parameters in range [1, 2]."""
        return 1.0 + torch.sigmoid(self.rho_logit)

    def forward(self, X: torch.Tensor, model_idx: Optional[int] = None) -> torch.Tensor:
        """Forward pass through mixing model."""
        if model_idx is not None:
            return self._forward_single(X, model_idx)
        else:
            return torch.stack(
                [self._forward_single(X, h) for h in range(self.n_models)]
            )

    def _forward_single(self, X: torch.Tensor, model_idx: int) -> torch.Tensor:
        """Forward pass through a single model."""
        A = self.A[model_idx]
        c = self.c[model_idx]

        # Compute unmixing matrix (avoid in-place)
        W = torch.linalg.pinv(A.T).T

        # Apply unmixing
        Y = W @ (X - c)

        return Y

    def compute_log_likelihood(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood avoiding in-place operations.
        """
        n_samples = X.shape[1]

        # Use log-space throughout for stability
        log_model_lls = []

        for h in range(self.n_models):
            # Get sources
            Y = self._forward_single(X, h)

            # Log-determinant (avoid in-place)
            A = self.A[h]
            log_det_W = -torch.logdet(A[: self.n_sources, : self.n_sources])

            # Vectorized computation over samples
            log_mix_probs = []

            for k in range(self.n_mix):
                # Get parameters (no slicing that could cause issues)
                mu_k = self.mu[k, :].unsqueeze(1)
                beta_k = self.beta[k, :].unsqueeze(1)
                rho_k = self.rho[k, :].unsqueeze(1)
                alpha_k = self.alpha[k, :]

                # Compute log-PDF for all samples at once
                log_pdf = self._compute_gg_log_pdf_vectorized(Y, mu_k, beta_k, rho_k)

                # Add mixture weight (use mean for scalar)
                log_prob = log_pdf + torch.log(alpha_k.mean() + self.eps)
                log_mix_probs.append(log_prob)

            # Combine mixture components
            log_mix_probs_tensor = torch.stack(log_mix_probs, dim=0)
            log_y_prob = torch.logsumexp(log_mix_probs_tensor, dim=0)

            # Add Jacobian and sum over samples
            log_model_ll = log_y_prob.sum() + n_samples * log_det_W

            # Add model weight
            log_model_ll = log_model_ll + n_samples * torch.log(self.gm[h] + self.eps)
            log_model_lls.append(log_model_ll)

        # Combine models
        log_model_lls_tensor = torch.stack(log_model_lls)
        total_ll = torch.logsumexp(log_model_lls_tensor, dim=0)

        return total_ll / n_samples

    def _compute_gg_log_pdf_vectorized(
        self, Y: torch.Tensor, mu: torch.Tensor, beta: torch.Tensor, rho: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized Generalized Gaussian log-PDF.

        Y: (n_sources, n_samples)
        mu, beta, rho: (n_sources, 1)
        Returns: scalar (sum over sources)
        """
        # Ensure no in-place operations
        beta_safe = torch.clamp(beta, min=self.eps)
        rho_safe = torch.clamp(rho, min=1.0, max=2.0)

        # Compute normalized distance
        diff = torch.abs(Y - mu) / beta_safe

        # GG log-PDF
        log_p = -torch.pow(diff + self.eps, rho_safe)
        log_p = (
            log_p
            + torch.log(rho_safe)
            - torch.log(2 * beta_safe)
            - torch.lgamma(1 / rho_safe)
        )

        # Sum over sources
        log_p = log_p.sum(dim=0)

        # Clamp for stability
        log_p = torch.clamp(log_p, min=self.min_log)

        return log_p

    def preprocess_data(
        self, X: np.ndarray, do_mean: bool = True, do_sphere: bool = True
    ) -> torch.Tensor:
        """Preprocess data with MPS-safe operations."""
        X = torch.from_numpy(X).to(self.dtype)

        # Move to CPU for eigendecomposition if on MPS
        if self.device.type == "mps":
            X_cpu = X.cpu()

            if do_mean:
                self.mean = X_cpu.mean(dim=1, keepdim=True).to(self.device)
                X_cpu = X_cpu - self.mean.cpu()

            if do_sphere:
                cov = torch.cov(X_cpu)

                # Eigendecomposition on CPU
                eigvals, eigvecs = torch.linalg.eigh(cov)

                # Sort and filter
                idx = torch.argsort(eigvals, descending=True)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]

                keep = eigvals > self.min_eig
                eigvals = eigvals[keep]
                eigvecs = eigvecs[:, keep]

                # Compute sphering matrix
                sphere_cpu = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + self.eps))
                self.sphere = sphere_cpu.to(self.device)

                # Apply sphering
                X_cpu = sphere_cpu.T @ X_cpu

            # Move back to device
            X = X_cpu.to(self.device)
        else:
            # Standard processing for CUDA/CPU
            X = X.to(self.device)

            if do_mean:
                self.mean = X.mean(dim=1, keepdim=True)
                X = X - self.mean

            if do_sphere:
                cov = torch.cov(X)
                eigvals, eigvecs = torch.linalg.eigh(cov)

                idx = torch.argsort(eigvals, descending=True)
                eigvals = eigvals[idx]
                eigvecs = eigvecs[:, idx]

                keep = eigvals > self.min_eig
                eigvals = eigvals[keep]
                eigvecs = eigvecs[:, keep]

                self.sphere = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + self.eps))
                X = self.sphere.T @ X

        return X

    def fit(
        self,
        X: np.ndarray,
        max_iter: int = 100,
        lrate: float = 0.1,
        do_newton: bool = False,  # Disabled for now due to issues
        debug: bool = False,
        output_dir: Optional[str] = None,
        do_mean: bool = True,
        do_sphere: bool = True,
        verbose: bool = True,
    ) -> "AMICATorch":
        """
        Fit AMICA model.

        Parameters
        ----------
        X : np.ndarray
            Input data (n_channels, n_samples)
        max_iter : int
            Maximum iterations
        lrate : float
            Learning rate
        do_newton : bool
            Use Newton optimization (disabled for stability)
        debug : bool
            If True, use Fortran-style output; if False, use tqdm
        output_dir : str
            Output directory for debug mode
        do_mean : bool
            Whether to remove the mean during preprocessing
        do_sphere : bool
            Whether to sphere (whiten) the data during preprocessing
        verbose : bool
            Whether to display the tqdm progress bar (non-debug mode only)
        """
        # Preprocess
        X_prep = self.preprocess_data(X, do_mean=do_mean, do_sphere=do_sphere)
        n_samples = X_prep.shape[1]

        # Set up optimizer (use Adam for stability)
        optimizer = torch.optim.Adam(self.parameters(), lr=lrate)

        # Initialize history
        ll_history = []

        if debug:
            # Fortran-style output
            if output_dir is None:
                output_dir = Path.cwd() / "amica_output"
            else:
                output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            with FortranStyleOutput(
                str(output_dir / "out.txt"), verbose=True
            ) as output:
                config = {
                    "n_channels": self.n_channels,
                    "n_sources": self.n_sources,
                    "n_samples": n_samples,
                    "n_models": self.n_models,
                    "n_mix": self.n_mix,
                    "max_iter": max_iter,
                    "lrate": lrate,
                    "device": str(self.device),
                }
                output.write_header(config)
                output.write_info("Starting optimization...")

                for iter in range(max_iter):
                    # Zero gradients
                    optimizer.zero_grad()

                    # Compute loss
                    neg_ll = -self.compute_log_likelihood(X_prep)

                    if torch.isnan(neg_ll):
                        output.write_warning(f"NaN at iteration {iter}")
                        break

                    # Backward (with anomaly detection in debug)
                    if debug:
                        with torch.autograd.set_detect_anomaly(True):
                            neg_ll.backward()
                    else:
                        neg_ll.backward()

                    # Gradient norm
                    grad_norm = (
                        sum(
                            p.grad.norm().item() ** 2
                            for p in self.parameters()
                            if p.grad is not None
                        )
                        ** 0.5
                    )

                    # Step
                    optimizer.step()

                    # Log
                    ll = -neg_ll.item()
                    ll_history.append(ll)

                    if iter % 5 == 0:
                        dll = ll - ll_history[-2] if len(ll_history) > 1 else 0
                        output.write_iteration(iter, lrate, ll, grad_norm, dll, dll)

                output.write_convergence("Completed", iter, ll_history[-1], grad_norm)

        else:
            # Normal mode with tqdm
            pbar = tqdm(range(max_iter), desc="AMICA", disable=not verbose)

            for iter in pbar:
                # Zero gradients
                optimizer.zero_grad()

                # Compute loss
                neg_ll = -self.compute_log_likelihood(X_prep)

                if torch.isnan(neg_ll):
                    print(f"\nNaN at iteration {iter}")
                    break

                # Backward
                neg_ll.backward()

                # Step
                optimizer.step()

                # Update progress
                ll = -neg_ll.item()
                ll_history.append(ll)

                pbar.set_postfix(
                    {
                        "LL": f"{ll:.4f}",
                        "iter/s": f"{pbar.format_dict['rate']:.1f}"
                        if pbar.format_dict.get("rate")
                        else "N/A",
                    }
                )

        self.ll_history = ll_history
        return self

    def transform(self, X: np.ndarray, model_idx: int = 0) -> np.ndarray:
        """Transform data to sources."""
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).to(self.device, dtype=self.dtype)
            X_prep = (X_tensor - self.mean) @ self.sphere.T
            S = self._forward_single(X_prep, model_idx)
            return S.cpu().numpy()

    def get_mixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """Get mixing matrix."""
        return self.A[model_idx].detach().cpu().numpy()

    def get_unmixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """Get unmixing matrix."""
        A = self.A[model_idx]
        W = torch.linalg.pinv(A.T).T
        return W.detach().cpu().numpy()
