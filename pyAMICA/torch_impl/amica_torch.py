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
        # log|det(sphere)|, set by preprocess_data (amica17.f90 "sldet"); 0.0
        # (log|det(I)|) until preprocessing runs or when do_sphere=False.
        self.register_buffer("sldet", torch.tensor(0.0, dtype=self.dtype))

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
        Compute the per-sample-per-source log-likelihood.

        Mixture reduction is per-source, matching Fortran (amica17.f90:1313-1360):
        for each source i, logsumexp over the k mixture components using that
        source's own alpha[k, i], THEN sum over sources. Summing sources before
        the mixture logsumexp would force every source to share one mixture
        label per time point, which is not the AMICA model.

        The log|det W| Jacobian term (`log_det_W` below) and the sphering
        log-determinant (`self.sldet`) ARE both part of Fortran's reported
        LL, and so are included here unconditionally. Fortran computes
        `Dtemp(h) = log|det W(h)|` via QR of W (amica17.f90:975-980) and
        seeds `Ptmp(:, h) = Dsum(h) + log(gm(h)) + sldet` before the
        per-sample mixture-density loop even starts (amica17.f90:1273),
        i.e. before the `LL = LLtmp2 / (num_samples * nw)` normalization at
        amica17.f90:1866. `sldet` is the log|det| of the sphering/whitening
        transform applied in preprocess_data (amica17.f90's `sldet`,
        computed there as `sum(-0.5 * log(kept eigenvalues))`; see
        preprocess_data for the matching computation here). Both terms are
        also needed for training stability: nothing else in this
        Adam-over-reparameterized-tensors objective (unlike Fortran's
        natural-gradient update, which keeps A's columns normalized every
        iteration) constrains beta/A from drifting into a degenerate region
        that inflates the mixture log-density without bound.
        """
        n_samples = X.shape[1]

        # Use log-space throughout for stability
        log_model_lls = []

        for h in range(self.n_models):
            # Get sources
            Y = self._forward_single(X, h)

            # Vectorized computation over samples
            log_mix_probs = []

            for k in range(self.n_mix):
                # Get parameters (no slicing that could cause issues)
                mu_k = self.mu[k, :].unsqueeze(1)
                beta_k = self.beta[k, :].unsqueeze(1)
                rho_k = self.rho[k, :].unsqueeze(1)
                alpha_k = self.alpha[k, :].unsqueeze(1)

                # Compute log-PDF for all samples at once, per source
                log_pdf = self._compute_gg_log_pdf_vectorized(Y, mu_k, beta_k, rho_k)

                # Add this source's own mixture weight (not meaned across sources)
                log_prob = log_pdf + torch.log(alpha_k + self.eps)
                log_mix_probs.append(log_prob)

            # Combine mixture components per source: (n_mix, n_sources, n_samples)
            log_mix_probs_tensor = torch.stack(log_mix_probs, dim=0)
            # Per-source mixture reduction (logsumexp over k for each source)
            log_source_probs = torch.logsumexp(log_mix_probs_tensor, dim=0)

            # Sum over sources and samples (Fortran sums per-source LL into Ptmp(h))
            log_model_ll = log_source_probs.sum()

            # Jacobian of the unmixing transform: log|det W| = -log|det A|
            # for square invertible A (Fortran computes this directly from W
            # via QR; slogdet's log-ABSOLUTE-determinant matches Fortran's
            # own `log(abs(Wtmp(i,i)))`, unlike plain logdet which NaNs on a
            # negative determinant)
            A = self.A[h]
            _, log_abs_det_A = torch.linalg.slogdet(
                A[: self.n_sources, : self.n_sources]
            )
            log_det_W = -log_abs_det_A
            log_model_ll = log_model_ll + n_samples * log_det_W

            # Add model weight
            log_model_ll = log_model_ll + n_samples * torch.log(self.gm[h] + self.eps)
            log_model_lls.append(log_model_ll)

        # Combine models
        log_model_lls_tensor = torch.stack(log_model_lls)
        total_ll = torch.logsumexp(log_model_lls_tensor, dim=0)

        # Sphering log-determinant (constant across models/samples, so it
        # factors out of the logsumexp over h the same way it would if added
        # per-model as Fortran does at amica17.f90:1273)
        total_ll = total_ll + n_samples * self.sldet

        return total_ll / (n_samples * self.n_sources)

    def _compute_gg_log_pdf_vectorized(
        self, Y: torch.Tensor, mu: torch.Tensor, beta: torch.Tensor, rho: torch.Tensor
    ) -> torch.Tensor:
        """
        Vectorized Generalized Gaussian log-PDF.

        Y: (n_sources, n_samples)
        mu, beta, rho: (n_sources, 1)
        Returns: (n_sources, n_samples), per-source log-density (NOT summed
        over sources -- the mixture reduction over k must happen per-source
        before sources are combined; see compute_log_likelihood).
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

        # Clamp for stability
        log_p = torch.clamp(log_p, min=self.min_log)

        return log_p

    def preprocess_data(
        self, X: np.ndarray, do_mean: bool = True, do_sphere: bool = True
    ) -> torch.Tensor:
        """Preprocess data with MPS-safe operations."""
        X = torch.from_numpy(X).to(self.dtype)

        # log|det(sphere)|; only nonzero when do_sphere actually builds a
        # non-identity transform below (amica17.f90 "sldet").
        self.sldet = torch.tensor(0.0, dtype=self.dtype, device=self.device)

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
                # log|det(sphere)| = log|det(eigvecs)| + sum(-0.5*log(eigvals))
                # = sum(-0.5*log(eigvals)) since eigvecs is orthonormal
                # (amica17.f90:474 computes this same per-eigenvalue sum,
                # including in the PCA-reduced-rank case where the
                # transform isn't square and a literal determinant isn't
                # otherwise well-defined)
                self.sldet = (
                    (-0.5 * torch.log(eigvals + self.eps)).sum().to(self.device)
                )

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
                self.sldet = (-0.5 * torch.log(eigvals + self.eps)).sum()
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

                    # Compute loss (Fortran-parity LL, Jacobian included --
                    # see compute_log_likelihood docstring)
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

                # Compute loss (Fortran-parity LL, Jacobian included -- see
                # compute_log_likelihood docstring)
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
