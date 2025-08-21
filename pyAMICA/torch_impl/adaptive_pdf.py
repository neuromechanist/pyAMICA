"""
Adaptive PDF selection and implementation for PyTorch AMICA.

This module provides multiple probability density functions and automatic
selection based on source statistics, matching the Fortran implementation.
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional, List
import numpy as np


class AdaptivePDF(nn.Module):
    """
    Adaptive probability density functions for AMICA sources.
    
    Supports multiple PDF types and automatic selection based on
    source statistics (kurtosis, skewness).
    
    PDF Types:
    1. Generalized Gaussian: p(x) ∝ exp(-|x|^ρ)
    2. Logistic: p(x) = exp(x)/(1+exp(x))^2
    3. Student-t: Heavy-tailed distribution
    4. Laplace: p(x) ∝ exp(-|x|)
    5. Uniform: For bounded sources
    
    Parameters
    ----------
    n_sources : int
        Number of sources
    n_mix : int
        Number of mixture components
    adaptive : bool, default=True
        Whether to adaptively select PDF types
    initial_pdf : str, default='gg'
        Initial PDF type ('gg', 'logistic', 'student', 'laplace')
    """
    
    def __init__(
        self,
        n_sources: int,
        n_mix: int = 3,
        adaptive: bool = True,
        initial_pdf: str = 'gg',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.n_sources = n_sources
        self.n_mix = n_mix
        self.adaptive = adaptive
        self.device = device or torch.device('cpu')
        self.dtype = dtype
        
        # PDF type for each source (can be different)
        self.pdf_types = ['gg'] * n_sources  # Start with generalized Gaussian
        
        # Shape parameters for each PDF type
        self.register_buffer(
            'rho',
            torch.ones(n_mix, n_sources, dtype=dtype, device=device) * 1.5
        )
        
        # Degrees of freedom for Student-t
        self.register_buffer(
            'nu',
            torch.ones(n_mix, n_sources, dtype=dtype, device=device) * 5.0
        )
        
        # Statistics for adaptive selection
        self.register_buffer(
            'kurtosis',
            torch.zeros(n_sources, dtype=dtype, device=device)
        )
        self.register_buffer(
            'skewness', 
            torch.zeros(n_sources, dtype=dtype, device=device)
        )
        
        # Iteration counter for adaptation
        self.adapt_iter = 0
        self.adapt_interval = 10  # Check every N iterations
    
    def forward(
        self,
        Y: torch.Tensor,
        mu: torch.Tensor,
        beta: torch.Tensor,
        compute_deriv: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Compute PDF values and optionally derivatives.
        
        Parameters
        ----------
        Y : torch.Tensor
            Sources (n_sources, n_samples) or (n_mix, n_sources, n_samples)
        mu : torch.Tensor
            Means (n_mix, n_sources)
        beta : torch.Tensor
            Scales (n_mix, n_sources)
        compute_deriv : bool
            Whether to compute derivatives
            
        Returns
        -------
        log_pdf : torch.Tensor
            Log PDF values
        dlog_pdf : torch.Tensor or None
            Log PDF derivatives (if compute_deriv=True)
        """
        # Normalize sources
        Y_norm = (Y - mu.unsqueeze(-1)) / (beta.unsqueeze(-1) + 1e-10)
        
        # Initialize outputs
        log_pdf = torch.zeros_like(Y_norm)
        dlog_pdf = torch.zeros_like(Y_norm) if compute_deriv else None
        
        # Compute PDF for each source based on its type
        for i in range(self.n_sources):
            pdf_type = self.pdf_types[i]
            
            # Extract data for this source
            if Y_norm.dim() == 3:
                # (n_mix, n_sources, n_samples)
                y_i = Y_norm[:, i, :]  # (n_mix, n_samples)
                rho_i = self.rho[:, i]  # (n_mix,)
            else:
                # (n_sources, n_samples)
                y_i = Y_norm[i, :].unsqueeze(0)  # (1, n_samples)
                rho_i = self.rho[:, i].mean().unsqueeze(0)  # (1,)
            
            if pdf_type == 'gg':
                log_p, dlog_p = self._generalized_gaussian(
                    y_i,
                    rho_i,
                    compute_deriv
                )
            elif pdf_type == 'logistic':
                log_p, dlog_p = self._logistic(y_i, compute_deriv)
            elif pdf_type == 'student':
                nu_i = self.nu[:, i] if Y_norm.dim() == 3 else self.nu[:, i].mean().unsqueeze(0)
                log_p, dlog_p = self._student_t(y_i, nu_i, compute_deriv)
            elif pdf_type == 'laplace':
                log_p, dlog_p = self._laplace(y_i, compute_deriv)
            else:  # uniform
                log_p, dlog_p = self._uniform(y_i, compute_deriv)
            
            if Y_norm.dim() == 3:
                log_pdf[:, i, :] = log_p
                if compute_deriv:
                    dlog_pdf[:, i, :] = dlog_p
            else:
                log_pdf[i, :] = log_p.squeeze(0)
                if compute_deriv:
                    dlog_pdf[i, :] = dlog_p.squeeze(0)
        
        # Scale derivatives by beta
        if compute_deriv:
            dlog_pdf = dlog_pdf / (beta.unsqueeze(-1) + 1e-10)
        
        return log_pdf, dlog_pdf
    
    def _generalized_gaussian(
        self,
        y: torch.Tensor,
        rho: torch.Tensor,
        compute_deriv: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Generalized Gaussian PDF."""
        eps = 1e-10
        
        # Ensure rho is in valid range [1, 2]
        rho = torch.clamp(rho, 1.0, 2.0)
        
        if rho.numel() == 1 and abs(rho.item() - 2.0) < 0.01:
            # Gaussian case (ρ = 2)
            log_pdf = -0.5 * y**2 - 0.5 * torch.log(2 * torch.tensor(np.pi))
            if compute_deriv:
                dlog_pdf = -y
            else:
                dlog_pdf = None
                
        elif rho.numel() == 1 and abs(rho.item() - 1.0) < 0.01:
            # Laplace case (ρ = 1)
            log_pdf = -torch.abs(y) - torch.log(torch.tensor(2.0))
            if compute_deriv:
                dlog_pdf = -torch.sign(y)
            else:
                dlog_pdf = None
        else:
            # General case
            abs_y = torch.abs(y) + eps
            # Handle broadcasting for rho
            if rho.dim() > 0 and rho.shape[0] > 1:
                # rho is (n_mix,) and y is (n_mix, n_samples) or (n_samples,)
                if y.dim() == 2:
                    log_pdf = -torch.pow(abs_y, rho.unsqueeze(1))
                    log_norm = torch.lgamma(1.0 + 1.0/rho.unsqueeze(1)) + torch.log(torch.tensor(2.0))
                else:
                    log_pdf = -torch.pow(abs_y, rho.mean())
                    log_norm = torch.lgamma(1.0 + 1.0/rho.mean()) + torch.log(torch.tensor(2.0))
            else:
                # Scalar rho
                rho_val = rho.item() if rho.numel() == 1 else rho
                log_pdf = -torch.pow(abs_y, rho_val)
                log_norm = torch.lgamma(1.0 + 1.0/rho_val) + torch.log(torch.tensor(2.0))
            
            # Add normalization constant
            log_pdf = log_pdf - log_norm
            
            if compute_deriv:
                if rho.dim() > 0 and rho.shape[0] > 1:
                    if y.dim() == 2:
                        dlog_pdf = -rho.unsqueeze(1) * torch.pow(abs_y, rho.unsqueeze(1) - 1) * torch.sign(y)
                    else:
                        dlog_pdf = -rho.mean() * torch.pow(abs_y, rho.mean() - 1) * torch.sign(y)
                else:
                    rho_val = rho.item() if rho.numel() == 1 else rho
                    dlog_pdf = -rho_val * torch.pow(abs_y, rho_val - 1) * torch.sign(y)
            else:
                dlog_pdf = None
        
        return log_pdf, dlog_pdf
    
    def _logistic(
        self,
        y: torch.Tensor,
        compute_deriv: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Logistic PDF."""
        # For numerical stability
        y_safe = torch.clamp(y, -10, 10)
        exp_y = torch.exp(y_safe)
        
        log_pdf = y_safe - 2 * torch.log(1 + exp_y)
        
        if compute_deriv:
            dlog_pdf = 1 - 2 * exp_y / (1 + exp_y)
        else:
            dlog_pdf = None
        
        return log_pdf, dlog_pdf
    
    def _student_t(
        self,
        y: torch.Tensor,
        nu: torch.Tensor,
        compute_deriv: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Student-t PDF."""
        # Ensure nu > 2 for finite variance
        nu = torch.clamp(nu, 2.1, 30.0)
        
        # Handle broadcasting
        if nu.dim() > 0 and nu.shape[0] > 1 and y.dim() == 2:
            # nu is (n_mix,) and y is (n_mix, n_samples)
            nu = nu.unsqueeze(1)  # (n_mix, 1)
        
        # Log PDF of Student-t
        log_pdf = -0.5 * (nu + 1) * torch.log(1 + y**2 / nu)
        # Add normalization
        log_pdf = log_pdf + torch.lgamma((nu + 1) / 2) - torch.lgamma(nu / 2)
        log_pdf = log_pdf - 0.5 * torch.log(nu * torch.tensor(np.pi))
        
        if compute_deriv:
            dlog_pdf = -(nu + 1) * y / (nu + y**2)
        else:
            dlog_pdf = None
        
        return log_pdf, dlog_pdf
    
    def _laplace(
        self,
        y: torch.Tensor,
        compute_deriv: bool
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Laplace PDF."""
        log_pdf = -torch.abs(y) - torch.log(torch.tensor(2.0))
        
        if compute_deriv:
            dlog_pdf = -torch.sign(y)
        else:
            dlog_pdf = None
        
        return log_pdf, dlog_pdf
    
    def _uniform(
        self,
        y: torch.Tensor,
        compute_deriv: bool,
        bound: float = 3.0
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Uniform PDF (for bounded sources)."""
        # Soft boundaries using sigmoid
        in_bounds = torch.sigmoid(10 * (bound - torch.abs(y)))
        log_pdf = torch.log(in_bounds / (2 * bound) + 1e-10)
        
        if compute_deriv:
            # Derivative of soft uniform
            dlog_pdf = -10 * torch.sign(y) * torch.sigmoid(10 * (torch.abs(y) - bound))
        else:
            dlog_pdf = None
        
        return log_pdf, dlog_pdf
    
    def update_statistics(self, Y: torch.Tensor):
        """
        Update source statistics for adaptive PDF selection.
        
        Parameters
        ----------
        Y : torch.Tensor
            Sources (n_sources, n_samples)
        """
        with torch.no_grad():
            # Standardize sources
            Y_std = (Y - Y.mean(dim=1, keepdim=True)) / (Y.std(dim=1, keepdim=True) + 1e-10)
            
            # Compute kurtosis (4th moment - 3)
            self.kurtosis = torch.mean(Y_std**4, dim=1) - 3.0
            
            # Compute skewness (3rd moment)
            self.skewness = torch.mean(Y_std**3, dim=1)
            
            # Update iteration counter
            self.adapt_iter += 1
            
            # Adapt PDF types based on statistics
            if self.adaptive and self.adapt_iter % self.adapt_interval == 0:
                self._adapt_pdf_types()
    
    def _adapt_pdf_types(self):
        """
        Adaptively select PDF types based on source statistics.
        
        Selection criteria:
        - High kurtosis (>3): Heavy-tailed → Student-t or Laplace
        - Low kurtosis (<0): Light-tailed → Gaussian
        - High skewness (>1): Asymmetric → Logistic
        - Moderate: Generalized Gaussian
        """
        with torch.no_grad():
            for i in range(self.n_sources):
                kurt = self.kurtosis[i].item()
                skew = abs(self.skewness[i].item())
                
                if kurt > 5:
                    # Very heavy tailed
                    self.pdf_types[i] = 'student'
                    self.nu[:, i] = 3.0  # Heavy tails
                elif kurt > 3:
                    # Heavy tailed
                    self.pdf_types[i] = 'laplace'
                elif kurt < -0.5:
                    # Light tailed (sub-Gaussian)
                    self.pdf_types[i] = 'gg'
                    self.rho[:, i] = 2.0  # Gaussian
                elif skew > 1.5:
                    # Highly asymmetric
                    self.pdf_types[i] = 'logistic'
                else:
                    # Default: Generalized Gaussian
                    self.pdf_types[i] = 'gg'
                    # Adjust rho based on kurtosis
                    if kurt > 0:
                        self.rho[:, i] = 1.5 - 0.1 * min(kurt, 3)  # More Laplace-like
                    else:
                        self.rho[:, i] = 1.5 + 0.1 * min(-kurt, 3)  # More Gaussian-like
    
    def get_pdf_info(self) -> List[str]:
        """Get current PDF types for each source."""
        return self.pdf_types.copy()