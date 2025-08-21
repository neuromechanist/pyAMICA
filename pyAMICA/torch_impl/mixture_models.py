"""
Mixture model components for PyTorch AMICA.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class GaussianMixtureICA(nn.Module):
    """
    Gaussian Mixture Model for ICA sources.
    
    This module implements the mixture model component of AMICA,
    where each source is modeled as a mixture of generalized Gaussians.
    
    Parameters
    ----------
    n_sources : int
        Number of ICA sources
    n_mix : int, default=3
        Number of mixture components per source
    pdf_type : str, default='gg'
        PDF type ('gg' for generalized Gaussian, 'laplace', 'gaussian')
    device : torch.device, optional
        Device for computations
    dtype : torch.dtype, default=torch.float32
        Data type for computations
    """
    
    def __init__(
        self,
        n_sources: int,
        n_mix: int = 3,
        pdf_type: str = 'gg',
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.n_sources = n_sources
        self.n_mix = n_mix
        self.pdf_type = pdf_type
        self.dtype = dtype
        self.device = device or torch.device('cpu')
        
        # Initialize mixture parameters
        self._init_parameters()
        
        # Numerical stability
        self.eps = 1e-10
        self.min_log = -1500.0
        
        self.to(self.device)
        
    def _init_parameters(self):
        """Initialize mixture model parameters."""
        # Mixture weights (alpha)
        self.log_alpha = nn.Parameter(
            torch.zeros(self.n_mix, self.n_sources, dtype=self.dtype)
        )
        
        # Component means (mu)
        self.mu = nn.Parameter(
            torch.zeros(self.n_mix, self.n_sources, dtype=self.dtype)
        )
        
        # Scale parameters (beta)
        self.log_beta = nn.Parameter(
            torch.zeros(self.n_mix, self.n_sources, dtype=self.dtype)
        )
        
        # Shape parameters (rho) - only for generalized Gaussian
        if self.pdf_type == 'gg':
            self.rho_logit = nn.Parameter(
                torch.zeros(self.n_mix, self.n_sources, dtype=self.dtype)
            )
            
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
        if self.pdf_type == 'gg':
            # Map from (-inf, inf) to (1, 2)
            return 1.0 + torch.sigmoid(self.rho_logit)
        elif self.pdf_type == 'laplace':
            return torch.ones_like(self.log_beta)
        else:  # gaussian
            return 2.0 * torch.ones_like(self.log_beta)
    
    def forward(
        self, 
        Y: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute log-likelihood of sources under mixture model.
        
        Parameters
        ----------
        Y : torch.Tensor
            Source signals of shape (n_sources, n_samples)
        return_components : bool, default=False
            If True, return per-component log-likelihoods
            
        Returns
        -------
        log_p : torch.Tensor
            Log-likelihood of sources
        """
        n_sources, n_samples = Y.shape
        
        # Compute log-probability for each mixture component
        log_probs = []
        
        for k in range(self.n_mix):
            # Get parameters for this component
            mu_k = self.mu[k].unsqueeze(1)  # (n_sources, 1)
            beta_k = self.beta[k].unsqueeze(1)  # (n_sources, 1)
            rho_k = self.rho[k].unsqueeze(1)  # (n_sources, 1)
            alpha_k = self.alpha[k].unsqueeze(1)  # (n_sources, 1)
            
            # Compute log-PDF
            if self.pdf_type == 'gg':
                log_pdf = self._generalized_gaussian_log_pdf(
                    Y, mu_k, beta_k, rho_k
                )
            elif self.pdf_type == 'laplace':
                log_pdf = self._laplace_log_pdf(Y, mu_k, beta_k)
            elif self.pdf_type == 'gaussian':
                log_pdf = self._gaussian_log_pdf(Y, mu_k, beta_k)
            else:
                raise ValueError(f"Unknown PDF type: {self.pdf_type}")
            
            # Add mixture weight
            log_prob = log_pdf + torch.log(alpha_k)
            log_probs.append(log_prob)
        
        # Stack and combine using log-sum-exp
        log_probs_stacked = torch.stack(log_probs, dim=0)  # (n_mix, n_sources, n_samples)
        
        # Sum over sources (independence assumption)
        log_p_sources = torch.logsumexp(log_probs_stacked, dim=0)  # (n_sources, n_samples)
        log_p = log_p_sources.sum(dim=0)  # (n_samples,)
        
        if return_components:
            return log_p, log_probs_stacked
        else:
            return log_p
    
    def _generalized_gaussian_log_pdf(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        beta: torch.Tensor,
        rho: torch.Tensor
    ) -> torch.Tensor:
        """
        Generalized Gaussian log-PDF.
        
        p(y) = (rho / (2 * beta * Gamma(1/rho))) * exp(-(|y - mu| / beta)^rho)
        """
        # Ensure numerical stability
        beta = torch.clamp(beta, min=self.eps)
        rho = torch.clamp(rho, min=1.0, max=2.0)
        
        # Compute normalized distance
        diff = torch.abs(y - mu) / beta
        
        # Log-PDF
        log_p = -torch.pow(diff + self.eps, rho)
        log_p += torch.log(rho) - torch.log(2 * beta) - torch.lgamma(1/rho)
        
        # Clamp to avoid numerical issues
        log_p = torch.clamp(log_p, min=self.min_log)
        
        return log_p
    
    def _laplace_log_pdf(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Laplace (double exponential) log-PDF.
        
        p(y) = (1 / (2 * beta)) * exp(-|y - mu| / beta)
        """
        beta = torch.clamp(beta, min=self.eps)
        
        # Log-PDF
        log_p = -torch.abs(y - mu) / beta - torch.log(2 * beta)
        log_p = torch.clamp(log_p, min=self.min_log)
        
        return log_p
    
    def _gaussian_log_pdf(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        beta: torch.Tensor
    ) -> torch.Tensor:
        """
        Gaussian log-PDF (parameterized with beta as std dev).
        
        p(y) = (1 / (beta * sqrt(2*pi))) * exp(-0.5 * ((y - mu) / beta)^2)
        """
        beta = torch.clamp(beta, min=self.eps)
        
        # Log-PDF
        log_p = -0.5 * torch.pow((y - mu) / beta, 2)
        log_p -= torch.log(beta) + 0.5 * torch.log(2 * torch.tensor(torch.pi))
        log_p = torch.clamp(log_p, min=self.min_log)
        
        return log_p
    
    def fit_em(
        self,
        Y: torch.Tensor,
        max_iter: int = 100,
        tol: float = 1e-6,
        verbose: bool = False
    ):
        """
        Fit mixture model using Expectation-Maximization.
        
        Parameters
        ----------
        Y : torch.Tensor
            Source signals of shape (n_sources, n_samples)
        max_iter : int, default=100
            Maximum EM iterations
        tol : float, default=1e-6
            Convergence tolerance
        verbose : bool, default=False
            Print progress
        """
        n_sources, n_samples = Y.shape
        prev_ll = -float('inf')
        
        for iter in range(max_iter):
            # E-step: Compute responsibilities
            with torch.no_grad():
                log_p, log_probs_components = self.forward(Y, return_components=True)
                
                # Responsibilities: p(k|y)
                log_resp = log_probs_components - log_p.unsqueeze(0).unsqueeze(0)
                resp = torch.exp(log_resp)  # (n_mix, n_sources, n_samples)
                
            # M-step: Update parameters
            with torch.no_grad():
                for j in range(n_sources):
                    for k in range(self.n_mix):
                        resp_kj = resp[k, j, :]  # (n_samples,)
                        resp_sum = resp_kj.sum() + self.eps
                        
                        # Update mean
                        self.mu.data[k, j] = (resp_kj * Y[j, :]).sum() / resp_sum
                        
                        # Update scale (beta)
                        if self.pdf_type == 'gaussian':
                            # For Gaussian, beta is std dev
                            diff_sq = torch.pow(Y[j, :] - self.mu[k, j], 2)
                            variance = (resp_kj * diff_sq).sum() / resp_sum
                            self.log_beta.data[k, j] = 0.5 * torch.log(variance + self.eps)
                        else:
                            # For Laplace/GG, use MAD estimator
                            diff_abs = torch.abs(Y[j, :] - self.mu[k, j])
                            scale = (resp_kj * diff_abs).sum() / resp_sum
                            self.log_beta.data[k, j] = torch.log(scale + self.eps)
                        
                        # Update mixture weight
                        self.log_alpha.data[k, j] = torch.log(resp_sum / n_samples)
                        
                        # Update shape (rho) for GG - use method of moments
                        if self.pdf_type == 'gg':
                            # Estimate rho using kurtosis
                            m2 = (resp_kj * torch.pow(diff_abs, 2)).sum() / resp_sum
                            m4 = (resp_kj * torch.pow(diff_abs, 4)).sum() / resp_sum
                            if m2 > self.eps:
                                kurtosis = m4 / (m2 ** 2)
                                # Map kurtosis to rho (approximate)
                                rho_est = 2.0 / (1.0 + torch.log(kurtosis + 1))
                                rho_est = torch.clamp(rho_est, min=1.0, max=2.0)
                                # Convert to logit
                                self.rho_logit.data[k, j] = torch.logit(rho_est - 1.0)
            
            # Check convergence
            with torch.no_grad():
                ll = self.forward(Y).mean().item()
                if verbose and iter % 10 == 0:
                    logger.info(f"EM iter {iter}: LL = {ll:.6f}")
                    
                if abs(ll - prev_ll) < tol:
                    if verbose:
                        logger.info(f"EM converged after {iter} iterations")
                    break
                    
                prev_ll = ll
    
    def sample(
        self,
        n_samples: int,
        model_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample from the mixture model.
        
        Parameters
        ----------
        n_samples : int
            Number of samples to generate
        model_idx : int, optional
            If specified, sample from a specific source
            
        Returns
        -------
        samples : torch.Tensor
            Generated samples of shape (n_sources, n_samples) or (n_samples,) if model_idx specified
        """
        if model_idx is not None:
            # Sample from specific source
            samples = self._sample_source(model_idx, n_samples)
        else:
            # Sample from all sources
            samples = []
            for j in range(self.n_sources):
                samples.append(self._sample_source(j, n_samples))
            samples = torch.stack(samples)
            
        return samples
    
    def _sample_source(self, source_idx: int, n_samples: int) -> torch.Tensor:
        """Sample from a specific source."""
        # Sample mixture components
        alpha = self.alpha[:, source_idx]
        component_idx = torch.multinomial(alpha, n_samples, replacement=True)
        
        # Sample from components
        samples = torch.zeros(n_samples, device=self.device, dtype=self.dtype)
        
        for k in range(self.n_mix):
            mask = (component_idx == k)
            n_k = mask.sum().item()
            
            if n_k > 0:
                mu_k = self.mu[k, source_idx]
                beta_k = self.beta[k, source_idx]
                
                if self.pdf_type == 'gaussian':
                    # Sample from Gaussian
                    samples[mask] = mu_k + beta_k * torch.randn(n_k, device=self.device, dtype=self.dtype)
                elif self.pdf_type == 'laplace':
                    # Sample from Laplace
                    u = torch.rand(n_k, device=self.device, dtype=self.dtype) - 0.5
                    samples[mask] = mu_k - beta_k * torch.sign(u) * torch.log(1 - 2 * torch.abs(u))
                elif self.pdf_type == 'gg':
                    # Sample from Generalized Gaussian (using rejection sampling)
                    rho_k = self.rho[k, source_idx]
                    # Simplified: use Gaussian approximation for now
                    samples[mask] = mu_k + beta_k * torch.randn(n_k, device=self.device, dtype=self.dtype)
                    
        return samples