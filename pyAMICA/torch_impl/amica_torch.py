"""
Main PyTorch AMICA implementation.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class AMICATorch(nn.Module):
    """
    PyTorch implementation of Adaptive Mixture ICA (AMICA).
    
    This implementation leverages GPU acceleration and automatic differentiation
    to provide efficient training of AMICA models.
    
    Parameters
    ----------
    n_channels : int
        Number of input channels (data dimensions)
    n_sources : int, optional
        Number of sources to extract. If None, equals n_channels
    n_models : int, default=1
        Number of mixing models
    n_mix : int, default=3
        Number of mixture components per source
    device : str or torch.device, optional
        Device to run computations on ('cpu', 'cuda', 'mps', or torch.device)
    dtype : torch.dtype, default=torch.float32
        Data type for computations
    """
    
    def __init__(
        self,
        n_channels: int,
        n_sources: Optional[int] = None,
        n_models: int = 1,
        n_mix: int = 3,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        
        self.n_channels = n_channels
        self.n_sources = n_sources or n_channels
        self.n_models = n_models
        self.n_mix = n_mix
        self.dtype = dtype
        
        # Set up device
        if device is None:
            self.device = self._setup_device()
        else:
            self.device = torch.device(device) if isinstance(device, str) else device
            
        # Initialize parameters
        self._init_parameters()
        
        # Move to device
        self.to(self.device)
        
        # Numerical stability parameters (from Fortran)
        self.eps = 1e-10
        self.min_cond = 1e-15
        self.min_log = -1500.0
        self.max_val = 1e32
        self.min_eig = 1e-15
        
        logger.info(f"Initialized AMICATorch on {self.device}")
        
    def _setup_device(self) -> torch.device:
        """Automatically select the best available device."""
        if torch.backends.mps.is_available():
            return torch.device("mps")
        elif torch.cuda.is_available():
            return torch.device("cuda")
        else:
            return torch.device("cpu")
    
    def _init_parameters(self):
        """Initialize model parameters."""
        # Mixing matrices (one per model)
        self.A = nn.ParameterList([
            nn.Parameter(torch.eye(self.n_channels, self.n_sources, dtype=self.dtype))
            for _ in range(self.n_models)
        ])
        
        # Bias terms (one per model)
        self.c = nn.ParameterList([
            nn.Parameter(torch.zeros(self.n_channels, 1, dtype=self.dtype))
            for _ in range(self.n_models)
        ])
        
        # Model weights
        self.gm = nn.Parameter(
            torch.ones(self.n_models, dtype=self.dtype) / self.n_models
        )
        
        # Mixture parameters (shared across models)
        self.alpha = nn.Parameter(
            torch.ones(self.n_mix, self.n_sources, dtype=self.dtype) / self.n_mix
        )
        self.mu = nn.Parameter(
            torch.zeros(self.n_mix, self.n_sources, dtype=self.dtype)
        )
        self.beta = nn.Parameter(
            torch.ones(self.n_mix, self.n_sources, dtype=self.dtype)
        )
        self.rho = nn.Parameter(
            1.5 * torch.ones(self.n_mix, self.n_sources, dtype=self.dtype)
        )
        
        # Preprocessing parameters (not trainable)
        self.register_buffer('mean', torch.zeros(self.n_channels, 1, dtype=self.dtype))
        self.register_buffer('sphere', torch.eye(self.n_channels, dtype=self.dtype))
        self.register_buffer('sphere_inv', torch.eye(self.n_channels, dtype=self.dtype))
        
        # Unmixing matrices (computed from A)
        self.W = [None] * self.n_models
        
    def forward(
        self, 
        X: torch.Tensor, 
        model_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Forward pass through the mixing model.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_channels, n_samples)
        model_idx : int, optional
            Index of model to use. If None, uses all models
            
        Returns
        -------
        Y : torch.Tensor
            Estimated sources of shape (n_sources, n_samples) or
            (n_models, n_sources, n_samples) if model_idx is None
        """
        # Ensure X is on the correct device
        X = X.to(self.device, dtype=self.dtype)
        
        if model_idx is not None:
            # Single model
            Y = self._forward_single(X, model_idx)
        else:
            # All models
            Y = torch.stack([
                self._forward_single(X, h) 
                for h in range(self.n_models)
            ])
            
        return Y
    
    def _forward_single(self, X: torch.Tensor, model_idx: int) -> torch.Tensor:
        """Forward pass through a single model."""
        A = self.A[model_idx]
        c = self.c[model_idx]
        
        # Compute unmixing matrix (W = inv(A))
        W = torch.linalg.pinv(A.T).T  # Transpose for correct dimensions
        
        # Apply unmixing: Y = W @ (X - c)
        Y = W @ (X - c)
        
        return Y
    
    def compute_log_likelihood(
        self, 
        X: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute the log-likelihood of the data.
        
        Parameters
        ----------
        X : torch.Tensor
            Input data of shape (n_channels, n_samples)
        return_components : bool, default=False
            If True, return per-model and per-sample log-likelihoods
            
        Returns
        -------
        ll : torch.Tensor
            Scalar log-likelihood, or detailed components if return_components=True
        """
        X = X.to(self.device, dtype=self.dtype)
        n_samples = X.shape[1]
        
        # Store per-model log-likelihoods
        model_lls = []
        
        for h in range(self.n_models):
            # Get sources for this model
            Y = self._forward_single(X, h)  # (n_sources, n_samples)
            
            # Compute log-determinant of W (= -log|det(A)|)
            A = self.A[h]
            log_det_W = -torch.logdet(A[:self.n_sources, :self.n_sources])
            
            # Compute mixture log-likelihood for each sample
            sample_lls = []
            for t in range(n_samples):
                y_t = Y[:, t:t+1]  # (n_sources, 1)
                
                # Compute log-probability under each mixture component
                mix_lls = []
                for k in range(self.n_mix):
                    # Generalized Gaussian log-PDF
                    log_p = self._compute_gg_log_pdf(
                        y_t, 
                        self.mu[k:k+1, :].T,  # (n_sources, 1)
                        self.beta[k:k+1, :].T,  # (n_sources, 1)
                        self.rho[k:k+1, :].T   # (n_sources, 1)
                    )
                    # Add mixture weight
                    log_p = log_p.sum() + torch.log(self.alpha[k, :].mean())
                    mix_lls.append(log_p)
                
                # Log-sum-exp for numerical stability
                sample_ll = torch.logsumexp(torch.stack(mix_lls), dim=0)
                sample_ll += log_det_W  # Add Jacobian
                sample_lls.append(sample_ll)
            
            # Model log-likelihood
            model_ll = torch.stack(sample_lls).sum()
            model_ll += n_samples * torch.log(self.gm[h])  # Model weight
            model_lls.append(model_ll)
        
        # Total log-likelihood
        total_ll = torch.logsumexp(torch.stack(model_lls), dim=0)
        
        if return_components:
            return total_ll, torch.stack(model_lls)
        else:
            return total_ll / n_samples  # Normalize
    
    def _compute_gg_log_pdf(
        self,
        y: torch.Tensor,
        mu: torch.Tensor,
        beta: torch.Tensor,
        rho: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Generalized Gaussian log-PDF.
        
        Parameters
        ----------
        y : torch.Tensor
            Data points
        mu : torch.Tensor
            Means
        beta : torch.Tensor
            Scale parameters
        rho : torch.Tensor
            Shape parameters
            
        Returns
        -------
        log_p : torch.Tensor
            Log-probabilities
        """
        # Ensure numerical stability
        beta = torch.clamp(beta, min=self.eps)
        rho = torch.clamp(rho, min=1.0, max=2.0)
        
        # Compute normalized distance
        diff = torch.abs(y - mu) / beta
        
        # Generalized Gaussian log-PDF
        log_p = -torch.pow(diff + self.eps, rho)
        log_p += torch.log(rho) - torch.log(2 * beta) - torch.lgamma(1/rho)
        
        # Clamp to avoid numerical issues
        log_p = torch.clamp(log_p, min=self.min_log)
        
        return log_p
    
    def preprocess_data(
        self,
        X: np.ndarray,
        do_mean: bool = True,
        do_sphere: bool = True,
        do_pca: bool = False,
        pca_keep: Optional[int] = None
    ) -> torch.Tensor:
        """
        Preprocess data (mean removal, sphering, PCA).
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_channels, n_samples)
        do_mean : bool, default=True
            Remove mean
        do_sphere : bool, default=True
            Apply sphering (whitening)
        do_pca : bool, default=False
            Apply PCA dimension reduction
        pca_keep : int, optional
            Number of components to keep if do_pca=True
            
        Returns
        -------
        X_prep : torch.Tensor
            Preprocessed data
        """
        X = torch.from_numpy(X).to(self.device, dtype=self.dtype)
        
        if do_mean:
            self.mean = X.mean(dim=1, keepdim=True)
            X = X - self.mean
            
        if do_sphere or do_pca:
            # Compute covariance
            cov = torch.cov(X)
            
            # Eigendecomposition
            eigvals, eigvecs = torch.linalg.eigh(cov)
            
            # Sort in descending order
            idx = torch.argsort(eigvals, descending=True)
            eigvals = eigvals[idx]
            eigvecs = eigvecs[:, idx]
            
            # Remove small eigenvalues
            keep = eigvals > self.min_eig
            if do_pca and pca_keep is not None:
                keep = keep[:pca_keep]
                
            eigvals = eigvals[keep]
            eigvecs = eigvecs[:, keep]
            
            # Compute sphering matrix
            self.sphere = eigvecs @ torch.diag(1.0 / torch.sqrt(eigvals + self.eps))
            self.sphere_inv = torch.diag(torch.sqrt(eigvals)) @ eigvecs.T
            
            # Apply sphering
            X = self.sphere.T @ X
            
        return X
    
    def fit(
        self,
        X: np.ndarray,
        max_iter: int = 100,
        lrate: float = 0.1,
        do_newton: bool = True,
        newton_start: int = 20,
        verbose: bool = True,
        **kwargs
    ) -> 'AMICATorch':
        """
        Fit the AMICA model to data.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_channels, n_samples)
        max_iter : int, default=100
            Maximum number of iterations
        lrate : float, default=0.1
            Learning rate
        do_newton : bool, default=True
            Use Newton optimization after warm-up
        newton_start : int, default=20
            Iteration to start Newton optimization
        verbose : bool, default=True
            Print progress
        **kwargs
            Additional optimizer arguments
            
        Returns
        -------
        self : AMICATorch
            Fitted model
        """
        # Preprocess data
        X_prep = self.preprocess_data(X, **kwargs)
        
        # Initialize optimizers
        from .optimizers import NaturalGradientOptimizer, NewtonOptimizer
        
        nat_grad_opt = NaturalGradientOptimizer(
            self.parameters(), 
            lr=lrate
        )
        
        if do_newton:
            newton_opt = NewtonOptimizer(self)
        
        # Training loop
        ll_history = []
        
        for iter in range(max_iter):
            # Natural gradient step
            nat_grad_opt.zero_grad()
            
            # Compute negative log-likelihood (to minimize)
            neg_ll = -self.compute_log_likelihood(X_prep)
            neg_ll.backward()
            
            # Check for NaN
            if torch.isnan(neg_ll):
                logger.warning(f"NaN encountered at iteration {iter}")
                break
                
            nat_grad_opt.step()
            
            # Newton step after warm-up
            if do_newton and iter >= newton_start:
                newton_opt.step(X_prep)
            
            # Log progress
            with torch.no_grad():
                ll = -neg_ll.item()
                ll_history.append(ll)
                
                if verbose and iter % 10 == 0:
                    logger.info(f"Iter {iter:4d}: LL = {ll:.6f}")
        
        self.ll_history = ll_history
        return self
    
    def transform(
        self, 
        X: np.ndarray,
        model_idx: int = 0
    ) -> np.ndarray:
        """
        Transform data to sources using fitted model.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_channels, n_samples)
        model_idx : int, default=0
            Model index to use for transformation
            
        Returns
        -------
        S : np.ndarray
            Estimated sources of shape (n_sources, n_samples)
        """
        with torch.no_grad():
            # Preprocess
            X_tensor = torch.from_numpy(X).to(self.device, dtype=self.dtype)
            X_prep = (X_tensor - self.mean) @ self.sphere.T
            
            # Get sources
            S = self._forward_single(X_prep, model_idx)
            
            return S.cpu().numpy()
    
    def get_mixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """Get mixing matrix A for specified model."""
        return self.A[model_idx].detach().cpu().numpy()
    
    def get_unmixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """Get unmixing matrix W for specified model."""
        A = self.A[model_idx]
        W = torch.linalg.pinv(A.T).T
        return W.detach().cpu().numpy()
    
    def save(self, filepath: str):
        """Save model to file."""
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'n_channels': self.n_channels,
                'n_sources': self.n_sources,
                'n_models': self.n_models,
                'n_mix': self.n_mix,
                'dtype': self.dtype
            },
            'll_history': getattr(self, 'll_history', [])
        }, filepath)
        
    @classmethod
    def load(cls, filepath: str, device: Optional[torch.device] = None) -> 'AMICATorch':
        """Load model from file."""
        checkpoint = torch.load(filepath, map_location='cpu')
        
        # Create model
        model = cls(
            **checkpoint['config'],
            device=device
        )
        
        # Load weights
        model.load_state_dict(checkpoint['state_dict'])
        model.ll_history = checkpoint.get('ll_history', [])
        
        return model