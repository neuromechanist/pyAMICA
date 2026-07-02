"""
Main AMICA interface using PyTorch implementation.

This module provides the primary AMICA class that uses the PyTorch
implementation for GPU-accelerated ICA with adaptive mixtures.
"""

import numpy as np
from typing import Optional, Union
import json
import logging

from .torch_impl import AMICATorch, AMICATorchNG, setup_device

logger = logging.getLogger(__name__)


class AMICA:
    """
    Adaptive Mixture ICA using PyTorch backend.

    This is the main interface for pyAMICA, providing a scikit-learn style
    API while using the GPU-accelerated PyTorch implementation underneath.

    Parameters
    ----------
    n_models : int, default=1
        Number of ICA models to learn
    n_mix : int, default=3
        Number of mixture components per source
    device : str or torch.device, optional
        Device to use ('cuda', 'mps', 'cpu', or None for auto)
    verbose : bool, default=True
        Whether to show progress during fitting
    backend : {"torch", "ng"}, default="torch"
        Which underlying implementation to use. ``"torch"`` (the default)
        is ``AMICATorch``, the existing Adam/autograd-driven backend.
        ``"ng"`` is ``AMICATorchNG`` (ADR 0001), a natural-gradient EM port
        that matches the Fortran/NumPy fixed-point update rule; it is
        opt-in and experimental (see ``.context/decisions/0001-torch-backend-natural-gradient-em.md``).
        Because the two backends have different constructor/fit shapes,
        ``**kwargs`` passed to :meth:`fit` are routed differently depending
        on ``backend`` -- see :meth:`fit`.

    Attributes
    ----------
    model_ : AMICATorch
        The underlying PyTorch model
    is_fitted_ : bool
        Whether the model has been fitted
    ll_history_ : list
        Log-likelihood history during training

    Examples
    --------
    >>> from pyAMICA import AMICA
    >>> import numpy as np
    >>>
    >>> # Generate sample data
    >>> X = np.random.randn(32, 10000)  # 32 channels, 10000 samples
    >>>
    >>> # Fit AMICA model
    >>> amica = AMICA(n_models=1, n_mix=3)
    >>> amica.fit(X, max_iter=100)
    >>>
    >>> # Transform data to sources
    >>> S = amica.transform(X)
    >>>
    >>> # Get mixing matrix
    >>> A = amica.get_mixing_matrix()
    """

    def __init__(
        self,
        n_models: int = 1,
        n_mix: int = 3,
        device: Optional[Union[str, object]] = None,
        verbose: bool = True,
        backend: str = "torch",
    ):
        if backend not in ("torch", "ng"):
            raise ValueError(f"backend must be 'torch' or 'ng', got {backend!r}")

        self.n_models = n_models
        self.n_mix = n_mix
        self.device = device
        self.verbose = verbose
        self.backend = backend

        self.model_ = None
        self.is_fitted_ = False
        self.ll_history_ = []

    def fit(
        self,
        X: np.ndarray,
        max_iter: int = 100,
        lrate: float = 0.05,
        do_mean: bool = True,
        do_sphere: bool = True,
        do_newton: bool = False,
        output_dir: Optional[str] = None,
        debug: bool = False,
        **kwargs,
    ) -> "AMICA":
        """
        Fit AMICA model to data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_channels, n_samples)
        max_iter : int, default=100
            Maximum number of iterations
        lrate : float, default=0.05
            Learning rate
        do_mean : bool, default=True
            Whether to remove mean from data
        do_sphere : bool, default=True
            Whether to sphere (whiten) the data
        do_newton : bool, default=False
            Whether to use Newton optimization (experimental). Only supported
            by ``backend="torch"``; passing ``do_newton=True`` with
            ``backend="ng"`` raises, since Newton is not yet ported to the
            natural-gradient backend (Phase 4).
        output_dir : str, optional
            Directory for debug output (only used if debug=True). Only
            supported by ``backend="torch"``.
        debug : bool, default=False
            If True, use Fortran-style output; if False, use tqdm progress
            bar. Only supported by ``backend="torch"``.
        **kwargs
            For ``backend="torch"``, additional parameters passed to
            ``AMICATorch.fit()``. For ``backend="ng"``, additional
            parameters passed to the ``AMICATorchNG`` constructor instead
            (e.g. ``block_size``, ``rho0``, ``seed``, ``dtype``) -- the NG
            backend's tunables are constructor arguments, not fit() kwargs.

        Returns
        -------
        self : AMICA
            Fitted model
        """
        # Validate input
        if X.ndim != 2:
            raise ValueError(f"X must be 2D array, got shape {X.shape}")

        n_channels, n_samples = X.shape

        if self.verbose:
            print(f"Fitting AMICA with {n_channels} channels, {n_samples} samples")
            print(f"Models: {self.n_models}, Mixture components: {self.n_mix}")

        # Setup device
        if self.device is None:
            device = setup_device()
        else:
            device = self.device

        if self.backend == "ng":
            if do_newton:
                raise ValueError(
                    "do_newton=True is not supported by backend='ng' yet "
                    "(Newton is Phase 4, not ported to AMICATorchNG)."
                )
            if debug:
                raise ValueError(
                    "debug=True (Fortran-style output) is not supported by "
                    "backend='ng'."
                )
            if output_dir is not None:
                raise ValueError(
                    "output_dir is not supported by backend='ng' (no debug "
                    "output path)."
                )

            self.model_ = AMICATorchNG(
                n_channels=n_channels,
                n_models=self.n_models,
                n_mix=self.n_mix,
                lrate=lrate,
                do_mean=do_mean,
                do_sphere=do_sphere,
                device=device,
                **kwargs,
            )
            self.model_.fit(X, max_iter=max_iter, verbose=self.verbose)

            self.ll_history_ = self.model_.ll_history
            self.is_fitted_ = True

            return self

        # Create PyTorch model
        self.model_ = AMICATorch(
            n_channels=n_channels,
            n_sources=n_channels,  # Default to square mixing
            n_models=self.n_models,
            n_mix=self.n_mix,
            device=device,
        )

        # Fit model
        self.model_.fit(
            X,
            max_iter=max_iter,
            lrate=lrate,
            do_mean=do_mean,
            do_sphere=do_sphere,
            do_newton=do_newton,
            debug=debug or not self.verbose,
            output_dir=output_dir,
            **kwargs,
        )

        # Store history
        self.ll_history_ = self.model_.ll_history
        self.is_fitted_ = True

        return self

    def transform(self, X: np.ndarray, model_idx: int = 0) -> np.ndarray:
        """
        Transform data to source space.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_channels, n_samples)
        model_idx : int, default=0
            Which model to use for transformation

        Returns
        -------
        S : np.ndarray
            Sources of shape (n_sources, n_samples)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before transform")

        return self.model_.transform(X, model_idx=model_idx)

    def fit_transform(self, X: np.ndarray, **fit_params) -> np.ndarray:
        """
        Fit model and transform data.

        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_channels, n_samples)
        **fit_params
            Parameters passed to fit()

        Returns
        -------
        S : np.ndarray
            Sources of shape (n_sources, n_samples)
        """
        self.fit(X, **fit_params)
        return self.transform(X)

    def get_mixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """
        Get the mixing matrix A.

        Parameters
        ----------
        model_idx : int, default=0
            Which model's mixing matrix to return

        Returns
        -------
        A : np.ndarray
            Mixing matrix of shape (n_channels, n_sources)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting mixing matrix")

        return self.model_.get_mixing_matrix(model_idx=model_idx)

    def get_unmixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """
        Get the unmixing matrix W.

        Parameters
        ----------
        model_idx : int, default=0
            Which model's unmixing matrix to return

        Returns
        -------
        W : np.ndarray
            Unmixing matrix of shape (n_sources, n_channels)
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before getting unmixing matrix")

        return self.model_.get_unmixing_matrix(model_idx=model_idx)

    def save(self, filepath: str):
        """
        Save model to file.

        Parameters
        ----------
        filepath : str
            Path to save the model
        """
        if not self.is_fitted_:
            raise ValueError("Cannot save unfitted model")
        if self.backend == "ng":
            raise NotImplementedError(
                "save() is not implemented for backend='ng': AMICATorchNG "
                "is not an nn.Module and has no state_dict()."
            )

        import torch

        torch.save(
            {
                "model_state": self.model_.state_dict(),
                "n_models": self.n_models,
                "n_mix": self.n_mix,
                "n_channels": self.model_.n_channels,
                "n_sources": self.model_.n_sources,
                "ll_history": self.ll_history_,
            },
            filepath,
        )

        if self.verbose:
            print(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """
        Load model from file.

        Parameters
        ----------
        filepath : str
            Path to load the model from
        """
        if self.backend == "ng":
            raise NotImplementedError(
                "load() is not implemented for backend='ng': AMICATorchNG "
                "has no save()/state_dict() counterpart."
            )

        import torch

        checkpoint = torch.load(filepath, map_location="cpu")

        # Setup device
        if self.device is None:
            device = setup_device()
        else:
            device = self.device

        # Create model with saved dimensions
        self.model_ = AMICATorch(
            n_channels=checkpoint["n_channels"],
            n_sources=checkpoint["n_sources"],
            n_models=checkpoint["n_models"],
            n_mix=checkpoint["n_mix"],
            device=device,
        )

        # Load state
        self.model_.load_state_dict(checkpoint["model_state"])
        self.model_.to(device)

        # Update attributes
        self.n_models = checkpoint["n_models"]
        self.n_mix = checkpoint["n_mix"]
        self.ll_history_ = checkpoint.get("ll_history", [])
        self.is_fitted_ = True

        if self.verbose:
            print(f"Model loaded from {filepath}")

    @classmethod
    def from_params_file(cls, params_file: str, **kwargs) -> "AMICA":
        """
        Create AMICA instance from parameter file.

        Parameters
        ----------
        params_file : str
            Path to JSON parameter file
        **kwargs
            Additional parameters to override

        Returns
        -------
        amica : AMICA
            Configured AMICA instance
        """
        with open(params_file, "r") as f:
            params = json.load(f)

        # Extract relevant parameters
        n_models = params.get("num_models", 1)
        n_mix = params.get("num_mix", 3)

        # Override with kwargs
        n_models = kwargs.pop("n_models", n_models)
        n_mix = kwargs.pop("n_mix", n_mix)

        return cls(n_models=n_models, n_mix=n_mix, **kwargs)
