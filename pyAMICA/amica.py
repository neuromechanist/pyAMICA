"""
Main AMICA interface using the PyTorch natural-gradient EM backend.

This module provides the primary :class:`AMICA` class, a scikit-learn-style
wrapper over :class:`AMICATorchNG` (the natural-gradient EM port that reaches
Fortran parity; see ``.context/decisions/0001-torch-backend-natural-gradient-em.md``).
"""

import numpy as np
import torch
from typing import Optional, Union
import inspect
import logging

from .torch_impl import AMICATorchNG, setup_device

logger = logging.getLogger(__name__)

# AMICATorchNG's default parameter dtype, derived from its signature so the
# wrapper's MPS/float64 fallback below stays in lockstep if that default ever
# changes (rather than duplicating the literal).
_NG_DEFAULT_DTYPE = inspect.signature(AMICATorchNG).parameters["dtype"].default


class AMICA:
    """
    Adaptive Mixture ICA using the PyTorch natural-gradient EM backend.

    This is the main interface for pyAMICA, providing a scikit-learn style
    API over :class:`AMICATorchNG`, the natural-gradient EM implementation
    that matches the Fortran reference (Newton, exact-EM mixture updates,
    symmetric-ZCA sphere, Jacobian LL).

    Parameters
    ----------
    n_models : int, default=1
        Number of ICA models to learn
    n_mix : int, default=3
        Number of mixture components per source
    device : str or torch.device, optional
        Device to use ('cuda', 'mps', 'cpu', or None for auto). With ``None``
        (auto), an auto-selected MPS device is redirected to CPU because the
        backend computes in float64 for Fortran parity and MPS cannot
        represent it; pass ``dtype=torch.float32`` (with ``device="mps"``) to
        run on MPS instead.
    verbose : bool, default=True
        Whether to show progress during fitting

    Attributes
    ----------
    model_ : AMICATorchNG
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
    ):
        self.n_models = n_models
        self.n_mix = n_mix
        self.device = device
        self.verbose = verbose

        self.model_ = None
        self.is_fitted_ = False
        self.ll_history_ = []

    def _select_device(self, ng_dtype) -> object:
        """Resolve the compute device, applying the MPS/float64 fallback.

        ``AMICATorchNG`` defaults to float64 for Fortran parity, which MPS
        cannot represent. When the device was auto-selected (the user did not
        pin one) and resolved to MPS for a float64 run, fall back to CPU so the
        default config runs instead of crashing. CUDA supports float64, so only
        MPS needs this. An explicit ``device="mps"`` is left untouched and
        surfaces ``AMICATorchNG``'s own ValueError; users wanting MPS pass
        ``dtype=torch.float32`` too.
        """
        device = setup_device() if self.device is None else self.device
        dev_type = getattr(device, "type", device)
        if self.device is None and dev_type == "mps" and ng_dtype == torch.float64:
            device = torch.device("cpu")
            msg = (
                "AMICA uses float64 for Fortran parity; MPS lacks float64 "
                "support, so falling back to CPU. Pass dtype=torch.float32 "
                "with device='mps' to run on MPS."
            )
            logger.warning(msg)
            if self.verbose:
                print(msg)
        return device

    def fit(
        self,
        X: np.ndarray,
        max_iter: int = 100,
        lrate: float = 0.05,
        do_mean: bool = True,
        do_sphere: bool = True,
        do_newton: bool = False,
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
            Whether to enable the Fortran-parity Newton preconditioner (tune
            via ``newt_start``/``newtrate`` in ``**kwargs``).
        **kwargs
            Additional parameters passed to the :class:`AMICATorchNG`
            constructor (e.g. ``block_size``, ``rho0``, ``seed``, ``dtype``) --
            the backend's tunables are constructor arguments, not fit() kwargs.

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

        # Setup device (with the MPS/float64 parity fallback, see _select_device).
        device = self._select_device(kwargs.get("dtype", _NG_DEFAULT_DTYPE))

        self.model_ = AMICATorchNG(
            n_channels=n_channels,
            n_models=self.n_models,
            n_mix=self.n_mix,
            lrate=lrate,
            do_mean=do_mean,
            do_sphere=do_sphere,
            do_newton=do_newton,
            device=device,
            **kwargs,
        )
        self.model_.fit(X, max_iter=max_iter, verbose=self.verbose)

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

    def save(self, filepath: str) -> None:
        """
        Save the fitted model to ``filepath`` via ``torch.save``.

        Persists the underlying :class:`AMICATorchNG` state (config + fitted
        tensors) plus the wrapper's own configuration, so :meth:`load` can
        fully reconstruct a transform-ready model. Everything written is a
        tensor or plain Python primitive (see
        :meth:`AMICATorchNG.state_dict`), so it reloads with
        ``weights_only=True``.

        Parameters
        ----------
        filepath : str
            Destination path (a ``.pt`` file by convention).
        """
        if not self.is_fitted_:
            raise ValueError("Model must be fitted before saving")

        payload = {
            "format_version": 1,
            "wrapper": {
                "n_models": self.n_models,
                "n_mix": self.n_mix,
                "verbose": self.verbose,
            },
            "backend": self.model_.state_dict(),
        }
        torch.save(payload, filepath)

    @classmethod
    def load(
        cls, filepath: str, device: Optional[Union[str, object]] = None
    ) -> "AMICA":
        """
        Load a fitted model saved by :meth:`save`.

        Parameters
        ----------
        filepath : str
            Path to a file written by :meth:`save`.
        device : str or torch.device, optional
            Device to place the restored model on. With ``None`` (auto), the
            same MPS/float64 fallback as :meth:`fit` applies so a float64
            parity model never lands on MPS.

        Returns
        -------
        amica : AMICA
            A fitted model ready for :meth:`transform` / :meth:`get_mixing_matrix`.
        """
        payload = torch.load(filepath, weights_only=True)
        version = payload.get("format_version")
        if version != 1:
            raise ValueError(
                f"unsupported AMICA save format_version: {version!r} (expected 1)"
            )

        wrapper = payload["wrapper"]
        model = cls(
            n_models=wrapper["n_models"],
            n_mix=wrapper["n_mix"],
            device=device,
            verbose=wrapper["verbose"],
        )

        # Resolve the device using the persisted backend dtype so the same
        # MPS/float64 fallback as fit() applies to an auto-selected device.
        ng_dtype = getattr(torch, payload["backend"]["config"]["dtype"])
        resolved_device = model._select_device(ng_dtype)
        model.model_ = AMICATorchNG.from_state_dict(
            payload["backend"], device=resolved_device
        )
        model.ll_history_ = model.model_.ll_history
        model.is_fitted_ = True
        return model

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
        import json

        with open(params_file, "r") as f:
            params = json.load(f)

        # Extract relevant parameters
        n_models = params.get("num_models", 1)
        n_mix = params.get("num_mix", 3)

        # Override with kwargs
        n_models = kwargs.pop("n_models", n_models)
        n_mix = kwargs.pop("n_mix", n_mix)

        return cls(n_models=n_models, n_mix=n_mix, **kwargs)
