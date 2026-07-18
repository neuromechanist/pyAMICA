"""MNE-Python-facing wrapper over pamica's AMICA backend (issue #139, phase 1).

:class:`AMICAICA` fits AMICA directly from an :class:`mne.io.Raw` /
:class:`mne.Epochs` and exposes the standard MNE ICA consumer surface
(``get_sources``, ``apply``, ``get_components``, ``plot_components``). Its core
value-add is :meth:`AMICAICA.to_mne_ica`, which maps pamica's fitted
mean/sphere/unmixing into a fully-populated :class:`mne.preprocessing.ICA`; the
other methods delegate to that object, so MNE performs the export/plot/back-
projection machinery unchanged.

This phase covers the single-model case (``n_models == 1``). Multi-model
exposure is issue #141.
"""

from typing import Optional, Union

import numpy as np
import torch

# MNE is an optional extra (`pip install pamica[mne]`); this subpackage is never
# imported by ``pamica.__init__``, so ``import pamica`` does not require it. The
# base type-check/CI env has no mne, hence the scoped ignore (issue #139).
import mne  # ty: ignore[unresolved-import]
from mne.preprocessing import ICA as _MNEICA  # ty: ignore[unresolved-import]

from ..amica import AMICA


class AMICAICA:
    """Fit AMICA from MNE objects and interoperate with ``mne.preprocessing.ICA``.

    The wrapper fits pamica's natural-gradient AMICA backend on the data of an
    MNE :class:`~mne.io.Raw` or :class:`~mne.Epochs` and lets MNE consume the
    result: :meth:`get_sources`, :meth:`apply`, :meth:`get_components` and
    :meth:`plot_components` all delegate to a real
    :class:`mne.preprocessing.ICA` built by :meth:`to_mne_ica`.

    Parameters
    ----------
    n_mix : int, default=3
        Number of mixture components per source (AMICA ``n_mix``).
    random_state : int or None, default=None
        Seed for the AMICA fit (passed through as the backend ``seed``) and
        stored on the exported :class:`~mne.preprocessing.ICA`.
    device : str or torch.device, optional
        Torch device for the fit (``None`` = auto; the float64 parity backend
        falls back to CPU when auto-selection lands on MPS). See :class:`AMICA`.
    verbose : bool, default=True
        Whether the underlying :class:`AMICA` prints fit progress.

    Attributes
    ----------
    amica_ : AMICA
        The fitted single-model pamica model.
    info_ : mne.Info
        The picked measurement info the fit was run on (channel subset only).
    ch_names_ : list of str
        Names of the fitted channels, in order.
    n_components_ : int
        Number of ICA components (equals the number of fitted channels; AMICA
        keeps ``n_sources == n_channels``).

    Notes
    -----
    The single-model AMICA transform is
    ``S = W_fort @ sphere @ (X - mean)`` (the per-model data-space center ``c``
    is exactly zero for one model). MNE computes sources as
    ``S = unmixing_matrix_ @ pca_components_ @ (X - pca_mean_)`` (with a unit
    pre-whitener). Writing the symmetric-ZCA ``sphere`` as
    ``V @ diag(1/sqrt(e)) @ V.T`` with ``V`` orthonormal, the exported ICA uses
    ``pca_components_ = V.T``, ``unmixing_matrix_ = W_fort @ sphere @ V`` and
    ``pca_mean_ = mean``. Keeping ``pca_components_`` orthonormal is what makes
    MNE's ``get_components`` (scalp maps ``inv(sphere) @ inv(W_fort)``) come out
    right, since MNE assumes orthonormal PCA rows. The mapping is pinned by a
    round-trip test (``to_mne_ica().get_sources(raw)`` equals
    ``amica_.transform(X)``).
    """

    def __init__(
        self,
        n_mix: int = 3,
        random_state: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True,
    ):
        self.n_mix = n_mix
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

        self.amica_: Optional[AMICA] = None
        self.info_ = None
        self.ch_names_: Optional[list] = None
        self.n_components_: Optional[int] = None
        self._fit_kind: Optional[str] = None
        self._mne_ica: Optional[_MNEICA] = None

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self,
        inst,
        picks=None,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        **fit_kwargs,
    ) -> "AMICAICA":
        """Fit AMICA to the data of an MNE ``Raw`` or ``Epochs``.

        Parameters
        ----------
        inst : mne.io.BaseRaw | mne.BaseEpochs
            The data to decompose. ``Epochs`` are concatenated along time
            (``np.hstack``), matching how MNE's own ICA fits epoched data.
        picks : str | list | slice | None, default=None
            Channels to fit, in any form MNE accepts (e.g. ``"eeg"``,
            ``"data"``, a name list). ``None`` selects all good data channels
            (bads excluded), matching MNE's ICA default.
        start, stop : int | None
            Sample range (``Raw`` only) passed to ``get_data``; ``None`` uses
            the full recording.
        **fit_kwargs
            Forwarded to :meth:`AMICA.fit` (e.g. ``max_iter``, ``lrate``,
            ``do_newton``) and the backend constructor (e.g. ``block_size``).

        Returns
        -------
        self : AMICAICA
        """
        if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError(
                "AMICAICA.fit expects an mne.io.Raw or mne.Epochs, got "
                f"{type(inst).__name__}."
            )

        picked = inst.copy().pick("data" if picks is None else picks, exclude="bads")
        self.info_ = picked.info
        self.ch_names_ = list(picked.ch_names)

        if isinstance(inst, mne.io.BaseRaw):
            range_kw = {}
            if start is not None:
                range_kw["start"] = start
            if stop is not None:
                range_kw["stop"] = stop
            X = picked.get_data(**range_kw)
            self._fit_kind = "raw"
        else:
            # Concatenate epochs along time, as MNE's ICA does for fitting.
            X = np.hstack(picked.get_data())
            self._fit_kind = "epochs"

        X = np.ascontiguousarray(X, dtype=np.float64)
        self.n_components_ = X.shape[0]

        if isinstance(self.random_state, (int, np.integer)):
            fit_kwargs.setdefault("seed", int(self.random_state))

        amica = AMICA(
            n_models=1, n_mix=self.n_mix, device=self.device, verbose=self.verbose
        )
        amica.fit(X, **fit_kwargs)

        self.amica_ = amica
        self._mne_ica = None  # invalidate any cached export
        return self

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_mne_ica(self) -> _MNEICA:
        """Build (and cache) a fully-populated :class:`mne.preprocessing.ICA`.

        The returned object is a genuine MNE ICA: ``get_sources``, ``apply``,
        ``get_components`` and the ``mne.viz`` plotters operate on it natively.
        See the class :class:`Notes <AMICAICA>` for the mean/sphere/unmixing to
        ``pca_mean_``/``pca_components_``/``unmixing_matrix_`` mapping.

        Returns
        -------
        ica : mne.preprocessing.ICA
        """
        self._check_fitted()
        if self._mne_ica is not None:
            return self._mne_ica

        assert (
            self.amica_ is not None
            and self.amica_.model_ is not None
            and self.ch_names_ is not None
        )
        backend = self.amica_.model_
        assert (
            backend.mean is not None
            and backend.sphere is not None
            and backend.c is not None
        )
        mean = backend.mean.cpu().numpy().ravel()
        sphere = backend.sphere.cpu().numpy()
        w_fort = self.amica_.get_unmixing_matrix(model_idx=0)
        c = backend.c.cpu().numpy()[:, 0]
        n_ch = sphere.shape[0]

        # Orthonormal eigenbasis of the symmetric-ZCA sphere
        # (sphere = V diag(1/sqrt(cov_eval)) V.T). eigh gives ascending
        # sphere-eigenvalues (= 1/sqrt(cov_eval)); reorder to descending
        # explained variance so pca_components_ matches MNE's PCA convention.
        sphere_evals, evecs = np.linalg.eigh(sphere)
        cov_evals = 1.0 / sphere_evals**2
        order = np.argsort(cov_evals)[::-1]
        v = evecs[:, order]
        cov_evals = cov_evals[order]

        pca_components = v.T
        unmixing = w_fort @ sphere @ v
        # c is the per-model center in sphered space; fold it into pca_mean via
        # the data-space offset inv(sphere) @ c. For a single model c is exactly
        # zero, so this leaves pca_mean == mean bit-for-bit (no eigh residual).
        if np.any(c):
            pca_mean = mean + np.linalg.solve(sphere, c)
        else:
            pca_mean = mean

        ica = _MNEICA(
            n_components=n_ch,
            method="infomax",
            max_iter="auto",
            random_state=self.random_state,
        )
        ica.info = self.info_
        ica.ch_names = list(self.ch_names_)
        ica.n_components_ = n_ch
        ica.pca_mean_ = pca_mean
        ica.pca_components_ = pca_components
        ica.pca_explained_variance_ = cov_evals
        ica.unmixing_matrix_ = unmixing
        ica.pre_whitener_ = np.ones((n_ch, 1))
        ica.n_iter_ = max(int(getattr(backend, "iteration", 0)), 1)
        ica._update_mixing_matrix()
        ica._update_ica_names()
        ica.current_fit = self._fit_kind

        self._mne_ica = ica
        return ica

    # ------------------------------------------------------------------
    # MNE consumer surface (delegates to the exported ICA)
    # ------------------------------------------------------------------
    def get_sources(self, inst):
        """Sources for ``inst`` as an MNE object (see ``ICA.get_sources``)."""
        return self.to_mne_ica().get_sources(inst)

    def apply(self, inst, **kwargs):
        """Remove selected components and back-project (see ``ICA.apply``).

        Pass ``exclude=[...]`` (or set it on the exported ICA) to drop
        components; with no exclusions this reconstructs the input.
        """
        return self.to_mne_ica().apply(inst, **kwargs)

    def get_components(self) -> np.ndarray:
        """Scalp maps, shape ``(n_channels, n_components)`` (``ICA.get_components``)."""
        return self.to_mne_ica().get_components()

    def plot_components(self, *args, **kwargs):
        """Plot component topographies (see ``ICA.plot_components``)."""
        return self.to_mne_ica().plot_components(*args, **kwargs)

    def plot_sources(self, inst, *args, **kwargs):
        """Plot component time courses (see ``ICA.plot_sources``)."""
        return self.to_mne_ica().plot_sources(inst, *args, **kwargs)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _check_fitted(self) -> None:
        if self.amica_ is None:
            raise RuntimeError("AMICAICA must be fitted before this call; run fit().")

    def __repr__(self) -> str:
        if self.amica_ is None:
            return f"<AMICAICA (unfitted, n_mix={self.n_mix})>"
        return (
            f"<AMICAICA (fitted: {self.n_components_} components, "
            f"n_mix={self.n_mix}, {self._fit_kind})>"
        )
