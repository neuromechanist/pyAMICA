"""MNE-Python-facing wrapper over pamica's AMICA backend (issue #139, phase 1).

:class:`AMICAICA` fits AMICA directly from an :class:`mne.io.Raw` /
:class:`mne.Epochs` and exposes the standard MNE ICA consumer surface
(``get_sources``, ``apply``, ``get_components``, ``plot_components``,
``plot_sources``). Its core value-add is :meth:`AMICAICA.to_mne_ica`, which maps
pamica's fitted mean/sphere/unmixing into a fully-populated
:class:`mne.preprocessing.ICA`; the other methods delegate to that object, so MNE
performs the export/plot/back-projection machinery unchanged.

Multi-model AMICA (``n_models > 1``) is exposed per-model (each model is its own
single-model MNE ICA), plus a per-sample model-dominance accessor
(:meth:`AMICAICA.get_model_probability`) that MNE's ``ICA`` cannot represent
(issue #141). PCA reduction is unsupported (full-rank whitening only).
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
from ..torch_impl import PDFTYPE_NAMES

__all__ = ["AMICAICA", "PDFTYPE_NAMES"]


class AMICAICA:
    """Fit AMICA from MNE objects and interoperate with ``mne.preprocessing.ICA``.

    The wrapper fits pamica's natural-gradient AMICA backend on the data of an
    MNE :class:`~mne.io.Raw` or :class:`~mne.Epochs` and lets MNE consume the
    result: :meth:`get_sources`, :meth:`apply`, :meth:`get_components`,
    :meth:`plot_components` and :meth:`plot_sources` all delegate to a real
    :class:`mne.preprocessing.ICA` built by :meth:`to_mne_ica`.

    For a multi-model fit (``n_models > 1``) each model is exported as its own
    single-model MNE ICA (``to_mne_ica(model_idx=...)`` / the ``model_idx``
    argument on the consumer methods), and the per-sample model dominance --
    which MNE's ``ICA`` cannot represent -- is exposed directly by
    :meth:`get_model_probability` / :meth:`plot_model_probability`.

    Separation-quality metrics (issue #133) are available directly on an MNE
    object: :meth:`mir` (Mutual Information Reduction) and :meth:`pmi` (pairwise
    mutual information between sources). The pamica-specific fitted metadata MNE
    cannot hold -- source-density family, GG shape, component sharing -- is
    inspectable via :meth:`get_pdftype` / :meth:`get_rho` / :meth:`shared_components`.

    Parameters
    ----------
    n_models : int, default=1
        Number of ICA models to learn (AMICA ``n_models``).
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
        The fitted pamica model (holds all ``n_models`` models).
    info_ : mne.Info
        The picked measurement info the fit was run on (channel subset only).
    ch_names_ : list of str
        Names of the fitted channels, in order.
    n_components_ : int
        Number of ICA components (equals the number of fitted channels; AMICA
        keeps ``n_sources == n_channels``).
    converged_ : bool
        Whether the last fit ended usable (not degenerate). A degenerate fit is
        kept for inspection but refused by the consumer methods (issue #50).
    stop_reason_ : str or None
        Why the backend fit stopped (e.g. ``"max_iter"``, ``"nan_ll"``).

    Notes
    -----
    Model ``h``'s AMICA transform is ``S = W_fort @ (sphere @ (X - mean) - c_h)``,
    where ``c_h`` is that model's data-space center (identically zero for a
    single model, since the ``c`` update is gated to ``n_models > 1``). MNE
    computes sources as ``S = unmixing_matrix_ @ pca_components_ @ (X - pca_mean_)``
    (with a unit pre-whitener). Writing the symmetric-ZCA ``sphere`` as
    ``V @ diag(1/sqrt(e)) @ V.T`` with ``V`` orthonormal, the exported ICA for
    model ``h`` uses ``pca_components_ = V.T``,
    ``unmixing_matrix_ = W_fort @ sphere @ V`` and
    ``pca_mean_ = mean + inv(sphere) @ c_h`` (which reduces to ``mean`` when
    ``c_h`` is zero). Keeping ``pca_components_`` orthonormal is what makes MNE's
    ``get_components`` (scalp maps ``inv(sphere) @ inv(W_fort)``) come out right,
    since MNE assumes orthonormal PCA rows. The mapping is pinned by a round-trip
    test (``to_mne_ica(model_idx=h).get_sources(raw)`` equals
    ``amica_.transform(X, model_idx=h)``).

    PCA reduction (``pcakeep``/``pcadb``) is unsupported: it leaves the sphere
    rank-deficient, so the export (which assumes a full-rank whitening) would be
    numerically invalid, and :meth:`fit` rejects it. Rank-deficient *data* (for
    example average-referenced EEG) is the same hazard reached through data
    conditioning rather than a keyword; such a fit typically diverges and is
    refused as degenerate.
    """

    def __init__(
        self,
        n_models: int = 1,
        n_mix: int = 3,
        random_state: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: bool = True,
    ):
        self.n_models = n_models
        self.n_mix = n_mix
        self.random_state = random_state
        self.device = device
        self.verbose = verbose

        self.amica_: Optional[AMICA] = None
        self.info_ = None
        self.ch_names_: Optional[list] = None
        self.n_components_: Optional[int] = None
        self.converged_: bool = False
        self.stop_reason_: Optional[str] = None
        self._n_samples: Optional[int] = None
        self._fit_kind: Optional[str] = None
        # Per-model export cache (one mne.preprocessing.ICA per model_idx).
        self._mne_ica_cache: dict = {}

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
            Sample range for ``Raw`` input, passed to ``get_data``; ``None`` uses
            the full recording. Not supported for ``Epochs`` (raises).
        **fit_kwargs
            Forwarded to :meth:`AMICA.fit` (e.g. ``max_iter``, ``lrate``,
            ``do_newton``) and the backend constructor (e.g. ``block_size``).
            ``pcakeep``/``pcadb`` (PCA reduction) are rejected, since they leave
            the sphere rank-deficient and the MNE export invalid.

        Returns
        -------
        self : AMICAICA

        Raises
        ------
        TypeError
            If ``inst`` is not an MNE ``Raw``/``Epochs``.
        ValueError
            If ``start``/``stop`` are given for ``Epochs``, ``stop`` exceeds the
            recording length, the selected data is non-finite, or the fit used
            PCA reduction.
        """
        if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError(
                "AMICAICA.fit expects an mne.io.Raw or mne.Epochs, got "
                f"{type(inst).__name__}."
            )
        is_raw = isinstance(inst, mne.io.BaseRaw)
        if not is_raw and (start is not None or stop is not None):
            raise ValueError("start/stop are only supported for Raw input, not Epochs.")

        picked = inst.copy().pick("data" if picks is None else picks, exclude="bads")
        ch_names = list(picked.ch_names)

        if is_raw:
            if stop is not None and stop > inst.n_times:
                raise ValueError(
                    f"stop={stop} exceeds the recording length ({inst.n_times} "
                    "samples)."
                )
            range_kw = {}
            if start is not None:
                range_kw["start"] = start
            if stop is not None:
                range_kw["stop"] = stop
            X = picked.get_data(**range_kw)
            fit_kind = "raw"
        else:
            # Concatenate epochs along time, as MNE's ICA does for fitting.
            X = np.hstack(picked.get_data())
            fit_kind = "epochs"

        X = np.ascontiguousarray(X, dtype=np.float64)
        if not np.isfinite(X).all():
            bad = [ch_names[i] for i in np.flatnonzero(~np.isfinite(X).all(axis=1))]
            raise ValueError(
                "AMICAICA.fit: input contains non-finite (NaN/Inf) samples in "
                f"channel(s) {bad}; clean bad segments/interpolation before "
                "fitting."
            )

        if isinstance(self.random_state, (int, np.integer)):
            fit_kwargs.setdefault("seed", int(self.random_state))

        amica = AMICA(
            n_models=self.n_models,
            n_mix=self.n_mix,
            device=self.device,
            verbose=self.verbose,
        )
        amica.fit(X, **fit_kwargs)

        if amica.model_ is not None and amica.model_._pca_reduced():
            raise ValueError(
                "AMICAICA does not support PCA reduction (pcakeep/pcadb): it "
                "leaves the sphere rank-deficient, so the MNE ICA export (which "
                "assumes a full-rank whitening) would be numerically invalid. "
                "Refit without pcakeep/pcadb."
            )

        # Publish to self only after every fallible step above succeeds, so a
        # failed (re)fit leaves the previously fitted state intact rather than a
        # mix of the old model and the new attempt's metadata (mirrors the
        # local-first pattern in AMICA.fit).
        self.info_ = picked.info
        self.ch_names_ = ch_names
        self.n_components_ = X.shape[0]
        self._n_samples = X.shape[1]
        self._fit_kind = fit_kind
        self.amica_ = amica
        self.converged_ = amica.converged_
        self.stop_reason_ = amica.stop_reason_
        self._mne_ica_cache = {}  # invalidate any cached exports
        return self

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------
    def to_mne_ica(self, model_idx: int = 0) -> _MNEICA:
        """Build (and cache) a fully-populated :class:`mne.preprocessing.ICA`.

        The returned object is a genuine MNE ICA: ``get_sources``, ``apply``,
        ``get_components``, ``save`` and the ``mne.viz`` plotters operate on it
        natively. See the class :class:`Notes <AMICAICA>` for the
        mean/sphere/unmixing to ``pca_mean_``/``pca_components_``/
        ``unmixing_matrix_`` mapping.

        For a multi-model fit each model is exported as its own single-model
        MNE ICA (MNE has no multi-model concept); pass ``model_idx`` to pick
        one. The per-model exports are cached and returned by reference:
        mutating one (for example setting ``.exclude``) persists across
        subsequent :meth:`apply`/:meth:`get_sources` calls for that model until
        the next :meth:`fit`.

        Parameters
        ----------
        model_idx : int, default=0
            Which AMICA model to export (``0..n_models-1``).

        Returns
        -------
        ica : mne.preprocessing.ICA
        """
        self._check_fitted("build the MNE ICA")
        model_idx = self._check_model_idx(model_idx)
        if model_idx in self._mne_ica_cache:
            return self._mne_ica_cache[model_idx]

        amica = self.amica_
        if amica is None or amica.model_ is None or self.ch_names_ is None:
            raise RuntimeError(
                "AMICAICA: internal state is inconsistent; refit before to_mne_ica()."
            )
        backend = amica.model_
        if backend.mean is None or backend.sphere is None or backend.c is None:
            raise RuntimeError(
                "AMICAICA: the fitted backend is missing mean/sphere/c; refit "
                "before to_mne_ica()."
            )

        mean = backend.mean.cpu().numpy().ravel()
        sphere = backend.sphere.cpu().numpy()
        w_fort = amica.get_unmixing_matrix(model_idx=model_idx)
        c = backend.c.cpu().numpy()[:, model_idx]  # per-model center (sphered space)
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
        # Fold the per-model center c (in sphered space) into pca_mean via the
        # data-space offset inv(sphere) @ c, so MNE's (X - pca_mean) reproduces
        # AMICA's W(sphere(X - mean) - c). c is identically zero for a single
        # model, leaving pca_mean == mean bit-for-bit.
        pca_mean = mean + np.linalg.solve(sphere, c) if np.any(c) else mean

        ica = _MNEICA(
            n_components=n_ch,
            method="infomax",
            max_iter="auto",
            random_state=self.random_state,
        )
        # `method` is inert here: the fitted attributes are hand-populated and
        # ICA.fit (the only place `method` branches) is never called, but
        # ICA.__init__ still requires a valid method name.
        ica.info = self.info_
        ica.ch_names = list(self.ch_names_)
        ica.n_components_ = n_ch
        ica.pca_mean_ = pca_mean
        ica.pca_components_ = pca_components
        ica.pca_explained_variance_ = cov_evals
        ica.unmixing_matrix_ = unmixing
        ica.pre_whitener_ = np.ones((n_ch, 1))
        ica.n_iter_ = max(int(getattr(backend, "iteration", 0)), 1)
        # MNE's own fit sets these; read_ica_eeglab (the precedent for building
        # an ICA from an external decomposition) sets reject_=None. Without them
        # ICA.save()/plot_properties raise AttributeError.
        ica.reject_ = None
        ica.n_samples_ = int(self._n_samples) if self._n_samples is not None else 0
        ica._update_mixing_matrix()
        ica._update_ica_names()
        ica.current_fit = self._fit_kind

        self._mne_ica_cache[model_idx] = ica
        return ica

    # ------------------------------------------------------------------
    # MNE consumer surface (delegates to the exported ICA)
    # ------------------------------------------------------------------
    def get_sources(self, inst, *args, model_idx: int = 0, **kwargs):
        """Sources for ``inst`` from model ``model_idx`` (see ``ICA.get_sources``).

        ``model_idx`` is keyword-only so positional arguments pass straight
        through to MNE's ``ICA.get_sources`` (e.g. ``add_channels``,
        ``start``/``stop``).
        """
        return self.to_mne_ica(model_idx).get_sources(inst, *args, **kwargs)

    def apply(self, inst, *args, model_idx: int = 0, **kwargs):
        """Remove selected components of model ``model_idx`` and back-project.

        ``model_idx`` is keyword-only so positional arguments pass straight
        through to MNE's ``ICA.apply``. Pass ``exclude=[...]`` (or set it on the
        exported ICA) to drop components; with no exclusions this reconstructs
        the input.
        """
        return self.to_mne_ica(model_idx).apply(inst, *args, **kwargs)

    def get_components(self, *, model_idx: int = 0) -> np.ndarray:
        """Scalp maps of model ``model_idx``, ``(n_channels, n_components)``."""
        return self.to_mne_ica(model_idx).get_components()

    def plot_components(self, *args, model_idx: int = 0, **kwargs):
        """Plot model ``model_idx`` component topographies (see ``ICA.plot_components``)."""
        return self.to_mne_ica(model_idx).plot_components(*args, **kwargs)

    def plot_sources(self, inst, *args, model_idx: int = 0, **kwargs):
        """Plot model ``model_idx`` component time courses (see ``ICA.plot_sources``)."""
        return self.to_mne_ica(model_idx).plot_sources(inst, *args, **kwargs)

    # ------------------------------------------------------------------
    # Multi-model dominance (issue #141)
    # ------------------------------------------------------------------
    def get_model_probability(self, inst) -> np.ndarray:
        """Per-sample posterior probability of each model on ``inst`` (dominance).

        Returns ``P(model | sample)`` as ``(n_models, n_samples)`` via
        :meth:`AMICA.model_probability` on ``inst``'s data (``Epochs`` are
        concatenated along time). Each column sums to 1; all ones for a single
        model. MNE's own ``ICA`` has no multi-model concept, so this is exposed
        here rather than through the exported per-model ICA objects.
        """
        self._check_fitted("compute the model probability")
        if self.amica_ is None:
            raise RuntimeError(
                "AMICAICA: internal state is inconsistent; refit before scoring."
            )
        return self.amica_.model_probability(self._data_for(inst))

    def plot_model_probability(self, inst, *, srate: Optional[float] = None, **kwargs):
        """Plot per-model probability + best-model log-likelihood over ``inst``.

        Delegates to :func:`pamica.viz.plot_model_probability` with the live
        per-model log-likelihood (:meth:`AMICA.model_loglik`) on ``inst``'s
        data. ``srate`` defaults to the fitted recording's sampling rate, so the
        x-axis is in seconds; extra keywords (``smooth_sec``, ``window_sec``,
        ``axes``) pass through.
        """
        from ..viz import plot_model_probability as _plot_model_probability

        self._check_fitted("plot the model probability")
        if self.amica_ is None or self.info_ is None:
            raise RuntimeError(
                "AMICAICA: internal state is inconsistent; refit before plotting."
            )
        lht = self.amica_.model_loglik(self._data_for(inst))
        if srate is None:
            srate = float(self.info_["sfreq"])
        return _plot_model_probability(lht=lht, srate=srate, **kwargs)

    # ------------------------------------------------------------------
    # Separation-quality metrics (issue #143, on top of #133)
    # ------------------------------------------------------------------
    def mir(self, inst, *, model_idx: int = 0, nbins: Optional[int] = None):
        """Mutual Information Reduction of model ``model_idx`` on ``inst``.

        How much mutual information the fitted unmixing removes from the data,
        in nats (issue #133). Delegates to :meth:`AMICA.mir` on ``inst``'s
        fitted-channel data; MIR is shift-invariant, so mean/``c`` centering is
        irrelevant.

        Parameters
        ----------
        inst : mne.io.BaseRaw | mne.BaseEpochs
            Data to score (``Epochs`` concatenated along time).
        model_idx : int, default=0
            Which model's unmixing to use.
        nbins : int, optional
            Histogram bin count; see :func:`pamica.metrics.mir`.

        Returns
        -------
        mir_nats : float
        variance : float
        """
        self._check_fitted("compute MIR")
        model_idx = self._check_model_idx(model_idx)
        assert self.amica_ is not None
        return self.amica_.mir(self._data_for(inst), model_idx=model_idx, nbins=nbins)

    def pmi(
        self, inst, *, model_idx: int = 0, nbins: Optional[int] = None
    ) -> np.ndarray:
        """Pairwise Mutual Information between model ``model_idx``'s sources on ``inst``.

        The residual pairwise dependence between fitted sources, in nats
        (issue #133). Delegates to :meth:`AMICA.pmi` on ``inst``'s fitted-channel
        data.

        Parameters
        ----------
        inst : mne.io.BaseRaw | mne.BaseEpochs
            Data to score (``Epochs`` concatenated along time).
        model_idx : int, default=0
            Which model's sources to use.
        nbins : int, optional
            Histogram bin count; see :func:`pamica.metrics.pairwise_mi`.

        Returns
        -------
        mi_matrix : np.ndarray of shape (n_components, n_components)
            Symmetric; the diagonal is each source's own entropy.
        """
        self._check_fitted("compute PMI")
        model_idx = self._check_model_idx(model_idx)
        assert self.amica_ is not None
        return self.amica_.pmi(self._data_for(inst), model_idx=model_idx, nbins=nbins)

    # ------------------------------------------------------------------
    # pamica-specific metadata (issue #142)
    #
    # MNE's ``ICA`` carries no source-density family, GG shape, or
    # component-sharing state, so these are exposed here rather than silently
    # dropped by the ``mne.preprocessing.ICA`` export.
    # ------------------------------------------------------------------
    def get_pdftype(self, *, model_idx: int = 0) -> np.ndarray:
        """Per-component source-density family code for model ``model_idx``.

        One integer per ICA component (0-4); map to names with
        :data:`pamica.mne_compat.PDFTYPE_NAMES`. All components share one family
        unless the adaptive switcher (``pdftype=1``) moved them (issue #26).
        """
        self._check_fitted("get the density family")
        model_idx = self._check_model_idx(model_idx)
        assert self.amica_ is not None
        return self.amica_.get_pdftype(model_idx=model_idx)

    def get_rho(self, *, model_idx: int = 0) -> np.ndarray:
        """Generalized-Gaussian shape ``rho`` for model ``model_idx``.

        Shape ``(n_mix, n_components)``; ``rho == 2`` is Gaussian-shaped,
        ``rho == 1`` Laplacian, ``rho < 1`` heavier-tailed. Meaningful only for
        the generalized-Gaussian family (``pdftype=0``).
        """
        self._check_fitted("get rho")
        model_idx = self._check_model_idx(model_idx)
        assert self.amica_ is not None
        return self.amica_.get_rho(model_idx=model_idx)

    def shared_components(self) -> list:
        """Components shared across models by ``share_comps`` (issue #60).

        One group of ``(model_idx, component_idx)`` pairs per shared column;
        empty when nothing is shared (always so for a single model or a default
        multi-model fit with ``share_comps`` off).
        """
        self._check_fitted("get the shared components")
        assert self.amica_ is not None
        return self.amica_.shared_components()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _data_for(self, inst) -> np.ndarray:
        """The fitted-channel data of ``inst`` as ``(n_channels, n_samples)``.

        Selects the exact channels the fit used (by name) so the array aligns
        with the stored sphere/unmixing; ``Epochs`` are concatenated along time.
        """
        if not isinstance(inst, (mne.io.BaseRaw, mne.BaseEpochs)):
            raise TypeError(
                f"expected an mne.io.Raw or mne.Epochs, got {type(inst).__name__}."
            )
        picked = inst.copy().pick(self.ch_names_)
        if isinstance(inst, mne.io.BaseRaw):
            X = picked.get_data()
        else:
            X = np.hstack(picked.get_data())
        return np.ascontiguousarray(X, dtype=np.float64)

    def _check_model_idx(self, model_idx: int) -> int:
        if not isinstance(model_idx, (int, np.integer)):
            raise TypeError(
                f"model_idx must be an int, got {type(model_idx).__name__}."
            )
        # Bound against the fitted backend's model count (the source of truth),
        # not the mutable constructor hyperparameter self.n_models.
        n = (
            self.amica_.model_.n_models
            if self.amica_ is not None and self.amica_.model_ is not None
            else self.n_models
        )
        if not (0 <= model_idx < n):
            raise ValueError(
                f"model_idx={model_idx} out of range for a {n}-model fit "
                f"(valid: 0..{n - 1})."
            )
        return int(model_idx)

    def _check_fitted(self, action: str = "this call") -> None:
        """Raise if no usable model is available.

        ``ValueError`` when never fitted (matching :class:`AMICA`'s convention);
        ``RuntimeError`` when the fit ended degenerate (non-finite parameters,
        issue #50), so the failure surfaces here rather than as opaque NaNs
        downstream.
        """
        if self.amica_ is None:
            raise ValueError(f"AMICAICA must be fitted before {action}; run fit().")
        if not self.converged_:
            raise RuntimeError(
                f"Refusing to {action}: the AMICA fit ended degenerate "
                f"(stop_reason={self.stop_reason_!r}), so it holds non-finite "
                "parameters and would produce NaN output. Lower lrate, disable "
                "Newton, or check data conditioning, then refit."
            )

    def __repr__(self) -> str:
        if self.amica_ is None:
            return (
                f"<AMICAICA (unfitted, n_models={self.n_models}, n_mix={self.n_mix})>"
            )
        if not self.converged_:
            return (
                f"<AMICAICA (degenerate fit, stop_reason={self.stop_reason_!r}, "
                f"n_models={self.n_models}, n_mix={self.n_mix}, {self._fit_kind})>"
            )
        return (
            f"<AMICAICA (fitted: {self.n_components_} components, "
            f"n_models={self.n_models}, n_mix={self.n_mix}, {self._fit_kind})>"
        )
