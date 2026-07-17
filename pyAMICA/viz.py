"""Top-level, backend-agnostic visualizations for AMICA output (issue #136).

Unlike ``numpy_impl.viz`` (numpy-backend-scoped, built on the legacy
``load_results`` dict, returns ``None`` and mutates pyplot global state), the
functions here consume ``AmicaOutput`` from
:func:`pyAMICA.numpy_impl.load.loadmodout` -- so they work for any backend's
written amicaout directory -- and **return a `Figure`**, accepting an optional
``ax``/``axes`` to draw on. That is what makes them testable beyond "did not
crash". ``numpy_impl/viz.py`` is left untouched; this module is not a
replacement for it.

The three plots mirror ``postAmicaUtility``'s ``modprobplot``/``pop_modPMI``
MATLAB behaviour (observed by running the real GPL-licensed functions and
reading their rendered output/``help`` text -- never their source, per the
project's clean-room posture for GPL code) plus a fresh design for the
topography/PDF view, which has no working upstream reference
(``pop_topohistplot`` is broken on current EEGLAB).
"""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal.windows import hann
from scipy.special import logsumexp

from .metrics import block_diagonal_order
from .numpy_impl.load import AmicaOutput
from .numpy_impl.pdf import compute_pdf


def plot_pmi_heatmap(
    mi_matrix: np.ndarray,
    *,
    order: np.ndarray | None = None,
    labels: Sequence[str] | None = None,
    model: int | None = None,
    mask_diagonal: bool = True,
    ax: Axes | None = None,
    cmap: str = "viridis",
) -> Figure:
    """Square components-x-components pairwise-MI heatmap (``pop_modPMI`` view).

    Parameters
    ----------
    mi_matrix : np.ndarray
        (n, n) symmetric mutual-information matrix, e.g. from
        `pyAMICA.metrics.pairwise_mi`.
    order : np.ndarray, optional
        Length-n permutation to reorder both axes before plotting. Defaults to
        `pyAMICA.metrics.block_diagonal_order(mi_matrix)`.
    labels : sequence of str, optional
        Per-original-component labels; tick label at reordered position ``k``
        is ``labels[order[k]]``. Defaults to the original (0-based) component
        index, matching MATLAB's "original component indices in reordered
        position" tick convention.
    model : int, optional
        0-based model index, used only for the title ("Model N", 1-based, to
        match MATLAB's per-model panel titles). Omit for a single, unlabeled
        heatmap.
    mask_diagonal : bool, default True
        `pairwise_mi`'s diagonal is each component's self-entropy (~2.83 on
        real data), not a mutual information -- an order of magnitude above
        the ~0.06 off-diagonal values. Left unmasked it blows out the colour
        scale and hides all off-diagonal structure, so this defaults to True.
        Deliberate divergence: MATLAB instead zeroes its diagonal. Masking is
        the closer analogue here because our diagonal is not a small value, it
        is a different quantity, and zeroing it would fabricate an MI of 0
        where none was measured. Do not "align" this with MATLAB by zeroing
        the diagonal of the matrix itself: `pairwise_mi`'s return value feeds
        other consumers, and this is a display choice, not a data one.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on. A new figure+axes is created if omitted.
    cmap : str, default "viridis"
        Colormap name. A colorbar is always added (a deliberate improvement;
        MATLAB's `pop_modPMI` has none).

    Returns
    -------
    Figure
        The figure the heatmap was drawn on (``ax.figure`` if `ax` was given).

    Raises
    ------
    ValueError
        If `mi_matrix` is not square.
    """
    mi_matrix = np.asarray(mi_matrix)
    if mi_matrix.ndim != 2 or mi_matrix.shape[0] != mi_matrix.shape[1]:
        raise ValueError(
            f"plot_pmi_heatmap: mi_matrix must be square, got shape {mi_matrix.shape}"
        )
    n = mi_matrix.shape[0]

    # Validate unconditionally, not only on the default-order branch. Passing an
    # explicit `order` used to skip this entirely (the check rode along inside
    # `block_diagonal_order`), so a non-finite matrix from an upstream failure
    # rendered silently: imshow draws NaN as a blank cell, which reads as "no
    # dependency measured here" rather than "the computation broke".
    if not np.all(np.isfinite(mi_matrix)):
        raise ValueError(
            "plot_pmi_heatmap: mi_matrix contains non-finite values (NaN/Inf); "
            "imshow would render these as blank cells indistinguishable from a "
            "genuine near-zero mutual information. Check the pairwise_mi input "
            "(e.g. a constant or non-finite source)."
        )

    if order is None:
        order = block_diagonal_order(mi_matrix)
    order = np.asarray(order)

    reordered = mi_matrix[np.ix_(order, order)]
    if mask_diagonal:
        reordered = np.ma.array(reordered, mask=np.eye(n, dtype=bool))

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 5))
    else:
        fig = ax.figure
        # ax.figure is typed Figure | SubFigure (an Axes can live in a
        # SubFigure); every caller here uses a real top-level Figure.
        assert isinstance(fig, Figure)

    im = ax.imshow(reordered, cmap=cmap)
    ticks = np.arange(n)
    if labels is not None:
        tick_labels = [str(labels[i]) for i in order]
    else:
        tick_labels = [str(int(i)) for i in order]
    ax.set_xticks(ticks)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    ax.set_yticks(ticks)
    ax.set_yticklabels(tick_labels, fontsize=7)

    ax.set_title("Pairwise MI" if model is None else f"Model {model + 1}")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mutual information (nats)")

    return fig


def plot_model_probability(
    out: AmicaOutput,
    *,
    srate: float | None = None,
    smooth_sec: float | None = None,
    window_sec: float | None = None,
    axes: Sequence[Axes] | None = None,
) -> Figure:
    """Two-panel per-model probability + best-model log-likelihood (``modprobplot``).

    Top panel: `softmax(Lht)` over models, one line per model ("Probability of
    Model Being Active"). Bottom panel: the per-sample log-likelihood of the
    single most probable model at each timepoint (``Lht.max(axis=0)``), not
    the total ``Lt``, matching the observed MATLAB behaviour.

    Parameters
    ----------
    out : AmicaOutput
        Must have `out.Lht` populated (per-model per-sample log-likelihood;
        see `pyAMICA.numpy_impl.load.loadmodout`).
    srate : float, optional
        Sampling rate in Hz. pyAMICA has no built-in notion of sample rate
        (`AmicaOutput`/`loadmodout`/`load_data_file` carry none), so a seconds
        x-axis is opt-in: without it, the x-axis is in samples ("Sample").
        Use `pyAMICA.numpy_impl.load.read_eeglab_set_metadata` to get a real
        recording's srate.
    smooth_sec : float, optional
        Hanning-smoothing window, in seconds (requires `srate`). Applied to
        `Lht` BEFORE the softmax (order matters: smoothing the already-
        normalized probabilities would not reproduce the same result).
    window_sec : float, optional
        Initial x-axis view width, in seconds (requires `srate`), analogous to
        `modprobplot`'s default 20 s scrolling window -- this static figure
        cannot scroll, so it is applied as a fixed initial ``xlim`` instead.
    axes : sequence of 2 Axes, optional
        ``(top, bottom)`` axes to draw on. A new figure+axes is created if
        omitted.

    Returns
    -------
    Figure
        The figure the two panels were drawn on.

    Raises
    ------
    ValueError
        If `out.Lht` is `None`, or if `smooth_sec`/`window_sec` is given
        without `srate`.
    """
    if out.Lht is None:
        raise ValueError(
            "plot_model_probability: out.Lht is None (no LLt saved with this "
            "fit); refit with a backend that writes LLt (see "
            "AMICA.write_amica_output) or pass an AmicaOutput loaded from a "
            "directory that has it."
        )
    if smooth_sec is not None and srate is None:
        raise ValueError(
            "plot_model_probability: smooth_sec requires srate (pyAMICA has "
            "no built-in sample rate; pass srate explicitly, e.g. from "
            "read_eeglab_set_metadata)."
        )
    if window_sec is not None and srate is None:
        raise ValueError(
            "plot_model_probability: window_sec requires srate (it is a "
            "seconds-based x-axis width; pass srate explicitly)."
        )

    Lht = np.asarray(out.Lht, dtype=np.float64)
    n_models, n_samples = Lht.shape

    if smooth_sec is not None:
        assert srate is not None  # guaranteed by the eager check above
        window = int(round(srate * smooth_sec))
        if window < 1:
            raise ValueError(
                f"plot_model_probability: smooth_sec={smooth_sec} at "
                f"srate={srate} gives a {window}-sample window (< 1); "
                "increase smooth_sec or srate."
            )
        if window > n_samples:
            raise ValueError(
                f"plot_model_probability: smooth_sec={smooth_sec} at "
                f"srate={srate} gives a {window}-sample window, longer than "
                f"the {n_samples}-sample recording; reduce smooth_sec."
            )
        # MATLAB hanning(n): no zero endpoints (unlike np.hanning/scipy's
        # symmetric hann(n)), so build it via a length-(n+2) symmetric window
        # with the zero endpoints sliced off.
        w = hann(window + 2, sym=True)[1:-1]

        # Edge-correct via window-overlap normalization: a naive
        # convolve(..., mode="same") zero-pads beyond the data, and since Lht
        # sits around -100 (nowhere near 0), that padding drags the first/last
        # ~window/2 samples violently toward zero -- confidently WRONG model
        # probabilities at the start/end of every plot after the softmax.
        # Dividing by the same window convolved with a ones-signal renormalizes
        # each output sample by the fraction of the window actually inside the
        # data. Verified against the MATLAB smoothing oracle
        # (`smooth_amica_prob`) on a real 2-model fit. Name which quantity is
        # which, because two get compared and their numbers differ: this
        # SMOOTHED LOG-LIKELIHOOD correlates at 0.9939 (1 s window) / 0.9836
        # (5 s), while the PROBABILITIES it becomes after the softmax below
        # correlate at 0.9886 / 0.9594. Both are recorded in
        # `.context/issue-136/matlab_viz_verification.md`; an unlabelled 0.994
        # here previously read as a stale copy of the 0.9886 figure (#136).
        #
        # Known, accepted divergence: MATLAB additionally pins its first/last
        # sample to the raw unsmoothed input (a MATLAB smooth() idiom); this
        # returns a locally-averaged value there instead. NOT replicated on
        # purpose -- matching it would require reading GPL source, and pinning
        # the very first sample of a "smoothed" signal to the raw value is
        # arguably wrong. Do not tune this toward MATLAB.
        def _edge_corrected_smooth(row: np.ndarray) -> np.ndarray:
            num = np.convolve(row, w, mode="same")
            den = np.convolve(np.ones_like(row), w, mode="same")
            return num / den

        Lht_smooth = np.array([_edge_corrected_smooth(row) for row in Lht])
    else:
        Lht_smooth = Lht

    v_smooth = np.exp(Lht_smooth - logsumexp(Lht_smooth, axis=0, keepdims=True))
    ll_best = Lht_smooth.max(axis=0)

    if srate is not None:
        t = np.arange(n_samples) / srate
        xlabel = "Time (s)"
    else:
        t = np.arange(n_samples)
        xlabel = "Sample"

    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    else:
        fig = axes[0].figure
        assert isinstance(fig, Figure)
    ax_top, ax_bottom = axes

    for h in range(n_models):
        ax_top.plot(t, v_smooth[h], label=f"Model {h + 1}")
    ax_top.set_ylabel("P(model active)")
    ax_top.set_ylim(0, 1)
    ax_top.set_title("Probability of Model Being Active")
    ax_top.legend()

    ax_bottom.plot(t, ll_best)
    ax_bottom.set_ylabel("Log-likelihood")
    ax_bottom.set_xlabel(xlabel)
    ax_bottom.set_title("Log-likelihood of data under most probable model")

    if window_sec is not None:
        ax_top.set_xlim(0, window_sec)
        ax_bottom.set_xlim(0, window_sec)

    fig.tight_layout()
    return fig


def plot_topo_pdf(
    out: AmicaOutput,
    positions: np.ndarray,
    *,
    data: np.ndarray | None = None,
    comps: Sequence[int] | None = None,
    model: int = 0,
    n_points: int = 1000,
    axes: np.ndarray | None = None,
) -> Figure:
    """Per-component scalp map beside its fitted generalized-Gaussian mixture PDF.

    Fresh design (issue #136 trap 5: `pop_topohistplot` is broken upstream on
    current EEGLAB, so there is no working visual reference to match). Each
    row is one component: a scalp map of ``out.A[:, i, model]`` next to the
    fitted mixture density (`pyAMICA.numpy_impl.pdf.compute_pdf`), optionally
    overlaid on a histogram of that component's real activations.

    Parameters
    ----------
    out : AmicaOutput
    positions : np.ndarray
        (n_channels, 3) channel positions (e.g. EEGLAB ``X``/``Y``/``Z``, from
        `pyAMICA.numpy_impl.load.read_eeglab_set_metadata`); pyAMICA itself
        has no channel-location data.
    data : np.ndarray, optional
        (n_channels, n_samples) raw data the model was fit on. When given,
        each component's real activations
        (``out.W[:, :, model] @ out.S[:out.num_pcs] @ (data - out.data_mean)``)
        are histogrammed behind the fitted PDF; when omitted, only the fitted
        density curve is drawn (over a fixed ``[-5, 5]`` range).
    comps : sequence of int, optional
        0-based component indices to plot (one row each). Defaults to all
        components in `out` -- pass an explicit, small list for large models.
    model : int, default 0
        0-based model index.
    n_points : int, default 1000
        Number of points the fitted PDF curve is evaluated at.
    axes : np.ndarray of Axes, optional
        ``(len(comps), 2)`` array of axes to draw on. A new figure is created
        if omitted.

    Returns
    -------
    Figure
        The figure the scalp maps and PDFs were drawn on.

    Raises
    ------
    ValueError
        If `positions` is not `(n_channels, 3)`, or its channel count does not
        match `out.data_dim`.
    ImportError
        If `mne` is not installed (optional `viz` extra: ``pip install
        pyAMICA[viz]``).
    """
    positions = np.asarray(positions)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(
            f"plot_topo_pdf: positions must be (n_channels, 3), got shape "
            f"{positions.shape}"
        )
    if positions.shape[0] != out.data_dim:
        raise ValueError(
            f"plot_topo_pdf: positions has {positions.shape[0]} channels but "
            f"out.data_dim is {out.data_dim}"
        )

    try:
        import mne  # ty: ignore[unresolved-import]
    except ImportError as e:
        raise ImportError(
            "plot_topo_pdf requires mne for scalp topography maps. Install "
            "it with `pip install pyAMICA[viz]` (or `uv pip install mne`)."
        ) from e

    n_comps = out.A.shape[1]
    if comps is None:
        comps = list(range(n_comps))
    comps = list(comps)

    if axes is not None and np.asarray(axes).shape[0] != len(comps):
        raise ValueError(
            f"plot_topo_pdf: axes has {np.asarray(axes).shape[0]} rows but "
            f"{len(comps)} components were requested"
        )

    n_mix = out.alpha.shape[0]

    activations = None
    if data is not None:
        data = np.asarray(data)
        centered = data - out.data_mean[:, None]
        sphered = out.S[: out.num_pcs] @ centered
        activations = out.W[:, :, model] @ sphered  # (n_comps, n_samples)

    if axes is None:
        fig, axes = plt.subplots(
            len(comps), 2, figsize=(8, 3 * len(comps)), squeeze=False
        )
    else:
        fig = np.asarray(axes)[0, 0].figure
    axes = np.asarray(axes)

    ch_names = [f"ch{i}" for i in range(positions.shape[0])]
    montage = mne.channels.make_dig_montage(
        ch_pos=dict(zip(ch_names, positions)), coord_frame="head"
    )
    info = mne.create_info(ch_names, sfreq=1.0, ch_types="eeg")
    info.set_montage(montage)

    for row, comp in enumerate(comps):
        ax_topo = axes[row, 0]
        ax_pdf = axes[row, 1]

        scalp = out.A[:, comp, model]
        mne.viz.plot_topomap(scalp, info, axes=ax_topo, show=False)
        ax_topo.set_title(f"Component {comp + 1}")

        if activations is not None:
            act = activations[comp]
            ax_pdf.hist(act, bins=50, density=True, alpha=0.5, color="gray")
            x = np.linspace(act.min(), act.max(), n_points)
        else:
            x = np.linspace(-5, 5, n_points)

        pdf_total = np.zeros_like(x)
        for j in range(n_mix):
            alpha_j = out.alpha[j, comp, model]
            mu_j = out.mu[j, comp, model]
            sbeta_j = out.sbeta[j, comp, model]
            rho_j = out.rho[j, comp, model]
            y = sbeta_j * (x - mu_j)
            pdf, _ = compute_pdf(y, rho_j)
            pdf_total += alpha_j * sbeta_j * pdf

        ax_pdf.plot(x, pdf_total, "r-", label="Fitted mixture")
        ax_pdf.set_xlabel("Activation")
        ax_pdf.set_ylabel("Density")
        ax_pdf.legend()

    fig.tight_layout()
    return fig
