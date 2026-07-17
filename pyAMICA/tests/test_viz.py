"""Tests for ``pyAMICA.viz`` (issue #136, Phase 4 visualizations).

Per the NO-MOCK policy (`.rules/testing.md`), every test exercises the real
bundled sample EEG data and a real (short but genuine) AMICA fit; none use
synthetic/random data as ground truth. The one deliberate exception is
`test_plot_topo_pdf_raises_importerror_without_mne`, which uses
``monkeypatch.setitem(sys.modules, "mne", None)`` -- the standard pytest idiom
for exercising an optional-dependency import guard -- since `mne` genuinely is
installed in this environment (the `viz` extra) and there is no other way to
hit that branch without actually uninstalling it. This does not fake any
data or numerical behavior; it only forces the real `import mne` statement in
`pyAMICA.viz` to fail, the same way it would in an environment without the
extra installed.

Assertions target structure (axes count, labels, line count, tick labels,
masked diagonal, error messages, probabilities summing to 1, the fitted-PDF
values themselves), matching the precedent in `test_metrics_pmi.py` /
`test_metrics_mir.py`, not pixel comparisons and not merely "did not crash".
"""

import dataclasses
import importlib.util
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure

from pyAMICA.amica import AMICA
from pyAMICA.metrics import block_diagonal_order, pairwise_mi
from pyAMICA.numpy_impl.load import (
    AmicaOutput,
    loadmodout,
    read_eeglab_set_metadata,
)
from pyAMICA.numpy_impl.pdf import compute_pdf
from pyAMICA.viz import plot_model_probability, plot_pmi_heatmap, plot_topo_pdf

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
SET_FILE = SAMPLE_DIR / "eeglab_data.set"

# `plot_topo_pdf` needs mne, the optional `viz` extra; CI installs the base env
# only (bare `uv sync`), matching how mlx_tests skips when mlx is absent. This
# is a per-test marker rather than a module-level `pytest.importorskip("mne")`
# ON PURPOSE: a module-level skip would also silently drop the ~28 tests here
# that need no mne at all, leaving a green suite that never exercised the viz
# module. Only mark tests that genuinely reach `import mne`; in particular
# `test_plot_topo_pdf_raises_importerror_without_mne` must stay unmarked, since
# it asserts the guard fires precisely when mne is missing.
requires_mne = pytest.mark.skipif(
    importlib.util.find_spec("mne") is None,
    reason="requires the optional `viz` extra (mne); install with pyAMICA[viz]",
)
NW = 32
FIELD = 30504


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    from pyAMICA.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


@pytest.fixture(scope="module")
def data_slice(real_data) -> np.ndarray:
    return real_data[:, :4096]


@pytest.fixture(scope="module")
def eeglab_metadata() -> dict:
    if not SET_FILE.exists():
        pytest.skip("eeglab_data.set missing")
    return read_eeglab_set_metadata(SET_FILE)


@pytest.fixture(scope="module")
def two_model_output(data_slice, tmp_path_factory) -> AmicaOutput:
    """A short, real 2-model NG fit, written and read back via loadmodout, so
    `out.Lht`/`out.A`/`out.alpha` etc. are genuine fitted values (not
    fabricated). Reused (read-only) across the tests in this module."""
    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(data_slice, max_iter=5, block_size=1024, seed=7)

    outdir = tmp_path_factory.mktemp("amicaout_viz")
    model.write_amica_output(str(outdir))
    return loadmodout(outdir)


@pytest.fixture(scope="module")
def small_mi_matrix(data_slice) -> np.ndarray:
    return pairwise_mi(data_slice[:6])


@pytest.fixture(autouse=True)
def _close_figures():
    """Every test creates at least one Figure; close them so the suite doesn't
    trip matplotlib's max-open-figures warning."""
    yield
    plt.close("all")


# --- plot_pmi_heatmap -------------------------------------------------------


def test_plot_pmi_heatmap_returns_figure(small_mi_matrix):
    fig = plot_pmi_heatmap(small_mi_matrix)
    assert isinstance(fig, Figure)
    assert len(fig.axes) >= 1


def test_plot_pmi_heatmap_masks_diagonal_by_default(small_mi_matrix):
    fig = plot_pmi_heatmap(small_mi_matrix)
    im = fig.axes[0].images[0]
    arr = im.get_array()
    assert arr is not None
    mask = np.ma.getmaskarray(arr)
    n = small_mi_matrix.shape[0]
    assert np.all(mask[np.eye(n, dtype=bool)])
    assert not np.any(mask[~np.eye(n, dtype=bool)])


def test_plot_pmi_heatmap_mask_diagonal_false_leaves_diagonal_visible(
    small_mi_matrix,
):
    fig = plot_pmi_heatmap(small_mi_matrix, mask_diagonal=False)
    im = fig.axes[0].images[0]
    arr = im.get_array()
    assert arr is not None
    mask = np.ma.getmaskarray(arr)
    assert not np.any(mask)


def test_plot_pmi_heatmap_default_order_matches_block_diagonal_order(
    small_mi_matrix,
):
    expected_order = block_diagonal_order(small_mi_matrix)
    fig = plot_pmi_heatmap(small_mi_matrix)
    tick_labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert tick_labels == [str(int(i)) for i in expected_order]


def test_plot_pmi_heatmap_custom_order_is_honored(small_mi_matrix):
    n = small_mi_matrix.shape[0]
    custom_order = np.arange(n)[::-1]
    fig = plot_pmi_heatmap(small_mi_matrix, order=custom_order)
    tick_labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert tick_labels == [str(int(i)) for i in custom_order]


def test_plot_pmi_heatmap_labels_param_indexed_by_order(small_mi_matrix):
    n = small_mi_matrix.shape[0]
    labels = [f"ch{i}" for i in range(n)]
    order = np.arange(n)[::-1]
    fig = plot_pmi_heatmap(small_mi_matrix, order=order, labels=labels)
    tick_labels = [t.get_text() for t in fig.axes[0].get_xticklabels()]
    assert tick_labels == [labels[i] for i in order]


def test_plot_pmi_heatmap_model_title(small_mi_matrix):
    fig_default = plot_pmi_heatmap(small_mi_matrix)
    assert fig_default.axes[0].get_title() == "Pairwise MI"

    fig_model = plot_pmi_heatmap(small_mi_matrix, model=1)
    assert fig_model.axes[0].get_title() == "Model 2"


def test_plot_pmi_heatmap_raises_on_non_square():
    with pytest.raises(ValueError, match="square"):
        plot_pmi_heatmap(np.zeros((3, 5)))


def test_plot_pmi_heatmap_accepts_existing_ax(small_mi_matrix):
    fig, ax = plt.subplots()
    returned = plot_pmi_heatmap(small_mi_matrix, ax=ax)
    assert returned is fig
    assert len(ax.images) == 1


# --- plot_model_probability --------------------------------------------------


def test_plot_model_probability_returns_two_axes(two_model_output):
    fig = plot_model_probability(two_model_output)
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 2


def test_plot_model_probability_one_line_per_model(two_model_output):
    fig = plot_model_probability(two_model_output)
    ax_top = fig.axes[0]
    assert len(ax_top.lines) == two_model_output.num_models == 2


def test_plot_model_probability_probabilities_sum_to_one(two_model_output):
    fig = plot_model_probability(two_model_output)
    probs = np.array([line.get_ydata() for line in fig.axes[0].lines])
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-10)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)


def test_plot_model_probability_bottom_panel_is_best_model_ll(two_model_output):
    fig = plot_model_probability(two_model_output)
    ax_bottom = fig.axes[1]
    assert len(ax_bottom.lines) == 1
    expected = two_model_output.Lht.max(axis=0)
    np.testing.assert_array_equal(ax_bottom.lines[0].get_ydata(), expected)


def test_plot_model_probability_xaxis_samples_without_srate(two_model_output):
    fig = plot_model_probability(two_model_output)
    ax_bottom = fig.axes[1]
    assert ax_bottom.get_xlabel() == "Sample"
    n_samples = two_model_output.Lht.shape[1]
    np.testing.assert_array_equal(ax_bottom.lines[0].get_xdata(), np.arange(n_samples))


def test_plot_model_probability_xaxis_seconds_with_srate(
    two_model_output, eeglab_metadata
):
    srate = eeglab_metadata["srate"]
    fig = plot_model_probability(two_model_output, srate=srate)
    ax_bottom = fig.axes[1]
    assert ax_bottom.get_xlabel() == "Time (s)"
    n_samples = two_model_output.Lht.shape[1]
    expected_t = np.arange(n_samples) / srate
    got_t = np.asarray(ax_bottom.lines[0].get_xdata(), dtype=float)
    np.testing.assert_allclose(got_t, expected_t)


def test_plot_model_probability_smooth_sec_requires_srate(two_model_output):
    with pytest.raises(ValueError, match="srate"):
        plot_model_probability(two_model_output, smooth_sec=1.0)


def test_plot_model_probability_window_sec_requires_srate(two_model_output):
    with pytest.raises(ValueError, match="srate"):
        plot_model_probability(two_model_output, window_sec=5.0)


def test_plot_model_probability_smoothing_preserves_normalization(
    two_model_output, eeglab_metadata
):
    srate = eeglab_metadata["srate"]
    fig = plot_model_probability(two_model_output, srate=srate, smooth_sec=1.0)
    probs = np.array([line.get_ydata() for line in fig.axes[0].lines])
    np.testing.assert_allclose(probs.sum(axis=0), 1.0, atol=1e-8)


def test_plot_model_probability_smoothing_edge_corrected_not_dragged_to_zero(
    two_model_output, eeglab_metadata
):
    """A naive convolve(..., mode="same") zero-pads beyond the data; since Lht
    sits around -100 (nowhere near 0), that would drag the boundary samples
    violently toward zero. The edge-corrected smoothing must keep the first
    and last smoothed samples close to their neighbours instead (issue #136
    MATLAB-oracle finding)."""
    srate = eeglab_metadata["srate"]
    fig = plot_model_probability(two_model_output, srate=srate, smooth_sec=1.0)
    ax_bottom = fig.axes[1]
    ll = np.asarray(ax_bottom.lines[0].get_ydata(), dtype=float)
    raw = two_model_output.Lht.max(axis=0)

    # The naive (uncorrected) approach would put ll[0] far from raw[0] (tens
    # of units away, per the MATLAB-oracle measurement); edge-corrected
    # smoothing keeps it within a small multiple of the local variation.
    local_scale = np.std(raw[:200]) + 1e-9
    assert abs(ll[0] - raw[0]) < 10 * local_scale
    assert abs(ll[-1] - raw[-1]) < 10 * local_scale


def test_plot_model_probability_smooth_sec_too_small_raises(
    two_model_output, eeglab_metadata
):
    srate = eeglab_metadata["srate"]
    with pytest.raises(ValueError, match="smooth_sec"):
        plot_model_probability(two_model_output, srate=srate, smooth_sec=0.001)


def test_plot_model_probability_smooth_sec_too_large_raises(
    two_model_output, eeglab_metadata
):
    srate = eeglab_metadata["srate"]
    n_samples = two_model_output.Lht.shape[1]
    too_long_sec = (n_samples + 1) / srate
    with pytest.raises(ValueError, match="smooth_sec"):
        plot_model_probability(two_model_output, srate=srate, smooth_sec=too_long_sec)


def test_plot_model_probability_window_sec_sets_xlim(two_model_output, eeglab_metadata):
    srate = eeglab_metadata["srate"]
    fig = plot_model_probability(two_model_output, srate=srate, window_sec=5.0)
    assert fig.axes[1].get_xlim() == (0.0, 5.0)


def test_plot_model_probability_legend_labels(two_model_output):
    fig = plot_model_probability(two_model_output)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    legend_labels = [t.get_text() for t in legend.get_texts()]
    assert legend_labels == ["Model 1", "Model 2"]


def test_plot_model_probability_accepts_existing_axes(two_model_output):
    fig, axes = plt.subplots(2, 1)
    returned = plot_model_probability(two_model_output, axes=axes)
    assert returned is fig


def test_plot_model_probability_raises_without_lht(two_model_output):
    out_no_lht = dataclasses.replace(two_model_output, Lht=None)
    with pytest.raises(ValueError, match="Lht"):
        plot_model_probability(out_no_lht)


# --- plot_topo_pdf -----------------------------------------------------------


@requires_mne
def test_plot_topo_pdf_returns_expected_axes_count(two_model_output, eeglab_metadata):
    fig = plot_topo_pdf(
        two_model_output, eeglab_metadata["positions"], comps=[0, 1], model=0
    )
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 4  # 2 components x (topo, pdf)


@requires_mne
def test_plot_topo_pdf_component_titles(two_model_output, eeglab_metadata):
    fig = plot_topo_pdf(
        two_model_output, eeglab_metadata["positions"], comps=[0, 1], model=0
    )
    # row-major creation order: (0,0)=topo0, (0,1)=pdf0, (1,0)=topo1, (1,1)=pdf1
    assert fig.axes[0].get_title() == "Component 1"
    assert fig.axes[2].get_title() == "Component 2"


@requires_mne
def test_plot_topo_pdf_pdf_panel_without_data(two_model_output, eeglab_metadata):
    fig = plot_topo_pdf(
        two_model_output, eeglab_metadata["positions"], comps=[0], model=0
    )
    ax_pdf = fig.axes[1]
    assert len(ax_pdf.lines) == 1  # fitted mixture curve only, no histogram
    assert len(ax_pdf.patches) == 0
    assert ax_pdf.get_xlabel() == "Activation"
    assert ax_pdf.get_ylabel() == "Density"


@requires_mne
def test_plot_topo_pdf_histogram_overlay_when_data_given(
    two_model_output, eeglab_metadata, data_slice
):
    fig = plot_topo_pdf(
        two_model_output,
        eeglab_metadata["positions"],
        data=data_slice,
        comps=[0],
        model=0,
    )
    ax_pdf = fig.axes[1]
    assert len(ax_pdf.lines) == 1
    assert len(ax_pdf.patches) > 0  # histogram bars


@requires_mne
def test_plot_topo_pdf_curve_matches_compute_pdf(two_model_output, eeglab_metadata):
    """The fitted curve must be exactly compute_pdf's mixture (reused, not
    reimplemented): y = sbeta*(x-mu), scaled by alpha*sbeta, summed."""
    fig = plot_topo_pdf(
        two_model_output,
        eeglab_metadata["positions"],
        comps=[0],
        model=0,
        n_points=200,
    )
    ax_pdf = fig.axes[1]
    x = np.asarray(ax_pdf.lines[0].get_xdata(), dtype=float)
    y = np.asarray(ax_pdf.lines[0].get_ydata(), dtype=float)

    n_mix = two_model_output.alpha.shape[0]
    expected = np.zeros_like(x)
    for j in range(n_mix):
        alpha_j = two_model_output.alpha[j, 0, 0]
        mu_j = two_model_output.mu[j, 0, 0]
        sbeta_j = two_model_output.sbeta[j, 0, 0]
        rho_j = two_model_output.rho[j, 0, 0]
        yv = sbeta_j * (x - mu_j)
        pdf, _ = compute_pdf(yv, rho_j)
        expected += alpha_j * sbeta_j * pdf

    np.testing.assert_allclose(y, expected, rtol=1e-10)


def test_plot_topo_pdf_raises_on_non_3col_positions(two_model_output):
    with pytest.raises(ValueError, match="positions"):
        plot_topo_pdf(two_model_output, np.zeros((NW, 2)), comps=[0])


def test_plot_topo_pdf_raises_on_channel_count_mismatch(two_model_output):
    with pytest.raises(ValueError, match="data_dim"):
        plot_topo_pdf(two_model_output, np.zeros((5, 3)), comps=[0])


@requires_mne
def test_plot_topo_pdf_raises_on_axes_shape_mismatch(two_model_output, eeglab_metadata):
    fig, axes = plt.subplots(1, 2, squeeze=False)
    with pytest.raises(ValueError, match="axes"):
        plot_topo_pdf(
            two_model_output,
            eeglab_metadata["positions"],
            comps=[0, 1],
            axes=axes,
        )


@requires_mne
def test_plot_topo_pdf_accepts_existing_axes(two_model_output, eeglab_metadata):
    fig, axes = plt.subplots(1, 2, squeeze=False)
    returned = plot_topo_pdf(
        two_model_output, eeglab_metadata["positions"], comps=[0], axes=axes
    )
    assert returned is fig


def test_plot_topo_pdf_raises_importerror_without_mne(
    two_model_output, eeglab_metadata, monkeypatch
):
    """`mne` is the optional `viz` extra; when it cannot be imported,
    plot_topo_pdf must raise a clear ImportError naming the install command,
    not a bare ModuleNotFoundError traceback."""
    monkeypatch.setitem(sys.modules, "mne", None)
    with pytest.raises(ImportError, match=r"pyAMICA\[viz\]"):
        plot_topo_pdf(two_model_output, eeglab_metadata["positions"], comps=[0])
