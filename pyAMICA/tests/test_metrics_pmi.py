"""Unit tests for ``pyAMICA.metrics.pmi`` (issue #135, PMI metric phase 2).

PMI is backend-agnostic pure NumPy, so it lives directly under
``pyAMICA/tests/`` (like ``test_metrics_mir.py``). Per the NO-MOCK policy,
every test exercises the real bundled sample EEG data; none use
synthetic/random data as ground truth. The error-path tests construct
degenerate inputs (a constant channel, a non-finite sample) by modifying a
copy of that real data, the same precedent as ``test_metrics_mir.py``.
"""

from pathlib import Path

import numpy as np
import pytest

from pyAMICA.metrics._common import resolve_nbins
from pyAMICA.metrics.pmi import (
    _binned_entropy_from_counts,
    block_diagonal_order,
    pairwise_mi,
)
from pyAMICA.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
PYRESULTS_DIR = SAMPLE_DIR / "pyresults"
NW = 32
FIELD = 30504


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


@pytest.fixture(scope="module")
def data_slice(real_data) -> np.ndarray:
    return real_data[:, :4000]


@pytest.fixture(scope="module")
def real_unmixing() -> np.ndarray:
    """The real fitted AMICA unmixing (W.T @ sphere) for the bundled sample data."""
    w_path = PYRESULTS_DIR / "W.npy"
    sphere_path = PYRESULTS_DIR / "sphere.npy"
    if not w_path.exists() or not sphere_path.exists():
        pytest.skip("fitted reference results missing")
    W = np.load(w_path)
    sphere = np.load(sphere_path)
    return W[:, :, 0].T @ sphere


def test_pairwise_mi_is_exactly_symmetric(data_slice):
    subset = data_slice[:8]
    mi_matrix = pairwise_mi(subset)
    np.testing.assert_array_equal(mi_matrix, mi_matrix.T)


def test_pairwise_mi_diagonal_matches_marginal_entropy(data_slice):
    n_channels = 4
    subset = data_slice[:n_channels]
    n_samples = subset.shape[1]
    nbins = resolve_nbins(n_samples, None)

    mi_matrix = pairwise_mi(subset)

    for ch in range(n_channels):
        counts, _ = np.histogram(subset[ch], bins=nbins)
        expected = _binned_entropy_from_counts(counts, n_samples)
        np.testing.assert_allclose(mi_matrix[ch, ch], expected, rtol=1e-8)


def test_ica_sources_have_lower_pairwise_mi_than_raw_channels(
    data_slice, real_unmixing
):
    """The real, checkable claim ICA makes: less pairwise redundancy than raw channels."""
    n_channels = 8
    raw = data_slice[:n_channels]
    sources = (real_unmixing @ data_slice)[:n_channels]

    mi_raw = pairwise_mi(raw)
    mi_sources = pairwise_mi(sources)

    off_diag_mask = ~np.eye(n_channels, dtype=bool)
    mean_raw = mi_raw[off_diag_mask].mean()
    mean_sources = mi_sources[off_diag_mask].mean()

    # Empirically mean_sources ~ 0.38 vs mean_raw ~ 0.61 on the bundled
    # sample data (see issue #135); a generous 0.8x margin avoids flakiness
    # while still requiring a meaningful (not just noise-level) reduction.
    assert mean_sources < 0.8 * mean_raw


def test_block_diagonal_order_returns_valid_permutation(data_slice):
    n_channels = 8
    mi_matrix = pairwise_mi(data_slice[:n_channels])
    order = block_diagonal_order(mi_matrix)
    assert sorted(order.tolist()) == list(range(n_channels))


def test_block_diagonal_order_increases_adjacent_weight(data_slice):
    n_channels = 8
    mi_matrix = pairwise_mi(data_slice[:n_channels])
    order = block_diagonal_order(mi_matrix)

    identity_weight = sum(mi_matrix[k, k + 1] for k in range(n_channels - 1))
    reordered_weight = sum(
        mi_matrix[order[k], order[k + 1]] for k in range(n_channels - 1)
    )

    assert reordered_weight >= identity_weight


def test_pairwise_mi_raises_on_non_finite_data(data_slice):
    contaminated = data_slice.copy()
    contaminated[0, 5] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        pairwise_mi(contaminated)


def test_pairwise_mi_raises_on_constant_channel(data_slice):
    flat = data_slice.copy()
    flat[0] = 0.0

    with pytest.raises(ValueError, match="constant"):
        pairwise_mi(flat)


def test_block_diagonal_order_trivial_cases():
    assert block_diagonal_order(np.zeros((0, 0))).tolist() == []
    assert block_diagonal_order(np.zeros((1, 1))).tolist() == [0]
