"""Unit tests for ``pyAMICA.metrics.mir`` (issue #134, MIR port phase 1).

MIR is backend-agnostic pure NumPy, so it lives directly under ``pyAMICA/tests/``
(like ``test_amari_distance.py``) rather than ``tests/torch_tests/``. Per the
NO-MOCK policy, every test exercises the real bundled sample EEG data; none use
synthetic/random data as ground truth. The error-path tests construct
degenerate inputs (a constant channel, a non-finite sample, a singular matrix)
by modifying a copy of that real data or a real fitted unmixing matrix -- the
same precedent as ``test_amari_distance.py``'s degenerate-input tests -- not
by fabricating synthetic ground truth.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from pyAMICA.metrics.mir import _marginal_entropies, mir
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


def test_marginal_entropies_default_nbins_matches_formula(data_slice):
    n_samples = data_slice.shape[1]
    expected_nbins = round(3 * math.log2(1 + n_samples / 10))

    h_default, v_default = _marginal_entropies(data_slice)
    h_explicit, v_explicit = _marginal_entropies(data_slice, nbins=expected_nbins)

    np.testing.assert_array_equal(h_default, h_explicit)
    np.testing.assert_array_equal(v_default, v_explicit)


def test_marginal_entropies_matches_histogram_rederivation(data_slice):
    n_channels = 3
    n_samples = data_slice.shape[1]
    nbins = round(3 * math.log2(1 + n_samples / 10))

    h_impl, v_impl = _marginal_entropies(data_slice[:n_channels], nbins=nbins)

    for ch in range(n_channels):
        row = data_slice[ch]
        umin, umax = row.min(), row.max()
        delta = (umax - umin) / nbins
        counts, _ = np.histogram(row, bins=nbins, range=(umin, umax))
        counts = counts[counts > 0]
        p = counts / n_samples
        h_uncorrected = -np.sum(p * np.log(p))
        v_ref = np.sum(p * np.log(p) ** 2) - h_uncorrected**2
        h_ref = h_uncorrected + (nbins - 1) / (2 * n_samples) + np.log(delta)
        np.testing.assert_allclose(h_impl[ch], h_ref, rtol=1e-2)
        # Variance (a squared-log-probability term) is more sensitive than
        # entropy to the bin-boundary differences between the two
        # independent binning schemes, so it needs a looser tolerance.
        np.testing.assert_allclose(v_impl[ch], v_ref, rtol=3e-2)


def test_mir_identity_unmixing_is_approximately_zero(data_slice):
    n, n_samples = data_slice.shape
    _, vx = _marginal_entropies(data_slice)

    mir_val, variance = mir(np.eye(n), data_slice)

    assert abs(mir_val) < 1e-9
    expected_variance = 2 * np.sum(vx) / n_samples
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-9)


def test_mir_scaled_identity_matches_closed_form(data_slice):
    n, n_samples = data_slice.shape
    c = 2.5
    _, vx = _marginal_entropies(data_slice)

    mir_val, variance = mir(c * np.eye(n), data_slice)

    assert abs(mir_val) < 1e-9
    expected_variance = 2 * np.sum(vx) / n_samples
    np.testing.assert_allclose(variance, expected_variance, rtol=1e-9)


def test_mir_returns_finite_scalars_for_full_channel_data(real_data):
    mir_val, variance = mir(np.eye(NW), real_data)
    assert isinstance(mir_val, float)
    assert isinstance(variance, float)
    assert math.isfinite(mir_val)
    assert math.isfinite(variance)


def test_mir_real_fitted_unmixing_is_large_and_positive(data_slice, real_unmixing):
    """A real, non-trivial (non-diagonal) unmixing should show substantial MIR.

    Exercises the metric's actual purpose -- unlike the identity/scaled-identity
    tests above, which are information-preserving by construction and can only
    ever produce MIR ~= 0.
    """
    mir_val, variance = mir(real_unmixing, data_slice)
    identity_mir, _ = mir(np.eye(real_unmixing.shape[0]), data_slice)

    assert math.isfinite(variance)
    assert mir_val > 1.0
    assert mir_val > identity_mir


def test_mir_nbins_passthrough_matches_manual_computation(data_slice, real_unmixing):
    nbins = 20
    hx, vx = _marginal_entropies(data_slice, nbins=nbins)
    y = real_unmixing @ data_slice
    hy, vy = _marginal_entropies(y, nbins=nbins)
    eigvals = np.linalg.eigvals(real_unmixing)
    expected_mir = float(np.sum(np.log(np.abs(eigvals))) + np.sum(hx) - np.sum(hy))
    expected_variance = float((np.sum(vx) + np.sum(vy)) / data_slice.shape[1])

    mir_val, variance = mir(real_unmixing, data_slice, nbins=nbins)

    assert mir_val == pytest.approx(expected_mir)
    assert variance == pytest.approx(expected_variance)


def test_mir_raises_on_singular_unmixing(data_slice, real_unmixing):
    singular = real_unmixing.copy()
    singular[1] = singular[0]

    with pytest.raises(ValueError, match="singular"):
        mir(singular, data_slice)


def test_mir_raises_on_non_square_unmixing(data_slice):
    non_square = np.eye(data_slice.shape[0])[:16]

    with pytest.raises(np.linalg.LinAlgError):
        mir(non_square, data_slice)


def test_mir_raises_on_non_finite_data(data_slice):
    contaminated = data_slice.copy()
    contaminated[0, 5] = np.nan

    with pytest.raises(ValueError, match="non-finite"):
        mir(np.eye(contaminated.shape[0]), contaminated)


def test_marginal_entropies_raises_on_constant_channel(data_slice):
    flat = data_slice.copy()
    flat[0] = 0.0

    with pytest.raises(ValueError, match="constant"):
        _marginal_entropies(flat)
