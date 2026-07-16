"""Unit tests for ``pyAMICA.metrics.mir`` (issue #134, MIR port phase 1).

MIR is backend-agnostic pure NumPy, so it lives directly under ``pyAMICA/tests/``
(like ``test_amari_distance.py``) rather than ``tests/torch_tests/``. Per the
NO-MOCK policy, every test exercises the real bundled sample EEG data; none use
synthetic/random data as ground truth.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from pyAMICA.metrics.mir import _marginal_entropies, mir
from pyAMICA.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parents[1] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
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

    h_impl, _ = _marginal_entropies(data_slice[:n_channels], nbins=nbins)

    for ch in range(n_channels):
        row = data_slice[ch]
        umin, umax = row.min(), row.max()
        delta = (umax - umin) / nbins
        counts, _ = np.histogram(row, bins=nbins, range=(umin, umax))
        counts = counts[counts > 0]
        p = counts / n_samples
        h_ref = -np.sum(p * np.log(p)) + (nbins - 1) / (2 * n_samples) + np.log(delta)
        np.testing.assert_allclose(h_impl[ch], h_ref, rtol=1e-2)


def test_mir_identity_unmixing_is_approximately_zero(data_slice):
    n = data_slice.shape[0]
    mir_val, _ = mir(np.eye(n), data_slice)
    assert abs(mir_val) < 1e-6


def test_mir_scaled_identity_matches_closed_form(data_slice):
    n, n_samples = data_slice.shape
    c = 2.5
    hx, vx = _marginal_entropies(data_slice)

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
