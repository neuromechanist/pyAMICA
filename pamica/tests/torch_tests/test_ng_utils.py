"""Unit tests for ``pamica.torch_impl.utils`` helpers.

These exercise the preprocessing / device / comparison utilities that back the
PyTorch path (previously ~20% covered). They are pure-function tests of linear
algebra and device selection, not AMICA-vs-Fortran correctness checks (those
live in the parity suite); where a matrix is needed it is derived from the real
sample EEG rather than fabricated, per the NO-MOCK policy. Tensors carrying a
literal NaN/Inf are inherent to testing the numerical-stability guards.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pamica.torch_impl.utils import (
    check_numerical_stability,
    compute_correlation_matrix,
    find_best_permutation,
    load_eeglab_data,
    setup_device,
    stabilize_log_probabilities,
)

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
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


# --- setup_device -----------------------------------------------------------


def test_setup_device_cpu_explicit():
    assert setup_device("cpu").type == "cpu"


def test_setup_device_auto_returns_valid_device():
    dev = setup_device()
    assert dev.type in {"cpu", "cuda", "mps"}


def test_setup_device_cuda_falls_back_when_unavailable():
    dev = setup_device("cuda")
    # Honors the request only if CUDA is actually present; otherwise CPU.
    if torch.cuda.is_available():
        assert dev.type == "cuda"
    else:
        assert dev.type == "cpu"


def test_setup_device_mps_falls_back_when_unavailable():
    dev = setup_device("mps")
    if torch.backends.mps.is_available():
        assert dev.type == "mps"
    else:
        assert dev.type == "cpu"


# --- check_numerical_stability ---------------------------------------------


def test_check_numerical_stability_clean():
    assert check_numerical_stability(torch.tensor([1.0, 2.0, 3.0])) is True


def test_check_numerical_stability_flags_nan():
    assert check_numerical_stability(torch.tensor([1.0, float("nan")])) is False


def test_check_numerical_stability_flags_inf():
    assert check_numerical_stability(torch.tensor([1.0, float("inf")])) is False


def test_check_numerical_stability_raises_when_requested():
    with pytest.raises(ValueError, match="NaN"):
        check_numerical_stability(
            torch.tensor([float("nan")]), name="w", raise_on_error=True
        )


# --- stabilize_log_probabilities -------------------------------------------


def test_stabilize_log_probabilities_clamps_below_floor():
    lp = torch.tensor([-2000.0, -100.0, 0.0])
    out = stabilize_log_probabilities(lp)
    assert out[0].item() == -1500.0  # clamped to default floor
    assert out[1].item() == -100.0  # above floor, untouched
    assert out[2].item() == 0.0


def test_stabilize_log_probabilities_custom_floor():
    lp = torch.tensor([-50.0, -5.0])
    out = stabilize_log_probabilities(lp, min_log=-10.0)
    assert out[0].item() == -10.0
    assert out[1].item() == -5.0


# --- compute_correlation_matrix / find_best_permutation ---------------------


def test_compute_correlation_matrix_identity_on_self(real_data):
    # Two identical component sets: absolute correlation diagonal must be ~1.
    W = real_data[:8, :512]
    corr = compute_correlation_matrix(W, W)
    assert corr.shape == (8, 8)
    np.testing.assert_allclose(np.diag(corr), np.ones(8), atol=1e-8)


def test_find_best_permutation_recovers_shuffle_and_sign(real_data):
    # Permute + sign-flip real component vectors, then recover the mapping.
    W1 = real_data[:6, :512]
    perm = np.array([2, 0, 5, 1, 4, 3])
    signs = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    W2 = (W1[perm].T * signs).T
    corr = compute_correlation_matrix(W1, W2, absolute=True)
    col_ind, rec_signs = find_best_permutation(corr)
    # W2 row j corresponds to W1 row perm[j]; the assignment must invert it.
    np.testing.assert_array_equal(col_ind[perm], np.arange(6))
    assert set(rec_signs) <= {-1.0, 1.0}


# --- load_eeglab_data -------------------------------------------------------


def test_load_eeglab_data_shape(real_data):
    assert real_data.shape == (NW, FIELD)
    assert np.isfinite(real_data).all()
