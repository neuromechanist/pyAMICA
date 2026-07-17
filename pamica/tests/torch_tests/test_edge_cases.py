"""Edge-case and input-validation hardening for the AMICA wrapper (issue #61).

Covers degenerate data shapes (single channel, single sample), input
validation, and the ``from_params_file`` / ``load`` error branches. Real sample
EEG only (thin slices), never synthetic data. The contract under test is
*graceful* behavior: a clear exception or a model flagged unusable
(``is_fitted_``/``converged_`` False), never a silent NaN model (issue #50).
"""

from pathlib import Path

import numpy as np
import pytest

from pamica.amica import AMICA
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
PARAMS_FILE = SAMPLE_DIR / "sample_params.json"
NW = 32
FIELD = 30504


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def test_fit_rejects_non_2d_input():
    """A 1-D array must fail loudly at the shape check, not be misinterpreted."""
    model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    with pytest.raises(ValueError, match="2D"):
        model.fit(np.zeros(10))


def test_single_channel_fit_is_graceful(real_data):
    """A single-channel fit must not crash or silently return a NaN model."""
    model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    x = real_data[:1, :2048]
    try:
        model.fit(x, max_iter=3, block_size=1024, seed=0)
    except (ValueError, RuntimeError):
        return  # a clear refusal is acceptable graceful behavior
    if model.is_fitted_:
        assert np.isfinite(model.get_unmixing_matrix()).all()
        assert np.isfinite(model.get_mixing_matrix()).all()
    else:
        # Not fitted => flagged unusable, not a silent NaN model.
        assert model.converged_ is False


def test_single_sample_is_graceful(real_data):
    """One time point => singular covariance; must refuse, not produce a usable
    NaN model."""
    model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    x = real_data[:, :1]
    try:
        # torch.linalg.LinAlgError subclasses RuntimeError, so it is covered.
        model.fit(x, max_iter=3, seed=0)
    except (ValueError, RuntimeError):
        return  # clean failure is acceptable
    assert not model.is_fitted_  # otherwise it must be flagged unusable


def test_from_params_file_reads_and_overrides():
    """AMICA.from_params_file picks up JSON settings and honors kwarg overrides."""
    if not PARAMS_FILE.exists():
        pytest.skip("sample params missing")
    model = AMICA.from_params_file(str(PARAMS_FILE))
    assert model.n_models == 1
    assert model.n_mix == 3
    # kwargs override the file.
    overridden = AMICA.from_params_file(str(PARAMS_FILE), n_mix=5)
    assert overridden.n_mix == 5


def test_load_rejects_missing_backend_key(real_data, tmp_path):
    """A save file missing the 'backend' section must fail loudly on load."""
    import torch

    model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    model.fit(real_data[:, :2048], max_iter=3, block_size=1024, seed=0)
    path = str(tmp_path / "model.pt")
    model.save(path)

    payload = torch.load(path, weights_only=True)
    del payload["backend"]
    torch.save(payload, path)

    with pytest.raises(ValueError, match="missing"):
        AMICA.load(path)
