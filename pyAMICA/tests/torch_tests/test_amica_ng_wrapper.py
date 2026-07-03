"""Tests for the public ``AMICA`` wrapper driving ``backend="ng"``.

These cover the wiring the wrapper adds on top of ``AMICATorchNG``: the
constructor/fit guards, the save/load NotImplementedError, and the
device-selection fallback that keeps the float64 NG default from crashing
on Apple Silicon (MPS cannot represent float64). Real sample EEG data only
(no synthetic/mock).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pyAMICA.amica import AMICA

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504


def _load_real_data() -> np.ndarray:
    from pyAMICA.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    return _load_real_data()


@pytest.fixture(scope="module")
def fitted_ng(real_data) -> AMICA:
    """A small real-data NG fit reused across the fitted-model assertions."""
    model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False, backend="ng")
    model.fit(real_data[:, :4096], max_iter=3, block_size=1024, seed=42)
    return model


def test_backend_invalid_raises():
    with pytest.raises(ValueError, match="backend must be"):
        AMICA(backend="bogus")


def test_ng_debug_guard(real_data):
    model = AMICA(device="cpu", verbose=False, backend="ng")
    with pytest.raises(ValueError, match="debug=True"):
        model.fit(real_data[:, :256], max_iter=1, debug=True)


def test_ng_output_dir_guard(real_data, tmp_path):
    model = AMICA(device="cpu", verbose=False, backend="ng")
    with pytest.raises(ValueError, match="output_dir"):
        model.fit(real_data[:, :256], max_iter=1, output_dir=str(tmp_path))


def test_ng_load_not_implemented(tmp_path):
    model = AMICA(verbose=False, backend="ng")
    with pytest.raises(NotImplementedError, match="load"):
        model.load(str(tmp_path / "model.pt"))


def test_ng_save_not_implemented(fitted_ng, tmp_path):
    with pytest.raises(NotImplementedError, match="save"):
        fitted_ng.save(str(tmp_path / "model.pt"))


def test_ng_default_device_avoids_mps_float64(real_data):
    """The default float64 NG config must not crash when the auto-selected
    device is MPS; the wrapper falls back to CPU (regression for #29)."""
    model = AMICA(n_models=1, n_mix=3, verbose=False, backend="ng")  # device=None
    model.fit(real_data[:, :2048], max_iter=2, block_size=1024, seed=42)

    # float64 parity runs must never land on MPS.
    assert model.model_.device.type in ("cpu", "cuda")
    if torch.backends.mps.is_available():
        assert model.model_.device.type == "cpu"


def test_ng_wrapper_fit_transform_real_data(fitted_ng, real_data):
    assert fitted_ng.is_fitted_
    assert len(fitted_ng.ll_history_) >= 1

    S = fitted_ng.transform(real_data[:, :4096])
    assert S.shape == (NW, 4096)
    assert np.isfinite(S).all()

    A = fitted_ng.get_mixing_matrix()
    W = fitted_ng.get_unmixing_matrix()
    assert A.shape == (NW, NW)
    assert W.shape == (NW, NW)
    assert np.isfinite(A).all()
    assert np.isfinite(W).all()
