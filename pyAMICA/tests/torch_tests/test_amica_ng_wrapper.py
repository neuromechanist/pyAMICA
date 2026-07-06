"""Tests for the public ``AMICA`` wrapper over ``AMICATorchNG``.

These cover the wiring the wrapper adds on top of ``AMICATorchNG``: the
save/load round-trip (issue #36) and the device-selection fallback that keeps
the float64 parity default from crashing on Apple Silicon (MPS cannot
represent float64). Real sample EEG data only (no synthetic/mock).
"""

import logging
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
    model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    model.fit(real_data[:, :4096], max_iter=3, block_size=1024, seed=42)
    return model


def test_ng_save_requires_fit(tmp_path):
    model = AMICA(verbose=False)
    with pytest.raises(ValueError, match="fitted"):
        model.save(str(tmp_path / "model.pt"))


def test_ng_save_load_roundtrip(fitted_ng, real_data, tmp_path):
    """fit -> save -> load reconstructs a transform-ready model that reproduces
    the original mixing/unmixing matrices and source estimates exactly."""
    path = str(tmp_path / "model.pt")
    fitted_ng.save(path)
    assert Path(path).exists()

    loaded = AMICA.load(path, device="cpu")

    assert loaded.is_fitted_
    assert loaded.n_models == fitted_ng.n_models
    assert loaded.n_mix == fitted_ng.n_mix
    assert loaded.ll_history_ == fitted_ng.ll_history_

    # torch.save/load restores tensors bit-exactly and CPU matmul is
    # deterministic, so transform() on the restored tensors reproduces the
    # original output exactly (not guaranteed on non-deterministic GPU reductions).
    np.testing.assert_array_equal(
        loaded.get_mixing_matrix(), fitted_ng.get_mixing_matrix()
    )
    np.testing.assert_array_equal(
        loaded.get_unmixing_matrix(), fitted_ng.get_unmixing_matrix()
    )

    block = real_data[:, :4096]
    np.testing.assert_array_equal(loaded.transform(block), fitted_ng.transform(block))


def test_ng_load_rejects_unknown_version(fitted_ng, tmp_path):
    """A payload with an unexpected format_version must fail loudly, not load a
    half-formed model (no silent-failure)."""
    path = str(tmp_path / "model.pt")
    fitted_ng.save(path)
    payload = torch.load(path, weights_only=True)
    payload["format_version"] = 99
    torch.save(payload, path)

    with pytest.raises(ValueError, match="format_version"):
        AMICA.load(path)


def test_ng_default_device_avoids_mps_float64(real_data, caplog):
    """The default float64 NG config must not crash when the auto-selected
    device is MPS; the wrapper falls back to CPU (regression for #29)."""
    model = AMICA(n_models=1, n_mix=3, verbose=False)  # device=None
    with caplog.at_level(logging.WARNING, logger="pyAMICA.amica"):
        model.fit(real_data[:, :2048], max_iter=2, block_size=1024, seed=42)

    # float64 parity runs must never land on MPS.
    assert model.model_.device.type in ("cpu", "cuda")
    if torch.backends.mps.is_available():
        assert model.model_.device.type == "cpu"
        # The downgrade must be announced even with verbose=False (not silent).
        assert any("float64" in r.message for r in caplog.records)


def test_ng_explicit_mps_float64_raises(real_data):
    """A user-pinned device="mps" with the default float64 must NOT be
    silently coerced to CPU; it should surface AMICATorchNG's ValueError.
    Raised at construction (before device placement), so no MPS hardware
    is needed."""
    model = AMICA(device="mps", verbose=False)
    with pytest.raises(ValueError, match="MPS does not support float64"):
        model.fit(real_data[:, :256], max_iter=1, block_size=128, seed=1)


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="requires MPS hardware"
)
def test_ng_mps_float32_escape_hatch(real_data):
    """The documented workaround: dtype=torch.float32 lets the NG backend run
    on MPS."""
    model = AMICA(device="mps", verbose=False)
    model.fit(
        real_data[:, :2048], max_iter=2, block_size=1024, seed=42, dtype=torch.float32
    )
    assert model.model_.device.type == "mps"
    assert model.model_.dtype == torch.float32


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
