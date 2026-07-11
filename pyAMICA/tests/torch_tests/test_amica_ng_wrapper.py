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
    # final_ll_ (the fitted model's LL, issue #51) is populated and survives the
    # round-trip -- use it, not ll_history_[-1], as the model's log-likelihood.
    assert fitted_ng.final_ll_ is not None
    assert loaded.final_ll_ == fitted_ng.final_ll_
    # converged_/stop_reason_ (issue #50) round-trip too (a saved model is always
    # converged, since state_dict refuses degenerate ones).
    assert loaded.converged_ == fitted_ng.converged_
    assert loaded.stop_reason_ == fitted_ng.stop_reason_
    assert loaded.converged_ is True

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


# --- EEGLAB drop-in output (issue #92) -------------------------------------
# Files EEGLAB's loadmodout15.m / the numpy port loadmodout() read.
_AMICAOUT_FILES = ("gm", "W", "S", "mean", "c", "alpha", "mu", "sbeta", "rho",
                   "comp_list", "LL")  # fmt: skip


def test_write_amica_output_requires_fit(tmp_path):
    """An unfit model must refuse to write output, mirroring save() (#50)."""
    model = AMICA(verbose=False)
    with pytest.raises(ValueError, match="fitted"):
        model.write_amica_output(str(tmp_path / "amicaout"))


def test_write_amica_output_bytes(fitted_ng, tmp_path):
    """The written files are the model's exact float64 parameters: the on-disk
    EEGLAB directory is a lossless serialization, not a lossy export (#92).
    Convention-free (compares raw bytes, no reader in the loop)."""
    outdir = tmp_path / "amicaout"
    fitted_ng.write_amica_output(str(outdir))
    ng = fitted_ng.model_

    def _read(name, dtype=np.float64):
        return np.fromfile(outdir / name, dtype=dtype)

    for name, attr in [
        ("gm", ng.gm), ("W", ng.W), ("S", ng.sphere), ("mean", ng.mean),
        ("c", ng.c), ("alpha", ng.alpha), ("mu", ng.mu), ("sbeta", ng.beta),
        ("rho", ng.rho),
    ]:  # fmt: skip
        np.testing.assert_array_equal(
            _read(name).reshape(attr.shape), attr.cpu().numpy(), err_msg=name
        )
    # comp_list is written 1-based int32.
    np.testing.assert_array_equal(
        _read("comp_list", np.int32).reshape(ng.comp_list.shape),
        ng.comp_list.cpu().numpy() + 1,
    )
    np.testing.assert_array_equal(
        _read("LL"), np.asarray(ng.ll_history, dtype=np.float64)
    )


def test_write_amica_output_loadmodout_readable(fitted_ng, tmp_path):
    """A PyTorch NG fit written with write_amica_output() is a directory the
    EEGLAB reader (loadmodout / loadmodout15) loads, with the expected shapes
    and EEGLAB's back-projected-variance ordering applied (issue #92)."""
    from pyAMICA.numpy_impl.load import loadmodout

    outdir = tmp_path / "amicaout"
    fitted_ng.write_amica_output(str(outdir))
    for name in _AMICAOUT_FILES:
        assert (outdir / name).exists(), f"missing {name}"

    mod = loadmodout(outdir)
    assert mod.num_models == 1
    assert mod.W.shape == (NW, NW, 1)
    assert mod.A.shape == (NW, NW, 1)
    assert mod.S.shape == (NW, NW)
    # origord holds the fit-order indices sorted by descending back-projected
    # variance (EEGLAB IC1 = highest): svar taken in that order is non-increasing.
    assert np.all(np.diff(mod.svar[mod.origord[:, 0], 0]) <= 1e-9)


def test_variance_order(fitted_ng):
    """variance_order() returns a permutation of the sources ranked by descending
    back-projected variance (EEGLAB IC1 = highest), for use in Python without a
    disk round-trip (issue #92)."""
    order, svar = fitted_ng.variance_order(return_svar=True)
    assert sorted(order.tolist()) == list(range(NW))  # a permutation
    assert np.all(np.diff(svar) <= 1e-9)  # descending


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


def test_fit_exposes_converged_and_stop_reason(fitted_ng):
    """A normal fit is marked usable and exposes its stop reason (issue #50):
    converged_ is True, is_fitted_ is True, and stop_reason_ is a non-degenerate
    marker."""
    assert fitted_ng.converged_ is True
    assert fitted_ng.is_fitted_ is True
    assert fitted_ng.stop_reason_ not in ("nan_ll", "singular_ll")
    assert fitted_ng.stop_reason_ is not None


def test_unfitted_output_raises_not_fitted():
    """Before any fit, the output methods raise a clear 'not fitted' error --
    distinct from the degenerate-fit refusal below (issue #50)."""
    model = AMICA(verbose=False)
    assert model.is_fitted_ is False
    with pytest.raises(ValueError, match="fitted"):
        model.transform(np.zeros((NW, 16)))
    with pytest.raises(ValueError, match="fitted"):
        model.get_unmixing_matrix()


def test_degenerate_fit_refuses_output(real_data, tmp_path, caplog):
    """A genuinely degenerate fit is marked unusable and every output method
    refuses it rather than return NaN sources (issue #50). A single NaN injected
    into the real EEG forces an actual ``nan_ll`` divergence in the backend --
    an error-path robustness test, not a parity/correctness claim, so it is not
    a synthetic-data oracle -- exercising the real ``fit`` bookkeeping (not a
    forced marker): is_fitted_/converged_ False, stop_reason_ named, and the
    wrapper warning emitted."""
    bad = real_data[:, :4096].copy()
    bad[0, 0] = np.nan  # propagates to a nan_ll stop in the backend
    model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    with caplog.at_level(logging.WARNING, logger="pyAMICA.amica"):
        model.fit(bad, max_iter=3, block_size=1024, seed=0)

    assert model.stop_reason_ == "nan_ll"
    assert model.converged_ is False
    assert model.is_fitted_ is False
    assert any("degenerate" in r.getMessage() for r in caplog.records)

    # transform/get_*/save refuse the degenerate model with a diagnosable error
    # (names the stop_reason), not a misleading plain "not fitted".
    for action in (
        lambda: model.transform(real_data[:, :512]),
        lambda: model.get_mixing_matrix(),
        lambda: model.get_unmixing_matrix(),
        lambda: model.save(str(tmp_path / "degenerate.pt")),
    ):
        with pytest.raises(RuntimeError, match="degenerate.*nan_ll"):
            action()

    # fit_transform routes through the guarded transform, so a degenerate refit
    # cannot leak NaN sources either.
    with pytest.raises(RuntimeError, match="degenerate"):
        AMICA(n_models=1, n_mix=3, device="cpu", verbose=False).fit_transform(
            bad, max_iter=3, block_size=1024, seed=0
        )
