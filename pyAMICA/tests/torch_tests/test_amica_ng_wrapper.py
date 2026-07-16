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
    assert model.model_ is not None
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
    assert model.model_ is not None
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


def test_variance_order_requires_fit():
    """variance_order also refuses an unfit model (same _check_usable guard)."""
    model = AMICA(verbose=False)
    with pytest.raises(ValueError, match="fitted"):
        model.variance_order()


def test_write_amica_output_bytes(fitted_ng, tmp_path):
    """The written files are the model's exact float64 parameters: the on-disk
    EEGLAB directory is a lossless serialization, not a lossy export (#92). W and
    the symmetric sphere are byte-identical in C order; the non-square mixture
    params and c/comp_list are column-major (Fortran layout), so read order="F".
    """
    outdir = tmp_path / "amicaout"
    fitted_ng.write_amica_output(str(outdir))
    ng = fitted_ng.model_

    for name, attr in [("gm", ng.gm), ("W", ng.W), ("S", ng.sphere),
                       ("mean", ng.mean)]:  # fmt: skip
        got = np.fromfile(outdir / name).reshape(attr.shape)  # C order
        np.testing.assert_array_equal(got, attr.cpu().numpy(), err_msg=name)
    for name, attr in [("c", ng.c), ("alpha", ng.alpha), ("mu", ng.mu),
                       ("sbeta", ng.beta), ("rho", ng.rho)]:  # fmt: skip
        got = np.fromfile(outdir / name).reshape(attr.shape, order="F")
        np.testing.assert_array_equal(got, attr.cpu().numpy(), err_msg=name)
    # comp_list is written 1-based int32, column-major.
    np.testing.assert_array_equal(
        np.fromfile(outdir / "comp_list", np.int32).reshape(
            ng.comp_list.shape, order="F"
        ),
        ng.comp_list.cpu().numpy() + 1,
    )
    # LL ends at the exported (kept) iterate; the fixture fit is monotone, so the
    # full trajectory is written.
    ll = np.fromfile(outdir / "LL")
    assert ll[-1] == ng.final_ll_
    np.testing.assert_array_equal(
        ll, np.asarray(ng.ll_history[: len(ll)], dtype=np.float64)
    )


def test_write_amica_output_loadmodout_readable(fitted_ng, tmp_path):
    """A PyTorch NG fit written with write_amica_output() is a directory the
    EEGLAB reader (loadmodout / loadmodout15) loads with the expected shapes and
    correct Fortran (column-major) mixture-param layout (issue #92)."""
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
    # Mixture proportions per component must sum to 1: a meaningful check that the
    # (num_mix, n_comp) params were read back with the correct column-major layout
    # (a C-order write would scramble them and break this).
    np.testing.assert_allclose(mod.alpha[:, :, 0].sum(axis=0), 1.0, atol=1e-9)


def test_variance_order(fitted_ng):
    """variance_order() returns a permutation of the sources ranked by descending
    back-projected variance (EEGLAB IC1 = highest), for use in Python without a
    disk round-trip (issue #92)."""
    order, svar = fitted_ng.variance_order(return_svar=True)
    assert sorted(order.tolist()) == list(range(NW))  # a permutation
    assert np.all(np.diff(svar) <= 1e-9)  # descending


def test_write_amica_output_ll_matches_kept_iterate(real_data, tmp_path):
    """When keep_best (#51) restores an earlier iterate, the written LL trajectory
    ends at that kept iterate (LL[-1] == final_ll_), not at a later discarded
    overshoot -- so a user reading mod.LL(end) in EEGLAB sees the loaded model's
    likelihood (review finding, #92)."""
    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(
        real_data[:, :4096],
        max_iter=60,
        do_newton=True,
        newt_start=1,
        lrate=0.5,
        seed=0,
        block_size=1024,
    )
    if not model.is_fitted_:
        pytest.skip("aggressive run ended degenerate; not the case under test")
    ng = model.model_
    assert ng is not None and ng.final_ll_ is not None
    if np.isclose(ng.ll_history[-1], ng.final_ll_):
        pytest.skip("run was monotone; keep_best restore did not fire")

    outdir = tmp_path / "amicaout"
    model.write_amica_output(str(outdir))
    ll = np.fromfile(outdir / "LL")
    assert np.isclose(ll[-1], ng.final_ll_)  # ends at the kept iterate
    assert len(ll) < len(ng.ll_history)  # the later overshoot is dropped


def test_write_amica_output_llt_roundtrip(fitted_ng, tmp_path):
    """A real (short but genuine) single-model NG fit writes an ``LLt`` file
    that round-trips through ``loadmodout`` (issue #155).

    ``Lht[0]`` must equal ``Lt`` exactly for a single model. ``Lt.mean()`` (the
    per-sample total log-density, summed over channels) should be close to
    ``nw * final_ll_`` -- the NG backend's ``final_ll_`` is already the
    per-sample-per-channel normalized log-likelihood, matching the Fortran
    ``LL`` file convention directly.
    """
    from pyAMICA.numpy_impl.load import loadmodout

    outdir = tmp_path / "amicaout"
    fitted_ng.write_amica_output(str(outdir))

    out = loadmodout(outdir)
    assert out.Lht is not None and out.Lt is not None
    assert out.Lht.shape == (1, 4096)
    np.testing.assert_array_equal(out.Lht[0], out.Lt)

    assert fitted_ng.final_ll_ is not None
    np.testing.assert_allclose(out.Lt.mean(), NW * fitted_ng.final_ll_, rtol=1e-2)


def test_write_amica_output_llt_multimodel(real_data, tmp_path):
    """A small real 2-model NG fit's ``LLt`` satisfies its definitional
    relationship: the total per-sample log-likelihood is the log-sum-exp of
    the per-model log-likelihoods over models (issue #155). Few iterations --
    this exercises the multi-model LLt code path, not convergence."""
    from pyAMICA.numpy_impl.load import loadmodout

    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(real_data[:, :4096], max_iter=5, block_size=1024, seed=7)

    outdir = tmp_path / "amicaout"
    model.write_amica_output(str(outdir))

    out = loadmodout(outdir)
    assert out.Lht is not None and out.Lt is not None
    assert out.Lht.shape == (2, 4096)

    from scipy.special import logsumexp

    np.testing.assert_allclose(out.Lt, logsumexp(out.Lht, axis=0), rtol=1e-10)


def test_loadmodout_llt_gm_reorder_alignment(real_data, tmp_path):
    """``loadmodout`` must permute ``Lht`` by the SAME ``gm_ord`` it applies to
    ``W``/``mod_prob``/``comp_list``/etc, not leave it in on-disk order (issue
    #155 review Addition B).

    ``test_write_amica_output_llt_multimodel``'s ``Lt == logsumexp(Lht,
    axis=0)`` check is permutation-invariant over the model axis, so it
    cannot detect a model-axis misalignment between ``Lht`` and ``W``/``gm``.
    This test instead reads the raw on-disk ``gm``/``LLt`` bytes directly,
    computes the permutation by hand, and checks ``loadmodout``'s ``Lht``
    against it.

    Requires a genuinely non-identity ``gm_ord`` (model 1 outweighing model
    0), or this degenerates into a tautology that passes regardless of
    whether the permutation is applied; seed=4 gives exactly that on real
    sample EEG.
    """
    from pyAMICA.numpy_impl.load import loadmodout

    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(real_data[:, :4096], max_iter=8, block_size=1024, seed=4)

    outdir = tmp_path / "amicaout"
    model.write_amica_output(str(outdir))

    gm_raw = np.fromfile(outdir / "gm", dtype=np.float64)
    num_models = gm_raw.size
    gm_ord = np.argsort(gm_raw)[::-1]
    assert not np.array_equal(gm_ord, np.arange(num_models)), (
        "fixture must give a non-identity gm ordering, or this test is a tautology"
    )

    LLt_raw = np.fromfile(outdir / "LLt", dtype=np.float64)
    n_samples = LLt_raw.size // (num_models + 1)
    LLt_raw = LLt_raw.reshape(num_models + 1, n_samples, order="F")
    Lht_raw = LLt_raw[:num_models]

    out = loadmodout(outdir)
    assert out.Lht is not None
    np.testing.assert_array_equal(out.Lht, Lht_raw[gm_ord])
    assert not np.array_equal(out.Lht, Lht_raw), (
        "Lht must actually be permuted by gm_ord, not left in on-disk order"
    )


def test_write_amica_output_llt_reject_zeroes_rejected_samples(real_data, tmp_path):
    """Under ``do_reject``, rejected samples must be exactly zero in the
    written ``LLt`` (issue #155 Fix 1): Fortran zeroes a rejected sample's
    ``modloglik``/``loglik`` on write (amica15.f90:2211-2216) and its
    ``load_rej`` reader reconstructs the rejection mask from that exact zero
    sentinel (``sum(modloglik(:,i)) == 0.0``, amica15.f90:887-896). Good
    samples must stay non-zero and finite.
    """
    from pyAMICA.numpy_impl.load import loadmodout

    model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    model.fit(
        real_data[:, :4096],
        max_iter=15,
        block_size=1024,
        seed=1,
        do_reject=True,
        rejstart=2,
        rejint=3,
        maxrej=3,
        rejsig=2.0,
    )
    ng = model.model_
    assert ng is not None and ng.good_idx is not None
    assert ng.good_idx.numel() < 4096, "fixture must reject something"

    outdir = tmp_path / "amicaout"
    model.write_amica_output(str(outdir))
    out = loadmodout(outdir)
    assert out.Lht is not None and out.Lt is not None

    good = np.zeros(4096, dtype=bool)
    good[ng.good_idx.detach().cpu().numpy()] = True

    np.testing.assert_array_equal(out.Lht[:, ~good], 0.0)
    np.testing.assert_array_equal(out.Lt[~good], 0.0)
    assert np.all(out.Lht[:, good] != 0.0)
    assert np.all(np.isfinite(out.Lht[:, good]))
    assert np.all(out.Lt[good] != 0.0)
    assert np.all(np.isfinite(out.Lt[good]))


def test_write_amica_output_llt_partial_final_block(real_data, tmp_path):
    """The block loop's remainder branch (``end = min(start + block_size,
    n_samples)``) is exercised by a sample count NOT evenly divisible by
    ``block_size`` (issue #155 Fix 6d -- the existing LLt tests all used
    exactly-divisible counts, e.g. 4096/1024, so this branch was untested).
    """
    from pyAMICA.numpy_impl.load import loadmodout

    n = 4100
    assert n % 1024 != 0, "fixture must not be block_size-divisible"
    model = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    model.fit(real_data[:, :n], max_iter=5, block_size=1024, seed=1)

    outdir = tmp_path / "amicaout"
    model.write_amica_output(str(outdir))
    out = loadmodout(outdir)
    assert out.Lht is not None and out.Lt is not None
    assert out.Lht.shape == (1, n)
    np.testing.assert_array_equal(out.Lht[0], out.Lt)
    assert np.all(np.isfinite(out.Lt))


def test_from_state_dict_write_amica_output_omits_llt(fitted_ng, tmp_path, caplog):
    """A model restored via ``from_state_dict`` (the ``AMICA.load`` path) has
    no training data to recompute ``LLt`` from (issue #155 Fix 2/3): a warning
    must fire and no ``LLt`` file must be written, while the rest of the
    output is unaffected."""
    path = str(tmp_path / "model.pt")
    fitted_ng.save(path)
    loaded = AMICA.load(path, device="cpu")

    outdir = tmp_path / "amicaout"
    with caplog.at_level(logging.WARNING, logger="pyAMICA.torch_impl.core"):
        loaded.write_amica_output(str(outdir))

    assert not (outdir / "LLt").exists()
    assert (outdir / "W").exists()  # the rest of the output is unaffected
    assert any(
        "LLt" in r.getMessage() and "restored" in r.getMessage().lower()
        for r in caplog.records
    )


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
        lambda: model.write_amica_output(str(tmp_path / "degenerate_out")),
        lambda: model.variance_order(),
    ):
        with pytest.raises(RuntimeError, match="degenerate.*nan_ll"):
            action()

    # fit_transform routes through the guarded transform, so a degenerate refit
    # cannot leak NaN sources either.
    with pytest.raises(RuntimeError, match="degenerate"):
        AMICA(n_models=1, n_mix=3, device="cpu", verbose=False).fit_transform(
            bad, max_iter=3, block_size=1024, seed=0
        )
