"""Tests for the natural-gradient EM backend (AMICATorchNG).

Real sample EEG data only (no synthetic/mock). The decisive correctness
check is that the vectorized per-block sufficient statistics match the
validated NumPy reference (``AMICA_NumPy._get_block_updates``) to float64
precision on an identical block with identical parameters.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pyAMICA.torch_impl import AMICATorchNG
from pyAMICA.pyAMICA import AMICA as AMICA_NumPy

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42


def _load_real_data() -> np.ndarray:
    from pyAMICA.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _fresh_ng(block_size: int = 256, **kwargs) -> AMICATorchNG:
    return AMICATorchNG(
        n_channels=NW,
        n_models=1,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=block_size,
        **kwargs,
    )


def _numpy_ref_like(ng: AMICATorchNG, blk: int, **kwargs) -> AMICA_NumPy:
    """A NumPy AMICA carrying the NG backend's exact parameters (the
    established copy-params parity pattern)."""
    npm = AMICA_NumPy(num_models=1, num_mix=NMIX, **kwargs)
    npm.data_dim = NW
    npm.num_comps = NW
    npm.num_models = 1
    npm.num_mix = NMIX
    npm.block_size = blk
    npm.comp_list = ng.comp_list.cpu().numpy()
    npm.A = ng.A.cpu().numpy().copy()
    npm.W = ng.W.cpu().numpy().copy()
    npm.c = ng.c.cpu().numpy().copy()
    npm.mu = ng.mu.cpu().numpy().copy()
    npm.alpha = ng.alpha.cpu().numpy().copy()
    npm.beta = ng.beta.cpu().numpy().copy()
    npm.rho = ng.rho.cpu().numpy().copy()
    npm.gm = ng.gm.cpu().numpy().copy()
    return npm


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_sufficient_stats_match_numpy_reference():
    """Decisive parity: NG per-block sufficient statistics == NumPy reference.

    Copies the NG backend's exact parameters into a NumPy AMICA instance and
    compares ``_get_block_updates`` on an identical real-data block. These
    accumulators are what drive every M-step update, so machine-precision
    agreement proves the vectorized port is faithful. (``ll`` is excluded: the
    NG module intentionally computes the log-likelihood from pre-normalization
    mixture logits, which is more correct than the NumPy reference's
    post-normalization value; LL is checked against Fortran separately.)
    """
    data = _load_real_data()
    ng = _fresh_ng()
    X_t = ng._preprocess(data)
    ng._initialize_parameters()

    blk = 256
    block = X_t[:, :blk].contiguous()
    ng_upd = ng._get_block_updates(block)

    npm = AMICA_NumPy(num_models=1, num_mix=NMIX, do_newton=False)
    npm.data_dim = NW
    npm.num_comps = NW
    npm.num_models = 1
    npm.num_mix = NMIX
    npm.block_size = blk
    npm.comp_list = ng.comp_list.cpu().numpy()
    npm.A = ng.A.cpu().numpy().copy()
    npm.W = ng.W.cpu().numpy().copy()
    npm.c = ng.c.cpu().numpy().copy()
    npm.mu = ng.mu.cpu().numpy().copy()
    npm.alpha = ng.alpha.cpu().numpy().copy()
    npm.beta = ng.beta.cpu().numpy().copy()
    npm.rho = ng.rho.cpu().numpy().copy()
    npm.gm = ng.gm.cpu().numpy().copy()

    np_upd = npm._get_block_updates(block.cpu().numpy())

    for key in ["dgm", "dalpha", "dmu", "dbeta", "drho", "dA", "dc"]:
        a = np.asarray(ng_upd[key].cpu().numpy(), dtype=np.float64)
        b = np.asarray(np_upd[key], dtype=np.float64).reshape(a.shape)
        max_diff = float(np.max(np.abs(a - b)))
        assert max_diff < 1e-8, f"{key} differs from NumPy reference by {max_diff:.3e}"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_blocking_invariance():
    """Accumulated sufficient statistics are independent of block_size."""
    data = _load_real_data()[:, :4096]

    def accumulate(block_size):
        m = _fresh_ng(block_size=block_size)
        X_t = m._preprocess(data)
        m._initialize_parameters()
        return m._accumulate_blocks(X_t)

    acc_a = accumulate(256)
    acc_b = accumulate(512)
    for key in ["dgm", "dalpha", "dmu", "dbeta", "drho", "dA", "dc", "ll"]:
        a = acc_a[key].cpu().numpy()
        b = acc_b[key].cpu().numpy()
        assert np.allclose(a, b, atol=1e-8), f"{key} depends on block_size"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_reported_ll_includes_jacobian_near_fortran_at_init():
    """At initialization the reported per-sample-per-channel LL sits in
    Fortran's range (~ -3 to -4), which requires the log|det W| and sldet
    Jacobian terms to be present (without them it would be off by ~sldet/nw)."""
    data = _load_real_data()
    m = _fresh_ng()
    m.fit(data, max_iter=1, verbose=False)
    ll0 = m.ll_history[0]
    assert -6.0 < ll0 < -2.0, (
        f"init LL {ll0:.3f} outside Fortran range; Jacobian likely missing"
    )


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_fit_produces_finite_unmixing_on_real_data():
    """A short fit on real data yields a finite, correctly-shaped W (no NaN)."""
    data = _load_real_data()
    m = _fresh_ng()
    m.fit(data, max_iter=10, verbose=False)
    W = m.get_unmixing_matrix(0)
    assert W.shape == (NW, NW)
    assert np.all(np.isfinite(W))
    assert np.all(np.isfinite(m.ll_history))


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_newton_stats_match_numpy_reference():
    """Newton curvature statistics (sigma2, lambda, kappa) match the NumPy
    reference to float64 precision on an identical real-data block.

    Same copy-params pattern as the non-Newton parity test, with
    ``do_newton=True``. Both backends carry the Fortran-corrected Newton math
    (score ``fp`` rather than the density derivative, ``sbeta^2`` on kappa,
    the ``mu^2`` curvature term in lambda), so the finalized curvature stats
    that drive the Newton preconditioner must agree.
    """
    data = _load_real_data()
    ng = _fresh_ng(do_newton=True, newt_start=0)
    X_t = ng._preprocess(data)
    ng._initialize_parameters()

    blk = 256
    block = X_t[:, :blk].contiguous()
    ng_upd = ng._get_block_updates(block)
    sigma2_ng, lambda_ng, kappa_ng = ng._finalize_newton_stats(ng_upd)

    npm = _numpy_ref_like(ng, blk, do_newton=True)
    np_upd = npm._get_block_updates(block.cpu().numpy())
    dgm = np_upd["dgm"][None, :]
    refs = {
        "sigma2": (sigma2_ng, np_upd["dsigma2"] / dgm),
        "kappa": (kappa_ng, np_upd["dkappa"] / dgm),
        "lambda": (lambda_ng, np_upd["dlambda"] / dgm),
    }
    for name, (a_t, b_np) in refs.items():
        a = a_t.cpu().numpy()
        max_diff = float(np.max(np.abs(a - np.asarray(b_np).reshape(a.shape))))
        assert max_diff < 1e-8, f"{name} differs from NumPy reference by {max_diff:.3e}"


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_newton_mstep_matches_numpy_reference():
    """One full Newton M-step (``_update_parameters`` with Newton active)
    produces the same mixing matrix as the NumPy reference.

    On this early real-data block the curvature is not yet positive definite,
    so both backends take the natural-gradient fallback; the test verifies the
    two ports agree bit-for-bit on that path (finalization, the posdef guard,
    the fallback ramp, the A update) to float64 precision. The positive-definite
    ``H`` composition is covered separately by
    ``test_newton_posdef_mstep_matches_reference``.
    """
    data = _load_real_data()
    blk = 256
    it = 5  # >= newt_start so Newton is active

    ng = _fresh_ng(do_newton=True, newt_start=0, newtrate=0.5)
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    ng.iteration = it
    block = X_t[:, :blk].contiguous()
    acc = ng._get_block_updates(block)
    ng._update_parameters(acc, blk)
    A_ng = ng.A.cpu().numpy().copy()

    ng2 = _fresh_ng(do_newton=True, newt_start=0, newtrate=0.5)
    ng2._preprocess(data)
    ng2._initialize_parameters()
    npm = _numpy_ref_like(
        ng2, blk, do_newton=True, newt_start=0, newtrate=0.5, lrate=ng2.lrate0
    )
    npm.num_samples = blk
    npm.iter = it
    npm.doscaling = True
    npm.scalestep = 1
    npm.nd = []
    npm.ll = []
    npm.use_grad_norm = True
    np_upd = npm._get_block_updates(block.cpu().numpy())
    npm._update_parameters(np_upd)

    assert np.max(np.abs(A_ng - npm.A)) < 1e-8


def test_newton_direction_matches_formula():
    """``_newton_direction`` reproduces the Fortran 2x2 solve and its
    positive-definiteness guard on controlled inputs (no data needed)."""
    rng = np.random.RandomState(0)
    n = 6
    dA = torch.from_numpy(rng.randn(n, n))
    # Large sigma2/kappa so sk1*sk2 > 1 for all pairs -> positive definite.
    sigma2 = torch.from_numpy(np.abs(rng.randn(n)) + 2.0)
    kappa = torch.from_numpy(np.abs(rng.randn(n)) + 2.0)
    lambda_ = torch.from_numpy(np.abs(rng.randn(n)) + 1.0)

    ng = _fresh_ng()
    ng.n_channels = n
    H, posdef = ng._newton_direction(dA, sigma2, lambda_, kappa)
    assert posdef
    H = H.numpy()
    s, k, lam, d = (
        sigma2.numpy(),
        kappa.numpy(),
        lambda_.numpy(),
        dA.numpy(),
    )
    for i in range(n):
        assert abs(H[i, i] - d[i, i] / lam[i]) < 1e-12
        for j in range(n):
            if i != j:
                sk1, sk2 = s[i] * k[j], s[j] * k[i]
                expected = (sk1 * d[i, j] - d[j, i]) / (sk1 * sk2 - 1.0)
                assert abs(H[i, j] - expected) < 1e-12

    # Tiny sigma2/kappa so no pair satisfies sk1*sk2 > 1 -> not posdef.
    small = torch.from_numpy(np.full(n, 0.1))
    _, posdef2 = ng._newton_direction(dA, small, lambda_, small)
    assert not posdef2


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_newton_posdef_mstep_composition():
    """When the curvature is positive definite, the M-step applies the Newton
    direction ``H`` (not the natural gradient) and ramps toward ``newtrate``.

    On real data the curvature is not positive definite at these iterations
    (issue #21), so the posdef branch never fires in a plain fit. Here the
    finalized stats are forced positive definite to exercise that branch, and
    the resulting ``A`` is checked against an independent recomputation of the
    exact composition (slice -> H -> ``A + lrate*(A@H)`` -> column rescale).
    """
    data = _load_real_data()
    blk = 256
    big = torch.full((NW, 1), 5.0, dtype=torch.float64)
    lam = torch.full((NW, 1), 1.5, dtype=torch.float64)

    def forced(_acc):
        return big.clone(), lam.clone(), big.clone()

    ng = _fresh_ng(do_newton=True, newt_start=0, newtrate=1.0, lrate=0.1)
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    ng.iteration = 5
    block = X_t[:, :blk].contiguous()
    acc = ng._get_block_updates(block)
    A_before = ng.A.clone()

    # Independent expected A-update for the (single) model, using the same
    # forced stats and the unit-tested _newton_direction.
    eye = torch.eye(NW, dtype=torch.float64)
    dA_h = -acc["dA"][:, :, 0] / acc["dgm"][0] + eye
    H, posdef = ng._newton_direction(dA_h, big[:, 0], lam[:, 0], big[:, 0])
    assert posdef
    lrate_after = min(1.0, ng.lrate + min(1.0 / ng.newt_ramp, ng.lrate))
    idx = ng.comp_list[:, 0]
    A_expected = A_before.clone()
    A_expected[:, idx] = A_before[:, idx] + lrate_after * (A_before[:, idx] @ H)
    scale = torch.sqrt((A_expected**2).sum(dim=0))
    nz = scale > 0
    A_expected[:, nz] = A_expected[:, nz] / scale[nz]

    ng._finalize_newton_stats = forced
    ng._update_parameters(acc, blk)

    assert ng.n_newton_fallbacks == 0  # positive-definite branch taken
    assert ng.lrate == pytest.approx(lrate_after)  # ramped toward newtrate
    assert np.max(np.abs(ng.A.cpu().numpy() - A_expected.cpu().numpy())) < 1e-10
    assert np.all(np.isfinite(ng.A.cpu().numpy()))


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
@pytest.mark.parametrize("rho_val", [1.0, 2.0])
def test_newton_stats_match_at_rho_boundaries(rho_val):
    """Newton stats match the NumPy reference when rho is at the Laplace
    (1.0) / Gaussian (2.0) special cases, which real fits reach commonly.

    ``_score`` uses distinct closed forms at these boundaries (``sign(y)``,
    ``2y``), not just the generic ``rho*sign(y)*|y|^(rho-1)`` limit, so they
    need their own parity coverage.
    """
    data = _load_real_data()
    blk = 256
    ng = _fresh_ng(do_newton=True, newt_start=0)
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    ng.rho[:] = rho_val
    block = X_t[:, :blk].contiguous()
    ng_upd = ng._get_block_updates(block)
    sigma2_ng, lambda_ng, kappa_ng = ng._finalize_newton_stats(ng_upd)

    npm = _numpy_ref_like(ng, blk, do_newton=True)
    np_upd = npm._get_block_updates(block.cpu().numpy())
    dgm = np_upd["dgm"][None, :]
    for name, a_t, b_np in [
        ("sigma2", sigma2_ng, np_upd["dsigma2"] / dgm),
        ("kappa", kappa_ng, np_upd["dkappa"] / dgm),
        ("lambda", lambda_ng, np_upd["dlambda"] / dgm),
    ]:
        a = a_t.cpu().numpy()
        assert float(np.max(np.abs(a - np.asarray(b_np).reshape(a.shape)))) < 1e-8, name


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_newton_multimodel_finite_and_shaped():
    """Multi-model (n_models=2) Newton runs with correct per-model shapes and
    finite output.

    A NumPy-parity comparison is not valid here: for n_models>1 the NG backend
    intentionally includes the per-model log|det W| + sldet Jacobian in the
    model responsibility (Fortran-faithful; see the module docstring), whereas
    the legacy NumPy port omits it. So this checks per-model finalization
    (shape (n_channels, n_models), finite, no cross-model NaN) and that a short
    two-model Newton fit stays finite for both models.
    """
    data = _load_real_data()
    blk = 256
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=blk,
        do_newton=True,
        newt_start=0,
    )
    X_t = ng._preprocess(data)
    ng._initialize_parameters()
    block = X_t[:, :blk].contiguous()
    ng_upd = ng._get_block_updates(block)
    sigma2, lambda_, kappa = ng._finalize_newton_stats(ng_upd)
    for name, stat in [("sigma2", sigma2), ("lambda", lambda_), ("kappa", kappa)]:
        assert stat.shape == (NW, 2), name
        assert torch.all(torch.isfinite(stat)), name

    ng.fit(data, max_iter=6, verbose=False)
    for h in range(2):
        assert np.all(np.isfinite(ng.get_unmixing_matrix(h)))


def test_rejection_degenerate_ll_raises_clear_error():
    """If the per-sample log-likelihood is degenerate (e.g. all NaN from a
    diverged fit), rejection raises a clear ValueError rather than silently
    emptying the good set and crashing downstream on ``None`` accumulators."""
    m = _fresh_ng(do_reject=True)
    m.good_idx = torch.arange(16)
    m.numrej = 0
    ll = torch.full((16,), float("nan"), dtype=torch.float64)
    with pytest.raises(ValueError, match="removed all"):
        m._reject_outliers(ll)


def test_reject_param_validation():
    """Invalid rejection parameters are rejected at construction (guarding the
    otherwise-opaque downstream crashes: rejint=0 -> ZeroDivisionError,
    rejsig<=0 -> good set collapses to a ``None`` accumulator)."""
    with pytest.raises(ValueError, match="rejint"):
        _fresh_ng(do_reject=True, rejint=0)
    with pytest.raises(ValueError, match="rejsig"):
        _fresh_ng(do_reject=True, rejsig=0.0)
    with pytest.raises(ValueError, match="rejsig"):
        _fresh_ng(do_reject=True, rejsig=-5.0)


@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
def test_rejection_shrinks_good_sample_set():
    """With ``do_reject`` enabled, low-log-likelihood samples are dropped: the
    good-sample set shrinks, at most ``maxrej`` passes run, and the fit stays
    finite (Fortran-style ``reject_data``)."""
    data = _load_real_data()
    m = _fresh_ng(
        do_reject=True, rejsig=2.0, rejstart=2, rejint=3, maxrej=2, block_size=512
    )
    n_total = data.shape[1]
    m.fit(data, max_iter=12, verbose=False)

    assert m.numrej <= 2
    assert m.numrej >= 1
    n_good = int(m.good_idx.numel())
    assert n_good < n_total  # some samples rejected
    assert np.all(np.isfinite(m.get_unmixing_matrix(0)))
    assert np.all(np.isfinite(m.ll_history))

    # The reported LL is normalized by the good-sample count, not the original
    # total. Recompute the final iteration's LL from the surviving good set and
    # confirm it divides by n_good (dividing by n_total would be off by
    # n_good/n_total, well outside tolerance here since ~hundreds are dropped).
    X_t = m._preprocess(data)
    acc = m._accumulate_blocks(X_t[:, m.good_idx])
    ll_per_good = float(acc["ll"] / (n_good * m.n_channels))
    ll_per_total = float(acc["ll"] / (n_total * m.n_channels))
    assert abs(m.ll_history[-1] - ll_per_good) < abs(m.ll_history[-1] - ll_per_total)


@pytest.mark.slow
@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
@pytest.mark.xfail(
    strict=False,
    reason="Component correlation vs Fortran plateaus at ~0.68 regardless of "
    "Newton. Newton/PDF/rejection are ported and unit-parity-verified (Phase 4, "
    "issue #13), but they only accelerate convergence to the NG backend's own "
    "fixed point (LL ~ -3.47), which differs from Fortran's (LL -3.41). Closing "
    "the >0.95 gate needs a separate investigation of the NG-vs-Fortran "
    "fixed-point gap (initialization/E-step), tracked as issue #21.",
)
def test_end_to_end_correlation_vs_fortran():
    """Epic definition-of-done: Hungarian-matched component correlation vs the
    Fortran binary > 0.95 on the sample data.

    Currently xfails: see the marker reason. Newton is enabled here (matching
    the Fortran sample settings); on this data the curvature is not positive
    definite, so the Newton step falls back to natural gradient every iteration
    (``m.n_newton_fallbacks``), which is part of why the correlation plateaus at
    ~0.68. The positive-definite Newton composition is covered by
    ``test_newton_posdef_mstep_composition``.
    """
    import sys

    root = Path(__file__).resolve().parents[2].parent
    sys.path.insert(0, str(root))
    from validate_implementations import (
        load_sample_data,
        run_fortran_amica,
        compare_results,
    )

    data, params = load_sample_data()
    params = dict(params)
    params["max_iter"] = 100
    out = root / "pyAMICA" / "tests" / "torch_tests" / "_ng_e2e_tmp"
    out.mkdir(parents=True, exist_ok=True)
    fortran = run_fortran_amica(data, params, out, SEED)

    m = _fresh_ng(do_newton=True, newt_start=50, newtrate=1.0, lrate=0.05)
    m.fit(data.astype(np.float64), max_iter=100, verbose=False)
    ng_results = {
        "final_ll": m.ll_history[-1],
        "final_iter": len(m.ll_history),
        "W": m.get_unmixing_matrix(0),
        "A": m.get_mixing_matrix(0),
    }
    cmp = compare_results(fortran, ng_results)
    assert cmp["mean_correlation"] > 0.95
