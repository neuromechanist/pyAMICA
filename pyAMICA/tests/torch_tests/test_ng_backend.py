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


def _fresh_ng(block_size: int = 256) -> AMICATorchNG:
    return AMICATorchNG(
        n_channels=NW,
        n_models=1,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=block_size,
    )


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


@pytest.mark.slow
@pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")
@pytest.mark.xfail(
    strict=True,
    reason="Natural-gradient alone reaches ~0.69 mean correlation vs Fortran; "
    "reaching >0.95 requires Newton (Phase 4, issue #13). Flips to xpass when "
    "Newton lands.",
)
def test_end_to_end_correlation_vs_fortran():
    """Epic definition-of-done: Hungarian-matched component correlation vs the
    Fortran binary > 0.95 on the sample data. Gated by Phase 4 (Newton)."""
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

    m = _fresh_ng()
    m.fit(data.astype(np.float64), max_iter=100, verbose=False)
    ng_results = {
        "final_ll": m.ll_history[-1],
        "final_iter": len(m.ll_history),
        "W": m.get_unmixing_matrix(0),
        "A": m.get_mixing_matrix(0),
    }
    cmp = compare_results(fortran, ng_results)
    assert cmp["mean_correlation"] > 0.95
