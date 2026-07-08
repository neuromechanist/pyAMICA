"""float32 full-data stability (issue #75, epic #74 Phase A).

Before #75, ``AMICATorchNG`` in float32 diverged to NaN on the full 30504-sample
sample EEG across every seed (Newton on and off, crashing iter ~9-105), while
float64 converged. Root cause: at a sample sitting on a mixture mean, float32
rounds the scaled activation ``y`` to *exactly* 0, and the score ``fp(0)=0`` for
every family, so the mu-denominator term ``ufp/y`` is ``0/0 = NaN`` (float64
never hits exact 0). Diagnostics (`.context/issue-63/`, `.context/mps_pathways.md`)
ruled out summation precision: accumulating the block sufficient statistics in
float64 did *not* help, but guarding that single division does. The guard is a
no-op in float64 (bit-identical, so single-model #24 parity is preserved) and
needs no float64, so it also stabilizes the MPS/float32 path.

Real sample EEG only (no synthetic/mock). The full-data fit is the only regime
that reproduces the divergence, so it is the necessary regression surface.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pyAMICA.torch_impl import AMICATorchNG
from pyAMICA.torch_impl.core import _score
from pyAMICA.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
# Past the historical divergence window (naive float32 crashed by iter ~105); a
# reintroduced 0/0 would surface as a nan_ll stop well within this budget.
MAX_ITER = 150
_DEGENERATE = ("nan_ll", "singular_ll")


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _fit(data, dtype, seed, do_newton, device="cpu", max_iter=MAX_ITER, n_samples=None):
    m = AMICATorchNG(
        n_channels=NW,
        n_models=1,
        n_mix=3,
        device=device,
        dtype=dtype,
        do_newton=do_newton,
        seed=seed,
    )
    x = data if n_samples is None else data[:, :n_samples]
    m.fit(x, max_iter=max_iter, verbose=False)
    return m


# --- the invariant the guard relies on --------------------------------------


def test_score_is_zero_at_zero_activation():
    """``fp(0)=0`` for every family, so at ``y==0`` the numerator ``ufp=u*fp``
    is 0 too and the guarded ``ufp/1`` contributes 0 (not ``0/0=NaN``)."""
    y = torch.zeros(4, dtype=torch.float32)
    for rho in (1.0, 1.5, 2.0):  # Laplace, GG, Gaussian (pdtype None path)
        assert torch.all(_score(y, torch.full_like(y, rho)) == 0)
    for pdtype in (1, 2, 3, 4):  # the fixed families
        fp = _score(y, torch.full_like(y, 1.5), torch.full((4,), pdtype))
        assert torch.all(fp == 0)


# --- the regression: full-data float32 no longer diverges -------------------

# Five distinct seeds spanning both Newton settings. The 0/0 is in the always-on
# E-step accumulation (Newton-independent), so the full 5x2 cross-product is
# unnecessary CI cost; this still covers >=5 seeds with Newton on and off.
_STABILITY_CASES = [(1, True), (2, False), (3, True), (4, False), (5, True)]


@pytest.mark.parametrize("seed,do_newton", _STABILITY_CASES)
def test_float32_stable_on_full_data(real_data, seed, do_newton):
    m = _fit(real_data, torch.float32, seed, do_newton)
    assert np.all(np.isfinite(np.asarray(m.ll_history, dtype=float)))
    assert np.isfinite(m.final_ll_)
    assert torch.isfinite(m.A).all()
    # The exact failure mode was a nan_ll stop; a singular W would also be wrong.
    assert m.stop_reason not in _DEGENERATE


# --- quality: float32 tracks the float64 optimum ----------------------------


@pytest.mark.parametrize("do_newton", [False, True])
def test_float32_ll_matches_float64(real_data, do_newton):
    seed = 0
    f32 = _fit(real_data, torch.float32, seed, do_newton)
    f64 = _fit(real_data, torch.float64, seed, do_newton)
    # Relational to the in-test float64 fit, never a hardcoded LL. One-sided
    # "not materially worse" is load-bearing; the trajectories diverge only
    # chaotically, so the two-sided band is a loose sanity guard.
    assert f32.final_ll_ >= f64.final_ll_ - 0.05
    assert abs(f32.final_ll_ - f64.final_ll_) < 0.1


# --- MPS smoke: the actual Apple-GPU target (self-skips in CI) ---------------


@pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS device not available"
)
def test_float32_mps_smoke(real_data):
    """float32 on MPS (no float64 on Apple GPUs) stays finite. Short slice/iters:
    a smoke check that the guard also holds on-device, not a convergence gate."""
    m = _fit(
        real_data,
        torch.float32,
        seed=1,
        do_newton=True,
        device="mps",
        max_iter=30,
        n_samples=8192,
    )
    assert np.isfinite(m.final_ll_)
    assert torch.isfinite(m.A).all()
    assert m.stop_reason not in _DEGENERATE
