"""MLX backend (AMICAMLXNG) tests -- issue #76, epic #74 Phase C.

Apple-Silicon only. Real sample EEG (no synthetic/mock). The whole module
self-skips when MLX or an Apple GPU is unavailable, so CI on ubuntu skips it.

Correctness is checked two ways: the per-block sufficient statistics match the
validated NumPy float64 reference (tight, independent oracle), and a full fit's
converged log-likelihood matches the PyTorch float32 backend (the epic's
backend-vs-backend acceptance).
"""

from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42

pytestmark = [
    pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing"),
    pytest.mark.skipif(
        mx.default_device().type != mx.DeviceType.gpu, reason="no Apple GPU"
    ),
]

# Sufficient-stat keys the MLX backend produces (single-model, non-Newton).
_STAT_KEYS = [
    "dgm",
    "dalpha_n",
    "dmu_n",
    "dmu_d",
    "dbeta_n",
    "dbeta_d",
    "drho_n",
    "dWtmp",
]


def _load_real_data() -> np.ndarray:
    from pyAMICA.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def test_mvp_rejects_unsupported_config():
    """The v1 MVP boundaries fail loudly, not silently."""
    from pyAMICA.mlx_impl import AMICAMLXNG

    for kw in ({"n_models": 2}, {"do_newton": True}, {"pdftype": 2, "n_mix": NMIX}):
        with pytest.raises(NotImplementedError):
            AMICAMLXNG(n_channels=NW, n_mix=kw.pop("n_mix", 1), **kw)


def test_sufficient_stats_match_numpy_reference():
    """Tight algorithmic faithfulness: the MLX (float32) per-block sufficient
    statistics match the validated NumPy float64 reference on an identical block
    with identical (shared-seed) parameters. rtol=1e-3 accommodates float32
    vs float64 (measured max relerr ~1.6e-4 on dWtmp/dmu_d)."""
    from pyAMICA.mlx_impl import AMICAMLXNG
    from pyAMICA.numpy_impl.core import AMICA as AMICA_NumPy

    data = _load_real_data()
    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED, block_size=256)
    x_t = m._preprocess(data)
    m._initialize_parameters()

    blk = 256
    block = np.array(x_t[:, :blk])  # float32 sphered block, on host
    mlx_upd = m._get_block_updates(mx.array(block))

    npm = AMICA_NumPy(num_models=1, num_mix=NMIX, do_newton=False)
    npm.data_dim = NW
    npm.num_comps = NW
    npm.num_models = 1
    npm.num_mix = NMIX
    npm.block_size = blk
    npm.comp_list = np.arange(NW).reshape(NW, 1)
    npm.A = np.array(m.A, dtype=np.float64)
    npm.W = np.array(m.W, dtype=np.float64)[:, :, None]  # NumPy expects (n,n,models)
    npm.c = np.zeros((NW, 1))
    npm.mu = np.array(m.mu, dtype=np.float64)
    npm.alpha = np.array(m.alpha, dtype=np.float64)
    npm.beta = np.array(m.beta, dtype=np.float64)
    npm.rho = np.array(m.rho, dtype=np.float64)
    npm.gm = np.array(m.gm, dtype=np.float64)
    np_upd = npm._get_block_updates(block.astype(np.float64))

    for key in _STAT_KEYS:
        a = np.asarray(np.array(mlx_upd[key]), dtype=np.float64)
        b = np.asarray(np_upd[key], dtype=np.float64).reshape(a.shape)
        assert np.allclose(a, b, rtol=1e-3, atol=1e-4), (
            f"{key} differs from NumPy reference by {np.max(np.abs(a - b)):.3e}"
        )


def test_backend_stable_on_full_data():
    """The MLX backend fits the full 30504-sample EEG on the Apple GPU (float32)
    without diverging (the Phase A ``ufp/y`` guard is carried into MLX), and the
    log-likelihood improves over the run."""
    from pyAMICA.mlx_impl import AMICAMLXNG

    m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED)
    m.fit(_load_real_data(), max_iter=100, verbose=False)

    hist = np.asarray(m.ll_history, dtype=float)
    assert np.all(np.isfinite(hist))
    assert np.isfinite(m.final_ll_)
    assert np.all(np.isfinite(np.array(m.A)))
    assert m.stop_reason not in AMICAMLXNG._DEGENERATE_STOP_REASONS
    assert hist[-1] > hist[0]  # ascent


def test_converged_ll_matches_torch_float32():
    """Epic acceptance: the MLX backend is equivalent to the PyTorch float32
    backend. At matched settings/seed both reach the same natural-gradient GG
    fixed point; the converged per-sample-per-channel LL agrees within 1e-2
    (measured gap ~2e-6). Bit-identity is impossible (GPU-vs-CPU float32 +
    reduction order), so the tolerance is relational, never a hardcoded LL."""
    import torch

    from pyAMICA.mlx_impl import AMICAMLXNG
    from pyAMICA.torch_impl import AMICATorchNG

    data = _load_real_data()
    mlx_m = AMICAMLXNG(n_channels=NW, n_mix=NMIX, seed=SEED)
    mlx_m.fit(data, max_iter=100, verbose=False)

    torch_m = AMICATorchNG(
        n_channels=NW,
        n_models=1,
        n_mix=NMIX,
        seed=SEED,
        device="cpu",
        dtype=torch.float32,
        do_newton=False,
    )
    torch_m.fit(data, max_iter=100, verbose=False)

    assert np.isfinite(mlx_m.final_ll_)
    assert abs(mlx_m.final_ll_ - torch_m.final_ll_) < 1e-2
