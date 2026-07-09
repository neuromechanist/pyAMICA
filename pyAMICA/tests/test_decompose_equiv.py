"""Unit tests for the Phase 3 IC-equivalence matcher (issue #87).

The matcher must recover a correlation of 1.0 between two unmixing matrices that
are the SAME decomposition up to ICA's inherent ambiguities -- component
permutation and per-component sign flip -- and report a low correlation for
genuinely different components. No data/GPU/binary needed.
"""

import importlib.util
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]


def _load():
    path = _REPO / "benchmarks" / "benchmark_decompose.py"
    spec = importlib.util.spec_from_file_location("benchmark_decompose", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


dec = _load()


def _rng():
    # deterministic without Math.random-style hazards: fixed-seed generator
    return np.random.default_rng(0)


def test_match_identical_is_one():
    W = _rng().standard_normal((32, 32))
    corr = dec._match_correlation(W, W.copy())
    assert np.allclose(corr, 1.0, atol=1e-9)


def test_match_invariant_to_permutation_and_sign():
    """A permuted + sign-flipped copy is the same decomposition -> corr ~ 1."""
    rng = _rng()
    W = rng.standard_normal((32, 32))
    perm = rng.permutation(32)
    signs = rng.choice([-1.0, 1.0], size=32)[:, None]
    W2 = (W * signs)[perm]  # flip each component's sign, then permute components
    corr = dec._match_correlation(W, W2)
    assert np.allclose(corr, 1.0, atol=1e-9)


def test_match_low_for_unrelated():
    rng = _rng()
    W1 = rng.standard_normal((32, 32))
    W2 = rng.standard_normal((32, 32))
    # unrelated 32-dim components: best-assignment |corr| stays well below 1
    assert dec._match_correlation(W1, W2).mean() < 0.6
