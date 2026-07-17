"""Unit tests for the Amari distance metric (issue #116).

Amari distance is permutation- and scale-invariant by construction, unlike the
Hungarian-matched correlation elsewhere in this suite: it needs no assignment
step, and two unmixing matrices that are the SAME decomposition up to ICA's
inherent ambiguities (row permutation, per-row scaling) score 0. No
data/GPU/binary needed -- mirrors test_decompose_equiv.py's convention of
testing the metric's math properties on synthetic matrices.
"""

import importlib.util
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]


def _load():
    path = _REPO / "validate_implementations.py"
    spec = importlib.util.spec_from_file_location("validate_implementations", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vi = _load()


def _rng():
    return np.random.default_rng(0)


def test_amari_distance_self_is_zero():
    W = _rng().standard_normal((32, 32))
    assert np.allclose(vi.amari_distance(W, W.copy()), 0.0, atol=1e-9)


def test_amari_distance_invariant_to_permutation_and_scale():
    rng = _rng()
    W = rng.standard_normal((32, 32))
    perm = rng.permutation(32)
    scale = rng.uniform(0.5, 2.0, size=32)[:, None]
    W2 = (W * scale)[perm]
    assert np.allclose(vi.amari_distance(W, W2), 0.0, atol=1e-9)


def test_amari_distance_nonzero_for_unrelated():
    rng = _rng()
    W1 = rng.standard_normal((32, 32))
    W2 = rng.standard_normal((32, 32))
    assert vi.amari_distance(W1, W2) > 0.2


def test_amari_distance_nonnegative():
    rng = _rng()
    for _ in range(5):
        W1 = rng.standard_normal((32, 32))
        W2 = rng.standard_normal((32, 32))
        assert vi.amari_distance(W1, W2) >= 0.0


def test_amari_distance_symmetric():
    rng = _rng()
    W1 = rng.standard_normal((32, 32))
    W2 = rng.standard_normal((32, 32))
    assert np.allclose(vi.amari_distance(W1, W2), vi.amari_distance(W2, W1), atol=1e-12)


def test_amari_distance_rejects_degenerate_input():
    W = _rng().standard_normal((32, 32))
    W_zero_row = W.copy()
    W_zero_row[0] = 0.0
    try:
        vi.amari_distance(W_zero_row, W)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for an all-zero row")


def test_amari_distance_rejects_scalar_input():
    W = np.array([[1.0]])
    try:
        vi.amari_distance(W, W)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a 1x1 matrix")
