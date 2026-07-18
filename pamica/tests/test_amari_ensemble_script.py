"""Unit tests for the ensemble-analysis logic in
`.context/issue-27/amari_distance.py` (issue #116).

Covers the two pieces of non-trivial logic introduced there that
`test_amari_distance.py` does not exercise: `model_amari`'s try-both-pairings
model-label resolution, and `perm_test_not_worse`'s `higher_is_worse`
sign-flip generalization. Synthetic data only, no `ensemble.npz` needed --
mirrors `test_decompose_equiv.py`'s convention.
"""

import importlib.util
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]


def _load():
    path = _REPO / ".context" / "issue-27" / "amari_distance.py"
    spec = importlib.util.spec_from_file_location("amari_distance_script", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


script = _load()


def _rng():
    return np.random.default_rng(0)


def test_model_amari_finds_swapped_pairing():
    """If model 0 of A matches model 1 of B (and vice versa), model_amari must
    find the swapped pairing rather than staying stuck on the naive (0, 1)
    one."""
    rng = _rng()
    m0 = rng.standard_normal((32, 32))
    m1 = rng.standard_normal((32, 32))
    Wa64 = np.vstack([m0, m1])
    Wb64 = np.vstack([m1, m0])  # models swapped relative to Wa64
    assert np.allclose(script.model_amari(Wa64, Wb64), 0.0, atol=1e-9)


def test_model_amari_matches_best_of_naive_pairings():
    rng = _rng()
    Wa64 = rng.standard_normal((64, 32))
    Wb64 = rng.standard_normal((64, 32))
    naive = np.mean(
        [
            script.amari_distance(Wa64[:32], Wb64[:32]),
            script.amari_distance(Wa64[32:], Wb64[32:]),
        ]
    )
    swapped = np.mean(
        [
            script.amari_distance(Wa64[:32], Wb64[32:]),
            script.amari_distance(Wa64[32:], Wb64[:32]),
        ]
    )
    assert np.isclose(script.model_amari(Wa64, Wb64), min(naive, swapped))


def test_perm_test_distance_metric_detects_worse_between():
    """higher_is_worse=True: construct groups where cross-group distance is
    clearly larger than within-group distance, and check the p-value reports
    strong evidence that between IS worse (opposite of the "not worse" null
    used for the real ensembles)."""
    rng = np.random.default_rng(1)
    n, dim = 6, 8
    base_a = rng.standard_normal((dim, dim))
    base_b = rng.standard_normal((dim, dim))
    Fs = np.array([base_a + 0.01 * rng.standard_normal((dim, dim)) for _ in range(n)])
    Gs = np.array([base_b + 0.01 * rng.standard_normal((dim, dim)) for _ in range(n)])

    p = script.perm_test_not_worse(
        Fs, Gs, script.amari_distance, higher_is_worse=True, n_perm=2000, seed=0
    )
    assert p < 0.05


def test_perm_test_similarity_metric_detects_worse_between():
    """higher_is_worse=False (the correlation convention): same construction,
    but with a similarity metric where a LOWER between-group value indicates
    worse agreement."""
    rng = np.random.default_rng(1)
    n, dim = 6, 8
    base_a = rng.standard_normal((dim, dim))
    base_b = rng.standard_normal((dim, dim))
    Fs = np.array([base_a + 0.01 * rng.standard_normal((dim, dim)) for _ in range(n)])
    Gs = np.array([base_b + 0.01 * rng.standard_normal((dim, dim)) for _ in range(n)])

    def similarity(a, b):
        return -script.amari_distance(a, b)

    p = script.perm_test_not_worse(
        Fs, Gs, similarity, higher_is_worse=False, n_perm=2000, seed=0
    )
    assert p < 0.05


def test_perm_test_no_evidence_when_groups_are_equivalent():
    """Same distribution for both groups: neither direction should find
    significant evidence that between is worse."""
    rng = np.random.default_rng(2)
    n, dim = 8, 8
    Fs = np.array([rng.standard_normal((dim, dim)) for _ in range(n)])
    Gs = np.array([rng.standard_normal((dim, dim)) for _ in range(n)])

    p = script.perm_test_not_worse(
        Fs, Gs, script.amari_distance, higher_is_worse=True, n_perm=2000, seed=0
    )
    assert p > 0.05
