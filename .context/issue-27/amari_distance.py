"""Amari-distance detail for the multi-model ensemble (issue #116, follow-up to #27).

Reuses the already-saved `ensemble.npz` (20 Fortran + 20 NG stacked unmixing
matrices from real sample-EEG fits, `multimodel_ensemble.py`) to compute a
second, complementary parity metric: the Amari distance (Amari, Cichocki &
Yang 1996), which is permutation- and scale-invariant by construction (no
Hungarian assignment step needed, unlike the existing cross-correlation
metric). No re-fitting -- the raw matrices already exist on disk.

Each stacked (64, 32) matrix is two 32x32 per-model unmixing matrices; since
which Fortran model corresponds to which NG model is not identified, both
label pairings are tried and the lower-distance (better) one is kept, the
same non-identifiability treatment the existing Hungarian matching applies at
the component level.

    uv run python .context/issue-27/amari_distance.py
"""

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NW = 32


def _load_validate_implementations():
    path = REPO / "validate_implementations.py"
    spec = importlib.util.spec_from_file_location("validate_implementations", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


amari_distance = _load_validate_implementations().amari_distance


def xcorr(Wa, Wb):
    na = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-12)
    nb = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-12)
    corr = np.abs(na @ nb.T)
    r, c = linear_sum_assignment(1 - corr)
    return float(corr[r, c].mean())


def model_amari(Wa64, Wb64):
    """Best-pairing mean Amari distance between two stacked 2-model matrices."""
    Wa = [Wa64[:NW], Wa64[NW:]]
    Wb = [Wb64[:NW], Wb64[NW:]]
    best = None
    for pairing in ((0, 1), (1, 0)):
        ds = [amari_distance(Wa[i], Wb[j]) for i, j in enumerate(pairing)]
        mean_d = float(np.mean(ds))
        if best is None or mean_d < best:
            best = mean_d
    return best


def pairwise(A, B, same, metric):
    return np.array(
        [
            metric(A[i], B[j])
            for i in range(len(A))
            for j in range(len(B))
            if not (same and j <= i)
        ]
    )


def perm_test_not_worse(Fs, Gs, metric, higher_is_worse, n_perm=20000, seed=0):
    """Run-level permutation test, generalized from multimodel_ensemble.py to
    a metric where either higher or lower indicates worse agreement."""
    rng = np.random.default_rng(seed)
    allW = np.concatenate([Fs, Gs], axis=0)
    m = len(allW)
    n = len(Fs)
    P = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            P[i, j] = P[j, i] = metric(allW[i], allW[j])

    sign = 1.0 if higher_is_worse else -1.0

    def gap(mask):  # sign-adjusted within-group-A minus A-vs-rest
        a = np.flatnonzero(mask)
        b = np.flatnonzero(~mask)
        within = P[np.ix_(a, a)][np.triu_indices(a.size, 1)].mean()
        betw = P[np.ix_(a, b)].mean()
        return sign * (betw - within)

    true_mask = np.zeros(m, dtype=bool)
    true_mask[:n] = True
    obs_gap = gap(true_mask)
    ge = 1
    for _ in range(n_perm):
        mask = np.zeros(m, dtype=bool)
        mask[rng.permutation(m)[:n]] = True
        if gap(mask) >= obs_gap:
            ge += 1
    return ge / (n_perm + 1)


def per_run_detail(Fs, Gs, metric):
    """Each run's mean value to its own group's other runs (within) and to
    the opposite group (between), for one metric."""
    n = len(Fs)
    rows = []
    for i in range(n):
        within = np.mean([metric(Fs[i], Fs[j]) for j in range(n) if j != i])
        between = np.mean([metric(Fs[i], Gs[j]) for j in range(n)])
        rows.append(
            {
                "implementation": "Fortran",
                "run": i,
                "within": within,
                "between": between,
            }
        )
    for i in range(n):
        within = np.mean([metric(Gs[i], Gs[j]) for j in range(n) if j != i])
        between = np.mean([metric(Gs[i], Fs[j]) for j in range(n)])
        rows.append(
            {
                "implementation": "pamica",
                "run": i,
                "within": within,
                "between": between,
            }
        )
    return rows


def main():
    d = np.load(HERE / "ensemble.npz")
    Fs, Gs = d["Fs"], d["Gs"]
    n = len(Fs)

    metrics = {"corr": xcorr, "amari": model_amari}
    summary = {}
    for name, metric in metrics.items():
        within_F = pairwise(Fs, Fs, True, metric)
        within_G = pairwise(Gs, Gs, True, metric)
        between = pairwise(Gs, Fs, False, metric)
        higher_is_worse = name == "amari"  # lower correlation / higher distance = worse
        p = perm_test_not_worse(Fs, Gs, metric, higher_is_worse)
        summary[name] = {
            "within_Fortran": {"mean": within_F.mean(), "sd": within_F.std()},
            "within_pamica": {"mean": within_G.mean(), "sd": within_G.std()},
            "between": {"mean": between.mean(), "sd": between.std()},
            "perm_p_not_worse": p,
        }
        print(f"\n==== {name} (N={n} each) ====")
        for grp, a in [
            ("within-Fortran", within_F),
            ("within-pamica", within_G),
            ("between", between),
        ]:
            print(f"{grp:16s} mean={a.mean():.4f} sd={a.std():.4f}")
        print(
            f"run-level permutation (between not worse than within-Fortran): p={p:.3f}"
        )

    with open(HERE / "amari_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    corr_detail = {
        (r["implementation"], r["run"]): r for r in per_run_detail(Fs, Gs, xcorr)
    }
    amari_detail = per_run_detail(Fs, Gs, model_amari)
    with open(HERE / "per_run_detail.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "implementation",
                "run",
                "corr_within",
                "corr_between",
                "amari_within",
                "amari_between",
            ]
        )
        for row in amari_detail:
            key = (row["implementation"], row["run"])
            c = corr_detail[key]
            w.writerow(
                [
                    row["implementation"],
                    row["run"],
                    f"{c['within']:.4f}",
                    f"{c['between']:.4f}",
                    f"{row['within']:.4f}",
                    f"{row['between']:.4f}",
                ]
            )
    print(f"\nwrote {HERE / 'amari_summary.json'} and {HERE / 'per_run_detail.csv'}")


if __name__ == "__main__":
    main()
