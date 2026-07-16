"""Multi-model NG-vs-Fortran distributional-equivalence harness (issue #27).

Reproduces `multimodel_distributional_equivalence.md`: runs N Fortran + N NG
fits on the real sample EEG (n_models=2), then compares the within-Fortran,
within-NG, and between (NG-Fortran) distributions of the stacked 2*NW Hungarian
cross-correlation, plus the log-likelihood distributions. Writes the figure
next to this script. Real sample data + Fortran binary only (NO MOCK).

    uv run python .context/issue-27/multimodel_ensemble.py [N]
    uv run python .context/issue-27/multimodel_ensemble.py --from-cache  # reuse ensemble.npz, no re-fitting

The Fortran binary (`sample_data/amica15mac`) is x86_64 and runs under Rosetta
on Apple silicon. Absolute cross-corr magnitudes depend on config/seed; the
within-vs-between comparison is the config-independent result.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.optimize import linear_sum_assignment  # noqa: E402

from pyAMICA.torch_impl import AMICATorchNG  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIN = REPO / "pyAMICA/sample_data/amica15mac"
FDT = REPO / "pyAMICA/sample_data/eeglab_data.fdt"
FIXTURE = REPO / "pyAMICA/tests/torch_tests/_ng_e2e_tmp/fortran_run/input.param"
NW, FIELD, MAX_ITER, DELTA = 32, 30504, 100, 0.05
C_FORT, C_NG, C_BET = "#0072B2", "#E69F00", "#009E73"  # Okabe-Ito


def load_data():
    from pyAMICA.torch_impl.utils import load_eeglab_data

    return load_eeglab_data(str(FDT), data_dim=NW, field_dim=FIELD).astype(np.float64)


def run_fortran(work, tag):
    d = work / f"fort_{tag}"
    (d / "fortran_output").mkdir(parents=True, exist_ok=True)
    shutil.copy(FDT, d / "eeglab_data.fdt")
    lines = []
    for ln in FIXTURE.read_text().splitlines():
        if ln.startswith("num_models"):
            lines.append("num_models 2")
        elif ln.startswith("max_iter"):
            lines.append(f"max_iter {MAX_ITER}")
        else:
            lines.append(ln)
    (d / "input.param").write_text("\n".join(lines) + "\n")
    orig = os.getcwd()
    os.chdir(d)
    try:
        r = subprocess.run(
            [str(BIN), "input.param"], capture_output=True, text=True, timeout=600
        )
    finally:
        os.chdir(orig)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])
    W = np.fromfile(d / "fortran_output/W", dtype=np.float64).reshape(
        NW, NW, 2, order="F"
    )
    ll = next(
        (
            float(ln.split("LL =")[1].split()[0])
            for ln in reversed(r.stdout.splitlines())
            if "LL =" in ln
        ),
        np.nan,
    )
    return np.vstack([W[:, :, 0], W[:, :, 1]]), ll


def run_ng(data, seed):
    ng = AMICATorchNG(
        n_channels=NW, n_models=2, n_mix=3, block_size=512, lrate=0.05, minlrate=1e-8,
        lratefact=0.5, maxdecs=3, do_newton=True, newt_start=50, newt_ramp=10,
        newtrate=1.0, rho0=1.5, minrho=1.0, maxrho=2.0, rholrate=0.05,
        rholratefact=0.5, invsigmin=1e-8, invsigmax=100.0, doscaling=True,
        scalestep=1, seed=seed, device="cpu",
    )  # fmt: skip
    ng.fit(data, max_iter=MAX_ITER, verbose=False)
    return (
        np.vstack([ng.get_unmixing_matrix(0), ng.get_unmixing_matrix(1)]),
        ng.final_ll_,  # LL of the returned iterate (issue #51 best-iterate safeguard)
    )


def xcorr(Wa, Wb):
    na = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-12)
    nb = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-12)
    corr = np.abs(na @ nb.T)
    r, c = linear_sum_assignment(1 - corr)
    return float(corr[r, c].mean())


def pairwise(A, B, same):
    return np.array(
        [
            xcorr(A[i], B[j])
            for i in range(len(A))
            for j in range(len(B))
            if not (same and j <= i)
        ]
    )


def perm_test_not_worse(Fs, Gs, n_perm=20000, seed=0):
    """Run-level permutation test for 'between-implementation agreement is not
    worse than Fortran's own run-to-run agreement'.

    The pairwise cross-correlations are NOT independent (each of the 2N runs
    appears in ~2N-1 pairs), so a Mann-Whitney/TOST on the pairwise values is
    pseudoreplicated and its p-value is invalid. This test instead permutes the
    2N runs as intact units: the statistic is mean(within-group-A pairs) -
    mean(A-vs-B pairs), and the null relabels which N runs are "group A". That
    respects the shared-run dependence, so the p-value is valid.

    Returns (observed diff = mean(between) - mean(within-Fortran), one-sided p
    for 'between worse than within-Fortran').
    """
    rng = np.random.default_rng(seed)
    allW = np.concatenate([Fs, Gs], axis=0)
    m = len(allW)
    n = len(Fs)
    P = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            P[i, j] = P[j, i] = xcorr(allW[i], allW[j])

    def gap(mask):  # within-group-A minus A-vs-rest (large => between worse)
        a = np.flatnonzero(mask)
        b = np.flatnonzero(~mask)
        within = P[np.ix_(a, a)][np.triu_indices(a.size, 1)].mean()
        betw = P[np.ix_(a, b)].mean()
        return within - betw

    true_mask = np.zeros(m, dtype=bool)
    true_mask[:n] = True  # Fortran is group A
    obs_gap = gap(true_mask)
    ge = 1  # +1: include the observed permutation
    for _ in range(n_perm):
        mask = np.zeros(m, dtype=bool)
        mask[rng.permutation(m)[:n]] = True
        if gap(mask) >= obs_gap:
            ge += 1
    p = ge / (n_perm + 1)
    between_minus_withinF = -obs_gap
    return between_minus_withinF, p


def figure(within_F, within_G, between, F_ll, G_ll, diff, p_perm, ks, out):
    # Sized to its ACTUAL print footprint, not a big on-screen canvas: paper.md
    # embeds this at width=100% of a ~5.36in single-column page (measured from the
    # compiled paper.pdf), so figsize is set to that width directly -- LaTeX then
    # displays it near 1:1 instead of shrinking a much larger canvas down to fit,
    # which previously collapsed every font to an unreadable ~3pt in the printed
    # PDF even though it looked fine on screen (figure-qa print-scale finding).
    plt.rcParams.update(
        {"font.size": 7, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(5.36, 4.2))
    bins = np.linspace(0.5, 1.0, 26)
    for arr, col, lab in [
        (within_F, C_FORT, "within-Fortran"),
        (within_G, C_NG, "within-pyAMICA"),
        (between, C_BET, "between (pyAMICA-Fortran)"),
    ]:
        axA.hist(arr, bins=bins, density=True, color=col, alpha=0.35)
        axA.hist(
            arr, bins=bins, density=True, histtype="step", color=col, lw=2, label=lab
        )
        axA.axvline(arr.mean(), color=col, ls="--", lw=1.2)
    axA.set_xlabel("Hungarian-matched cross-correlation\n(stacked 2x32 components)")
    axA.set_ylabel("density")
    axA.set_title(
        "A  Partition-agreement distributions",
        loc="left",
        fontweight="bold",
        fontsize=8,
    )
    bins_ll = np.linspace(
        min(F_ll.min(), G_ll.min()) - 0.005, max(F_ll.max(), G_ll.max()) + 0.005, 24
    )
    for arr, col, lab in [(F_ll, C_FORT, "Fortran"), (G_ll, C_NG, "pyAMICA")]:
        axB.hist(arr, bins=bins_ll, density=True, color=col, alpha=0.35)
        axB.hist(
            arr, bins=bins_ll, density=True, histtype="step", color=col, lw=2, label=lab
        )
        axB.axvline(arr.mean(), color=col, ls="--", lw=1.2)
    axB.set_xlabel("final log-likelihood\n(mean per sample-channel)")
    axB.set_ylabel("density")
    axB.set_title(
        "B  Likelihood distributions", loc="left", fontweight="bold", fontsize=8
    )

    # Both the legend and the stats box previously sat inside the axes and ended up
    # overlapping the histogram bars (and each other) no matter where they were
    # anchored -- with 3 distributions filling most of the plotted range, there was
    # no empty pocket big enough to hold either. Reserve a fixed bottom margin for
    # both instead and place them with figure-fraction (not axes-fraction)
    # coordinates: axes-fraction anchors turned out to depend on the final axes
    # height that tight_layout picks, which isn't known in advance and caused the
    # legend/text to collide with the xlabel above it. get_position() gives each
    # axes' true horizontal center after the layout below is fixed, so this keeps
    # each panel's legend/text under its own histogram, not bleeding into the
    # other panel.
    fig.subplots_adjust(top=0.80, bottom=0.42, left=0.11, right=0.97, wspace=0.45)
    cx_a = sum(axA.get_position().intervalx) / 2
    cx_b = sum(axB.get_position().intervalx) / 2

    handles_a, labels_a = axA.get_legend_handles_labels()
    fig.legend(
        handles_a, labels_a, frameon=False, fontsize=6,
        loc="upper center", bbox_to_anchor=(cx_a, 0.31),
    )  # fmt: skip
    fig.text(
        cx_a,
        0.20,
        f"mean corr. (run pairs)\n"
        f"within-Fortran: {within_F.mean():.3f} (n={len(within_F)})\n"
        f"within-pyAMICA: {within_G.mean():.3f} (n={len(within_G)})\n"
        f"between: {between.mean():.3f} (n={len(between)})\n"
        f"diff: {diff:+.3f} (margin +/-0.05)\n"
        f"perm. p={p_perm:.2f}",
        ha="center",
        va="top",
        fontsize=5.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.95),
    )

    handles_b, labels_b = axB.get_legend_handles_labels()
    fig.legend(
        handles_b, labels_b, frameon=False, fontsize=6,
        loc="upper center", bbox_to_anchor=(cx_b, 0.31),
    )  # fmt: skip
    fig.text(
        cx_b,
        0.23,
        f"mean final LL\n"
        f"Fortran: {F_ll.mean():.4f} (sd {F_ll.std():.3f})\n"
        f"pyAMICA: {G_ll.mean():.4f} (sd {G_ll.std():.3f})\n"
        f"gap: {abs(F_ll.mean() - G_ll.mean()):.3f} (100-iter budget)\n"
        f"KS p={ks:.0e}",
        ha="center",
        va="top",
        fontsize=5.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )

    fig.text(
        0.5,
        0.90,
        "Dashed vertical lines mark each distribution's mean.",
        ha="center",
        fontsize=6.5,
        style="italic",
    )
    fig.suptitle(
        "Multi-model AMICA (n_models=2): pyAMICA vs Fortran ensembles, real sample EEG",
        fontweight="bold",
        fontsize=8.5,
        y=0.97,
    )
    fig.savefig(
        out / "multimodel_ensemble_distributions.png", bbox_inches="tight", dpi=300
    )
    fig.savefig(out / "multimodel_ensemble_distributions.pdf", bbox_inches="tight")


def main():
    if "--from-cache" in sys.argv:
        # Reuse the persisted ensemble (e.g. to regenerate the figure after a
        # labeling/legend change) instead of re-running 40 real fits.
        d = np.load(HERE / "ensemble.npz")
        Fs, Gs, F_ll, G_ll = d["Fs"], d["Gs"], d["F_ll"], d["G_ll"]
    else:
        n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
        data = load_data()
        work = Path(tempfile.mkdtemp(prefix="amica_ensemble_"))
        print(f"scratch: {work}")
        Fs, F_ll, Gs, G_ll = [], [], [], []
        for k in range(n):
            print(f"Fortran {k + 1}/{n}", flush=True)
            W, ll = run_fortran(work, str(k))
            Fs.append(W)
            F_ll.append(ll)
        for k in range(n):
            print(f"NG {k + 1}/{n}", flush=True)
            W, ll = run_ng(data, seed=k)
            Gs.append(W)
            G_ll.append(ll)
        Fs, Gs = np.array(Fs), np.array(Gs)
        F_ll, G_ll = np.array(F_ll), np.array(G_ll)

        # Persist the raw ensemble so the figure/tests can be regenerated without
        # re-running the 40 fits (the earlier run's data was lost, forcing this rerun).
        np.savez(HERE / "ensemble.npz", Fs=Fs, Gs=Gs, F_ll=F_ll, G_ll=G_ll)

    n = len(Fs)
    within_F = pairwise(Fs, Fs, True)
    within_G = pairwise(Gs, Gs, True)
    between = pairwise(Gs, Fs, False)
    diff = between.mean() - within_F.mean()
    # Valid run-level test (permutes whole runs), replacing the pseudoreplicated
    # Mann-Whitney/TOST on non-independent pairwise correlations.
    _, p_perm = perm_test_not_worse(Fs, Gs)
    ks = stats.ks_2samp(G_ll, F_ll).pvalue

    print(f"\n==== N={n} each ====")
    for name, a in [
        ("within-Fortran", within_F),
        ("within-NG", within_G),
        ("between", between),
    ]:
        print(f"{name:16s} mean={a.mean():.4f} sd={a.std():.4f} n={a.size}")
    print(f"between - within-Fortran mean diff = {diff:+.4f} (|diff| < {DELTA} margin)")
    print(
        f"run-level permutation (between not worse than within-Fortran): p={p_perm:.3f}"
    )
    print(
        f"LL Fortran={F_ll.mean():.4f}({F_ll.std():.3f}) "
        f"NG={G_ll.mean():.4f}({G_ll.std():.3f}) KS p={ks:.2e}"
    )
    figure(within_F, within_G, between, F_ll, G_ll, diff, p_perm, ks, HERE)
    print(f"figure -> {HERE / 'multimodel_ensemble_distributions.png'}")


if __name__ == "__main__":
    main()
