"""Multi-model NG-vs-Fortran distributional-equivalence harness (issue #27).

Reproduces `multimodel_distributional_equivalence.md`: runs N Fortran + N NG
fits on the real sample EEG (n_models=2), then compares the within-Fortran,
within-NG, and between (NG-Fortran) distributions of the stacked 2*NW Hungarian
cross-correlation, plus the log-likelihood distributions. Writes the figure
next to this script. Real sample data + Fortran binary only (NO MOCK).

    uv run python .context/issue-27/multimodel_ensemble.py [N]

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


def figure(within_F, within_G, between, F_ll, G_ll, mw, p_tost, ks, out):
    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.3))
    bins = np.linspace(0.5, 1.0, 26)
    for arr, col, lab in [
        (within_F, C_FORT, "within-Fortran"),
        (within_G, C_NG, "within-NG"),
        (between, C_BET, "between (NG-Fortran)"),
    ]:
        axA.hist(arr, bins=bins, density=True, color=col, alpha=0.35)
        axA.hist(
            arr, bins=bins, density=True, histtype="step", color=col, lw=2, label=lab
        )
        axA.axvline(arr.mean(), color=col, ls="--", lw=1.2)
    axA.set_xlabel("stacked 2x32 Hungarian cross-correlation")
    axA.set_ylabel("density")
    axA.set_title("A  Partition-agreement distributions", loc="left", fontweight="bold")
    axA.legend(frameon=False, fontsize=9, loc="upper right")
    axA.text(
        0.02,
        0.97,
        f"means: F-F {within_F.mean():.3f} | NG-NG {within_G.mean():.3f} | "
        f"NG-F {between.mean():.3f}\nMann-Whitney (between<within-F): p={mw:.2f}\n"
        f"TOST equivalence (±0.05): p={p_tost:.0e} → EQUIVALENT",
        transform=axA.transAxes,
        va="top",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )
    bins_ll = np.linspace(
        min(F_ll.min(), G_ll.min()) - 0.005, max(F_ll.max(), G_ll.max()) + 0.005, 24
    )
    for arr, col, lab in [(F_ll, C_FORT, "Fortran"), (G_ll, C_NG, "NG")]:
        axB.hist(arr, bins=bins_ll, density=True, color=col, alpha=0.35)
        axB.hist(
            arr, bins=bins_ll, density=True, histtype="step", color=col, lw=2, label=lab
        )
        axB.axvline(arr.mean(), color=col, ls="--", lw=1.2)
    axB.set_xlabel("final log-likelihood (per sample-channel)")
    axB.set_ylabel("density")
    axB.set_title("B  Likelihood distributions", loc="left", fontweight="bold")
    axB.legend(frameon=False, fontsize=9, loc="upper left")
    axB.text(
        0.98,
        0.97,
        f"Fortran {F_ll.mean():.4f} (sd {F_ll.std():.3f})\n"
        f"NG {G_ll.mean():.4f} (sd {G_ll.std():.3f})\nKS p={ks:.0e}",
        transform=axB.transAxes,
        va="top",
        ha="right",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )
    fig.suptitle(
        "Multi-model AMICA (n_models=2): NG vs Fortran ensembles, real sample EEG",
        fontweight="bold",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(
        out / "multimodel_ensemble_distributions.png", bbox_inches="tight", dpi=200
    )
    fig.savefig(out / "multimodel_ensemble_distributions.pdf", bbox_inches="tight")


def main():
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

    within_F = pairwise(Fs, Fs, True)
    within_G = pairwise(Gs, Gs, True)
    between = pairwise(Gs, Fs, False)
    mw = stats.mannwhitneyu(between, within_F, alternative="less").pvalue
    diff = between.mean() - within_F.mean()
    se = np.sqrt(
        between.var(ddof=1) / between.size + within_F.var(ddof=1) / within_F.size
    )
    p_tost = max(
        stats.norm.sf((diff + DELTA) / se), stats.norm.cdf((diff - DELTA) / se)
    )
    ks = stats.ks_2samp(G_ll, F_ll).pvalue

    print(f"\n==== N={n} each ====")
    for name, a in [
        ("within-Fortran", within_F),
        ("within-NG", within_G),
        ("between", between),
    ]:
        print(f"{name:16s} mean={a.mean():.4f} sd={a.std():.4f} n={a.size}")
    print(f"Mann-Whitney (between<within-F): p={mw:.3f}")
    print(
        f"TOST (±{DELTA}): diff={diff:+.4f} p={p_tost:.2e} "
        f"{'EQUIVALENT' if p_tost < 0.05 else 'inconclusive'}"
    )
    print(
        f"LL Fortran={F_ll.mean():.4f}({F_ll.std():.3f}) "
        f"NG={G_ll.mean():.4f}({G_ll.std():.3f}) KS p={ks:.2e}"
    )
    figure(within_F, within_G, between, F_ll, G_ll, mw, p_tost, ks, HERE)
    print(f"figure -> {HERE / 'multimodel_ensemble_distributions.png'}")


if __name__ == "__main__":
    main()
