"""Render the issue #51 before/after LL distribution figure from ensemble_ll.npz
(written by ensemble_ll.py). Fortran vs NG return-last vs NG keep_best."""

from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
C_FORT, C_LAST, C_BEST = "#0072B2", "#D55E00", "#009E73"  # Okabe-Ito


def main():
    d = np.load(HERE / "ensemble_ll.npz")
    F, G_last, G_best = d["F"], d["G_last"], d["G_best"]
    n, mi = int(d["n"]), int(d["max_iter"])

    plt.rcParams.update(
        {"font.size": 11, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    lo = min(F.min(), G_last.min()) - 0.005
    hi = max(F.max(), G_last.max()) + 0.005
    bins = np.linspace(lo, hi, 40)
    for arr, col, lab in [
        (F, C_FORT, f"Fortran  ({F.mean():.4f}, sd {F.std():.4f})"),
        (
            G_last,
            C_LAST,
            f"NG return-last  ({G_last.mean():.4f}, sd {G_last.std():.4f})",
        ),
        (G_best, C_BEST, f"NG keep_best  ({G_best.mean():.4f}, sd {G_best.std():.4f})"),
    ]:
        ax.hist(arr, bins=bins, density=True, color=col, alpha=0.32)
        ax.hist(
            arr, bins=bins, density=True, histtype="step", color=col, lw=2, label=lab
        )
        ax.axvline(arr.mean(), color=col, ls="--", lw=1.2)
    ax.set_xlabel("final log-likelihood (per sample-channel)")
    ax.set_ylabel("density")
    ax.set_title(
        f"Multi-model AMICA (n_models=2, N={n}, {mi} iters): LL distributions",
        loc="left",
        fontweight="bold",
    )
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.text(
        0.98,
        0.97,
        "keep_best (issue #51): sd 12.7x -> 2.0x Fortran,\n"
        "mean gap 0.020 -> 0.009. Removes the low-LL\n"
        "overshoot tail; small residual is optimizer\n"
        "efficiency (NG peaks sit ~0.009 below Fortran).",
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=8.5,
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )
    fig.tight_layout()
    fig.savefig(HERE / "ll_before_after.png", bbox_inches="tight", dpi=200)
    fig.savefig(HERE / "ll_before_after.pdf", bbox_inches="tight")
    print(f"figure -> {HERE / 'll_before_after.png'}")


if __name__ == "__main__":
    main()
