"""Issue #51 acceptance measurement: does the best-iterate safeguard (keep_best)
make NG's multi-model log-likelihood distribution equivalent to Fortran's?

Runs N Fortran + N NG fits on the real sample EEG (n_models=2, matched schedule),
comparing the final-LL distributions three ways:

  - Fortran vs NG *return-last*  (keep_best=False; reproduces the #51 defect)
  - Fortran vs NG *keep_best*    (keep_best=True; the fix, reports final_ll_)

Reports mean/sd, KS, and TOST (mean-equivalence within +/-DELTA) for each, and
the sd ratio. Real sample data + the macOS Fortran binary only (NO MOCK).

    uv run python .context/issue-51/ensemble_ll.py [N] [MAX_ITER]
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import stats

from pamica.torch_impl import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIN = REPO / "pamica/sample_data/amica15mac"
FDT = REPO / "pamica/sample_data/eeglab_data.fdt"
FIXTURE = REPO / "pamica/tests/torch_tests/_ng_e2e_tmp/fortran_run/input.param"
NW, FIELD = 32, 30504
DELTA_LL = 0.01  # TOST equivalence margin on the mean LL (per sample-channel)


def load_data():
    return load_eeglab_data(str(FDT), data_dim=NW, field_dim=FIELD).astype(np.float64)


def run_fortran(work, tag, max_iter):
    d = work / f"fort_{tag}"
    (d / "fortran_output").mkdir(parents=True, exist_ok=True)
    shutil.copy(FDT, d / "eeglab_data.fdt")
    lines = []
    for ln in FIXTURE.read_text().splitlines():
        if ln.startswith("num_models"):
            lines.append("num_models 2")
        elif ln.startswith("max_iter"):
            lines.append(f"max_iter {max_iter}")
        else:
            lines.append(ln)
    (d / "input.param").write_text("\n".join(lines) + "\n")
    orig = os.getcwd()
    os.chdir(d)
    try:
        r = subprocess.run(
            [str(BIN), "input.param"], capture_output=True, text=True, timeout=900
        )
    finally:
        os.chdir(orig)
    if r.returncode != 0:
        raise RuntimeError(r.stderr[-400:])
    return next(
        (
            float(ln.split("LL =")[1].split()[0])
            for ln in reversed(r.stdout.splitlines())
            if "LL =" in ln
        ),
        np.nan,
    )


def run_ng(data, seed, keep_best, max_iter):
    ng = AMICATorchNG(
        n_channels=NW, n_models=2, n_mix=3, block_size=512, lrate=0.05, minlrate=1e-8,
        lratefact=0.5, maxdecs=3, do_newton=True, newt_start=50, newt_ramp=10,
        newtrate=1.0, rho0=1.5, minrho=1.0, maxrho=2.0, rholrate=0.05,
        rholratefact=0.5, invsigmin=1e-8, invsigmax=100.0, doscaling=True,
        scalestep=1, seed=seed, device="cpu", keep_best=keep_best,
    )  # fmt: skip
    ng.fit(data, max_iter=max_iter, verbose=False)
    return ng.final_ll_


def compare(name, F, G):
    F, G = np.asarray(F), np.asarray(G)
    ks = stats.ks_2samp(G, F).pvalue
    diff = G.mean() - F.mean()
    se = np.sqrt(G.var(ddof=1) / G.size + F.var(ddof=1) / F.size)
    p_tost = max(
        stats.norm.sf((diff + DELTA_LL) / se), stats.norm.cdf((diff - DELTA_LL) / se)
    )
    print(
        f"  {name:16s} F={F.mean():.4f}(sd {F.std():.4f})  "
        f"NG={G.mean():.4f}(sd {G.std():.4f})  diff={diff:+.4f}  "
        f"sd_ratio={G.std() / max(F.std(), 1e-9):.1f}x  "
        f"KS p={ks:.1e}  TOST(+/-{DELTA_LL}) p={p_tost:.1e} "
        f"{'EQUIV' if p_tost < 0.05 else 'inconclusive'}"
    )


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    data = load_data()
    work = Path(tempfile.mkdtemp(prefix="amica_ll51_"))
    print(f"scratch: {work}  N={n}  max_iter={max_iter}")

    F = []
    for k in range(n):
        F.append(run_fortran(work, str(k), max_iter))
        print(f"  Fortran {k + 1}/{n}: LL={F[-1]:.4f}", flush=True)
    G_last, G_best = [], []
    for k in range(n):
        G_last.append(run_ng(data, k, False, max_iter))
        G_best.append(run_ng(data, k, True, max_iter))
        print(
            f"  NG {k + 1}/{n}: last={G_last[-1]:.4f} best={G_best[-1]:.4f}", flush=True
        )

    print(f"\n==== N={n} max_iter={max_iter} ====")
    compare("return-last", F, G_last)
    compare("keep_best", F, G_best)
    np.savez(
        HERE / "ensemble_ll.npz",
        F=F,
        G_last=G_last,
        G_best=G_best,
        n=n,
        max_iter=max_iter,
    )
    print(f"saved -> {HERE / 'ensemble_ll.npz'}")


if __name__ == "__main__":
    main()
