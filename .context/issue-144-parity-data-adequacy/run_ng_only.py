"""NG (CUDA) phase + correlation for one seed, picking up a Fortran W already
written by run_fortran_only.py to the same out_dir. Run these sequentially
(one GPU) after the Fortran-only jobs (which can run concurrently on CPU).

    uv run python run_ng_only.py <npy> <seed> <max_iter> <out_dir>
"""

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from pyAMICA.torch_impl import AMICATorchNG  # noqa: E402


def xcorr(Wa, Wb):
    na = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-12)
    nb = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-12)
    corr = np.abs(na @ nb.T)
    r, c = linear_sum_assignment(1 - corr)
    return corr[r, c]


def main():
    npy_path = Path(sys.argv[1])
    seed = int(sys.argv[2])
    max_iter = int(sys.argv[3])
    out_dir = Path(sys.argv[4])

    data = np.load(npy_path).astype(np.float64)
    nw, field = data.shape

    W_fortran = np.fromfile(out_dir / "fortran_output/W", dtype=np.float64).reshape(
        nw, nw, order="F"
    )
    fort_ll = float((out_dir / "fortran_ll.txt").read_text())

    print(f"seed {seed}: running AMICATorchNG on CUDA...", flush=True)
    m = AMICATorchNG(
        n_channels=nw, n_models=1, n_mix=3, block_size=512, lrate=0.05,
        minlrate=1e-8, lratefact=0.5, maxdecs=3, do_newton=True,
        newt_start=50, newt_ramp=10, newtrate=1.0, rho0=1.5, minrho=1.0,
        maxrho=2.0, rholrate=0.05, rholratefact=0.5, invsigmin=0.0,
        invsigmax=100.0, doscaling=True, scalestep=1, seed=seed, device="cuda",
    )  # fmt: skip
    m.fit(data, max_iter=max_iter, verbose=False)
    W_ng = m.get_unmixing_matrix(0)

    corrs = xcorr(W_fortran, W_ng)
    n_above_95 = int((corrs > 0.95).sum())
    print(
        f"seed {seed}: mean_corr={corrs.mean():.4f} min={corrs.min():.4f} "
        f"n_above_0.95={n_above_95}/{nw} fortran_LL={fort_ll:.4f} ng_LL={m.final_ll_:.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
