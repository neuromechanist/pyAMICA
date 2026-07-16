"""Single-model Fortran-vs-NG parity on the FULL ds002718 sub-002 recording
(70 channels, 747750 frames, k=152.6 -- the exact configuration issue #90's
k-sweep already validated at ~0.98 mean |corr|, native-fortran-f64 vs
torch-cuda-f64 specifically at 0.995), run on hallu: native Linux Fortran
build (24 threads) + PyTorch CUDA, across multiple independent seeds.

    uv run python .context/issue-144-parity-data-adequacy/test_ds002718_hallu.py <npy> [n_seeds] [max_iter] [threads]
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from pyAMICA.torch_impl import AMICATorchNG  # noqa: E402

BIN = REPO / "pyAMICA/sample_data/amica15_linux"
INPUT_PARAM = REPO / "pyAMICA/sample_data/input.param"


def xcorr(Wa, Wb):
    na = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-12)
    nb = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-12)
    corr = np.abs(na @ nb.T)
    r, c = linear_sum_assignment(1 - corr)
    return corr[r, c]


def write_fdt(data: np.ndarray, path: Path) -> None:
    """(n_channels, n_samples) -> amica's raw float32 .fdt, channel-fastest
    (column-major) order -- byte-identical to EEGLAB's own .fdt layout."""
    path.write_bytes(np.ascontiguousarray(data).astype("<f4").tobytes(order="F"))


def main():
    npy_path = Path(sys.argv[1])
    n_seeds = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    max_iter = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
    threads = int(sys.argv[4]) if len(sys.argv) > 4 else 24
    seed_offset = int(sys.argv[5]) if len(sys.argv) > 5 else 0

    data = np.load(npy_path).astype(np.float64)
    nw, field = data.shape
    k = field / nw**2
    print(
        f"data: {nw} channels x {field} frames, k={k:.1f}, max_iter={max_iter}, "
        f"threads={threads}",
        flush=True,
    )

    work = Path(tempfile.mkdtemp(prefix="ds002718_full_parity_"))
    fdt_path = work / "data.fdt"
    write_fdt(data, fdt_path)

    results = []
    for seed in range(seed_offset, seed_offset + n_seeds):
        d = work / f"run_{seed}"
        (d / "fortran_output").mkdir(parents=True, exist_ok=True)
        os.symlink(fdt_path, d / "data.fdt")
        lines = []
        for ln in INPUT_PARAM.read_text().splitlines():
            if ln.startswith("files"):
                lines.append("files ./data.fdt")
            elif ln.startswith("outdir"):
                lines.append("outdir ./fortran_output/")
            elif ln.startswith("data_dim"):
                lines.append(f"data_dim {nw}")
            elif ln.startswith("field_dim"):
                lines.append(f"field_dim {field}")
            elif ln.startswith("max_iter"):
                lines.append(f"max_iter {max_iter}")
            elif ln.startswith("max_threads"):
                lines.append(f"max_threads {threads}")
            elif ln.startswith("pcakeep"):
                # AMICATorchNG has no PCA source reduction (n_sources ==
                # n_channels always), so pcakeep must track nw or Fortran's
                # W comes back reduced-rank (the template's literal 32 is
                # only correct by coincidence at nw=32).
                lines.append(f"pcakeep {nw}")
            elif ln.startswith("use_min_dll"):
                # Force the full max_iter budget instead of Fortran's own
                # early-stopping (matches benchmark_dimsweep.py's convention):
                # otherwise Fortran can converge and stop well short of
                # max_iter while AMICATorchNG (no early-stopping equivalent)
                # keeps optimizing past that point, letting weakly-determined
                # components drift to a different, still-valid optimum -- an
                # asymmetry, not real disagreement.
                lines.append("use_min_dll 0")
            elif ln.startswith("use_grad_norm"):
                lines.append("use_grad_norm 0")
            else:
                lines.append(ln)
        (d / "input.param").write_text("\n".join(lines) + "\n")

        orig = os.getcwd()
        os.chdir(d)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
        try:
            print(
                f"seed {seed}: running Fortran (native, {threads} threads)...",
                flush=True,
            )
            r = subprocess.run(
                [str(BIN), "input.param"], capture_output=True, text=True,
                timeout=1800, env=env,
            )  # fmt: skip
        finally:
            os.chdir(orig)
        if r.returncode != 0:
            print(f"seed {seed}: Fortran failed: {r.stderr[-500:]}", flush=True)
            continue
        W_fortran = np.fromfile(d / "fortran_output/W", dtype=np.float64).reshape(
            nw, nw, order="F"
        )
        fort_ll = next(
            (
                float(ln.split("LL =")[1].split()[0])
                for ln in reversed(r.stdout.splitlines())
                if "LL =" in ln
            ),
            None,
        )
        if fort_ll is None:
            print(
                f"seed {seed}: WARNING could not parse LL from Fortran stdout",
                flush=True,
            )
            fort_ll = float("nan")

        print(f"seed {seed}: running AMICATorchNG on CUDA...", flush=True)
        m = AMICATorchNG(
            n_channels=nw, n_models=1, n_mix=3, block_size=512, lrate=0.05,
            minlrate=1e-8, lratefact=0.5, maxdecs=3, do_newton=True,
            newt_start=50, newt_ramp=10, newtrate=1.0, rho0=1.5, minrho=1.0,
            maxrho=2.0, rholrate=0.05, rholratefact=0.5, invsigmin=0.0,
            invsigmax=100.0, doscaling=True, scalestep=1, seed=seed, device="cuda",
        )  # fmt: skip
        m.fit(data, max_iter=max_iter, verbose=False)
        if m.stop_reason in AMICATorchNG._DEGENERATE_STOP_REASONS:
            print(
                f"seed {seed}: NG fit ended degenerate (stop_reason={m.stop_reason!r}), skipping",
                flush=True,
            )
            continue
        W_ng = m.get_unmixing_matrix(0)

        corrs = xcorr(W_fortran, W_ng)
        n_above_95 = int((corrs > 0.95).sum())
        print(
            f"seed {seed}: mean_corr={corrs.mean():.4f} min={corrs.min():.4f} "
            f"n_above_0.95={n_above_95}/{nw} "
            f"fortran_LL={fort_ll:.4f} ng_LL={m.final_ll_:.4f}",
            flush=True,
        )
        results.append(corrs.mean())

    if results:
        print(
            f"\n{nw}ch/{field}fr (k={k:.1f}), n={len(results)}: "
            f"mean={np.mean(results):.4f} range={np.min(results):.4f}-{np.max(results):.4f}",
            flush=True,
        )

    if len(results) < n_seeds:
        print(
            f"\n{n_seeds - len(results)}/{n_seeds} seed(s) failed or were degenerate",
            flush=True,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
