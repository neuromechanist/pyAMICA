#!/usr/bin/env python
"""Phase 3 (#87): full-decomposition equivalence + total-time benchmark.

Runs each AMICA backend to a full decomposition (default 2000 iters) on real EEG,
saving total wall-clock time, final log-likelihood, and the unmixing matrix W.
A ``--compare`` pass Hungarian-matches W across backends to show they all recover
the SAME independent components (the equivalence figure), and tabulates total
decompose time and LL.

Three axes (channel x core x FP):
  * channels  -- sweep with --channels (each needs k = frames/ch^2 >= ~20-30 for
    a well-determined decomposition; the k30 data has 147k frames -> k=30 at 70ch)
  * cores     -- --threads for the CPU backends (torch-cpu, native-fortran)
  * precision -- --dtypes f64,f32 (torch-cpu/cuda run both; mlx/mps are f32-only;
    native-fortran is f64-only)

numpy is intentionally excluded (too slow at k30 and not a backend to recommend).

Usage:
    # run one machine's backends to 2000 iters, save per-run npz into a dir
    uv run python benchmarks/benchmark_decompose.py --data DATA.npy \
        --channels 70 --iters 2000 --backends native-fortran-f64,torch-cuda-f64,torch-cuda-f32 \
        --out results_hallu/
    # merge every machine's dir and emit the equivalence figure + tables
    uv run python benchmarks/benchmark_decompose.py --compare results_hallu/ results_mac/ \
        --figure equivalence.png
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

N_MIX = 3
SEED = 42
BLOCK_SIZE = 512
_REPO = Path(__file__).resolve().parent.parent
_FORTRAN_TIMEOUT = 36000  # a full 2000-iter fit can be long at k30 frames


def _load_dimsweep():
    """Reuse the Fortran .fdt writer / param renderer / out.txt parser (#85)."""
    path = _REPO / "benchmarks" / "benchmark_dimsweep.py"
    spec = importlib.util.spec_from_file_location("benchmark_dimsweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ds = _load_dimsweep()


# --------------------------------------------------------------------------
# Per-backend fit + extract: returns dict(time, final_ll, W, A). W is the
# unmixing matrix in the (n_comps x n_channels) convention get_unmixing_matrix
# uses (rows = components), so all backends are directly Hungarian-comparable.
# --------------------------------------------------------------------------
def _fit_torch(data, device, dtype_str, iters, threads=None):
    import torch

    from pyAMICA.torch_impl import AMICATorchNG

    dtype = torch.float64 if dtype_str == "f64" else torch.float32
    prev = torch.get_num_threads()
    try:
        if device == "cpu" and threads:
            torch.set_num_threads(int(threads))
        m = AMICATorchNG(
            n_channels=data.shape[0],
            n_models=1,
            n_mix=N_MIX,
            device=device,
            dtype=dtype,
            do_newton=False,
            block_size=BLOCK_SIZE,
            seed=SEED,
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = perf_counter()
        m.fit(data, max_iter=iters, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = perf_counter() - t0
    finally:
        if device == "cpu":
            torch.set_num_threads(prev)
    return {
        "time": elapsed,
        "final_ll": float(m.final_ll_),
        "W": m.get_unmixing_matrix(0),
        "A": m.get_mixing_matrix(0),
    }


def _fit_mlx(data, iters):

    from pyAMICA.mlx_impl import AMICAMLXNG

    m = AMICAMLXNG(
        n_channels=data.shape[0],
        n_models=1,
        n_mix=N_MIX,
        do_newton=False,
        block_size=BLOCK_SIZE,
        seed=SEED,
    )
    t0 = perf_counter()
    m.fit(data, max_iter=iters, verbose=False)  # fit's mx.eval syncs
    elapsed = perf_counter() - t0
    # convert mlx -> numpy FIRST, then index (mlx cannot be indexed by a numpy array)
    A_np = np.array(m.A)
    W_np = np.array(m.W)
    idx = np.array(m.comp_list)[:, 0].astype(int)
    W = W_np[0].T  # (comps x channels), matching torch get_unmixing convention
    A = A_np[:, idx].T
    return {"time": elapsed, "final_ll": float(m.final_ll_), "W": W, "A": A}


def _fit_fortran(data, iters, binary, threads):
    binary = Path(binary)
    if not (binary.exists() and os.access(binary, os.X_OK)):
        raise RuntimeError(f"native fortran binary not found/executable: {binary}")
    threads = threads or (os.cpu_count() or 4)
    nc, ns = data.shape
    with tempfile.TemporaryDirectory(prefix="amica_decomp_") as td:
        work = Path(td)
        _ds._write_fdt(data, work / "bench.fdt")
        _ds._write_fortran_param(work, nc, ns, iters, threads, 1)
        (work / "bench_out").mkdir(exist_ok=True)
        env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
        t0 = perf_counter()
        res = subprocess.run(
            [str(binary), "input.param"],
            cwd=work,
            env=env,
            capture_output=True,
            text=True,
            timeout=_FORTRAN_TIMEOUT,
        )
        elapsed = perf_counter() - t0
        if res.returncode != 0:
            print(f"    fortran exit {res.returncode}\n{res.stderr[-2000:]}")
            raise RuntimeError(f"native fortran run failed (exit {res.returncode})")
        _secs, final_ll = _ds._parse_fortran_out(work / "bench_out" / "out.txt")
        # Fortran writes W/A as (n x n) float64 in column-major order; W's rows are
        # components (same convention validate_implementations.py matches on).
        W = np.fromfile(work / "bench_out" / "W", dtype=np.float64).reshape(
            nc, nc, order="F"
        )
        A = np.fromfile(work / "bench_out" / "A", dtype=np.float64).reshape(
            nc, nc, order="F"
        )
    return {"time": elapsed, "final_ll": final_ll, "W": W, "A": A}


# backend name -> (kind, kwargs); kind selects the fit function above
_BACKENDS = {
    "torch-cpu-f64": ("torch", {"device": "cpu", "dtype_str": "f64"}),
    "torch-cpu-f32": ("torch", {"device": "cpu", "dtype_str": "f32"}),
    "torch-cuda-f64": ("torch", {"device": "cuda", "dtype_str": "f64"}),
    "torch-cuda-f32": ("torch", {"device": "cuda", "dtype_str": "f32"}),
    "torch-mps-f32": ("torch", {"device": "mps", "dtype_str": "f32"}),
    "mlx-f32": ("mlx", {}),
    "native-fortran-f64": ("fortran", {}),
}


def _fit(name, data, iters, fortran_bin, threads):
    kind, kw = _BACKENDS[name]
    if kind == "torch":
        return _fit_torch(data, iters=iters, threads=threads, **kw)
    if kind == "mlx":
        return _fit_mlx(data, iters=iters)
    return _fit_fortran(data, iters=iters, binary=fortran_bin, threads=threads)


def _platform_tag() -> str:
    import platform

    return f"{platform.system()}-{platform.machine()}"


# --------------------------------------------------------------------------
# Equivalence: Hungarian-match unmixing rows (components) between two runs.
# --------------------------------------------------------------------------
def _match_correlation(W1: np.ndarray, W2: np.ndarray) -> np.ndarray:
    """Best per-component |correlation| after optimal (Hungarian) assignment.
    Rows are unit-normalized so W @ W.T is the cosine similarity; abs handles the
    ICA sign ambiguity, linear_sum_assignment handles the permutation."""
    from scipy.optimize import linear_sum_assignment

    n1 = W1 / (np.linalg.norm(W1, axis=1, keepdims=True) + 1e-12)
    n2 = W2 / (np.linalg.norm(W2, axis=1, keepdims=True) + 1e-12)
    corr = np.abs(n1 @ n2.T)
    row, col = linear_sum_assignment(1.0 - corr)
    return corr[row, col]


def _compare(dirs, figure=None):
    runs = []
    for d in dirs:
        for f in sorted(Path(d).glob("*.npz")):
            z = np.load(f, allow_pickle=True)
            runs.append(
                {
                    "label": str(z["label"]),
                    "channels": int(z["channels"]),
                    "time": float(z["time"]),
                    "final_ll": float(z["final_ll"]),
                    "W": z["W"],
                }
            )
    if not runs:
        print("no result npz found")
        return

    # equivalence is per channel-count (W shapes must match); use the largest
    by_ch: dict = {}
    for r in runs:
        by_ch.setdefault(r["channels"], []).append(r)
    target = max(by_ch)
    group = by_ch[target]
    labels = [r["label"] for r in group]
    n = len(group)

    print(f"\n=== decompose time + final LL @ {target} channels, sorted by time ===")
    for r in sorted(group, key=lambda x: x["time"]):
        print(f"  {r['label']:24s} {r['time']:8.1f} s   LL={r['final_ll']:+.5f}")

    print(f"\n=== IC equivalence: mean Hungarian-matched |corr| @ {target}ch ===")
    M = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            c = _match_correlation(group[i]["W"], group[j]["W"]).mean()
            M[i, j] = M[j, i] = c
    hdr = "                          " + " ".join(f"{lb[:8]:>8}" for lb in labels)
    print(hdr)
    for i, lb in enumerate(labels):
        print(f"  {lb:24s} " + " ".join(f"{M[i, j]:8.3f}" for j in range(n)))
    offdiag = M[~np.eye(n, dtype=bool)]
    print(
        f"\n  cross-backend mean |corr| = {offdiag.mean():.4f} "
        f"(min {offdiag.min():.4f}) -- ~1.0 => same ICs recovered"
    )

    if figure:
        _plot(labels, M, target, figure)


def _plot(labels, M, channels, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2, 1.0 * len(labels) + 1.5))
    im = ax.imshow(M, vmin=0.9, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(
                j,
                i,
                f"{M[i, j]:.3f}",
                ha="center",
                va="center",
                color="white" if M[i, j] < 0.97 else "black",
                fontsize=7,
            )
    ax.set_title(
        f"AMICA cross-backend IC equivalence @ {channels} channels\n"
        "mean Hungarian-matched |correlation| of unmixing components"
    )
    fig.colorbar(im, ax=ax, label="mean |corr|")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"wrote {path}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="path to the (channels, frames) real-EEG .npy")
    ap.add_argument("--channels", default="70")
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--backends", default="auto")
    ap.add_argument(
        "--fortran-bin",
        default=str(_REPO / "benchmarks/fortran/amica15"),
        dest="fortran_bin",
    )
    ap.add_argument("--out", help="directory to write per-run npz into")
    ap.add_argument("--compare", nargs="+", help="result dirs -> equivalence report")
    ap.add_argument("--figure", help="write the equivalence heatmap PNG here")
    args = ap.parse_args()

    if args.compare:
        _compare(args.compare, args.figure)
        return 0
    if not args.data or not args.out:
        ap.error("--data and --out are required unless --compare is given")

    full = np.load(args.data).astype(np.float64)
    channels = [int(c) for c in args.channels.split(",") if int(c) <= full.shape[0]]
    backends = list(_BACKENDS) if args.backends == "auto" else args.backends.split(",")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = _platform_tag()
    print(f"platform {tag} | channels {channels} | iters {args.iters} | {backends}")

    for nc in channels:
        data = np.ascontiguousarray(full[:nc, :])
        k = data.shape[1] / nc**2
        print(f"\n{nc}ch, {data.shape[1]} frames (k={k:.1f})")
        for b in backends:
            label = f"{b}@{tag}"
            try:
                t0 = perf_counter()
                res = _fit(b, data, args.iters, args.fortran_bin, args.threads)
                fname = out / f"{b}_{nc}ch_{tag}.npz"
                np.savez_compressed(
                    fname,
                    label=label,
                    backend=b,
                    channels=nc,
                    iters=args.iters,
                    time=res["time"],
                    final_ll=res["final_ll"],
                    W=res["W"],
                    A=res["A"],
                )
                print(
                    f"  {b:24s} {res['time']:8.1f} s   LL={res['final_ll']:+.5f}"
                    f"   [{perf_counter() - t0:.0f}s wall]  -> {fname.name}"
                )
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  {b:24s} FAILED: {type(exc).__name__}: {str(exc)[:80]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
