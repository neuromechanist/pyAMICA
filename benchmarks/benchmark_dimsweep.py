#!/usr/bin/env python
"""Cross-platform result + performance benchmark for AMICA backends (issue #77).

Sweeps the channel count on real 70-channel EEG (OpenNeuro ds002718 sub-002, the
Wakeman-Henson faces data) and, for every backend the current host supports,
records BOTH performance (ms/iteration, warmed, min-of-repeats) and results
(converged log-likelihood) so the two questions the epic (#74) has deferred can
finally be answered together:

  * do the backends/platforms/precisions agree on the *result*?
  * where does an Apple/NVIDIA GPU actually *beat* the CPU?

Backends (auto-detected per platform):
  numpy-cpu-f64   AMICA_NumPy (the float64 reference implementation)
  torch-cpu-f64   AMICATorchNG, CPU, float64 (the Fortran-parity path)
  torch-cpu-f32   AMICATorchNG, CPU, float32
  torch-mps-f32   AMICATorchNG, Apple GPU (MPS), float32
  torch-cuda-f64  AMICATorchNG, NVIDIA GPU, float64
  torch-cuda-f32  AMICATorchNG, NVIDIA GPU, float32
  mlx-f32         AMICAMLXNG, Apple GPU (MLX), float32
  native-fortran-f64  amica15 compiled from source (MPI+OpenMP), the reference
                  (#85; timed via amica's own per-iter stamps, startup-immune;
                  needs benchmarks/fortran/build_amica.sh -- not on Apple/CI)

All backends run at matched settings (n_models=1, n_mix=3, pdftype=0,
do_newton=False, same block_size/seed/channels/samples/iters) so the comparison
is apples-to-apples. Emits JSON (``--out``) so a Mac run (cpu/mps/mlx) and a CUDA
host run can be merged by ``--report``.

Data: pass ``--data path/to/ds002718_sub-002_eeg70.npy`` -- a (70, n_samples)
float64 array of the 70 EEG channels (see ``benchmarks/README_dimsweep.md`` for
how to fetch/extract it; not committed).

Usage:
    uv run python benchmarks/benchmark_dimsweep.py --data DATA --out mac.json
    uv run python benchmarks/benchmark_dimsweep.py --data DATA --backends torch-cuda-f64,torch-cuda-f32 --out cuda.json
    uv run python benchmarks/benchmark_dimsweep.py --report mac.json cuda.json
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter

import numpy as np

# Matched fit settings for every backend.
N_MIX = 3
SEED = 42
BLOCK_SIZE = 512

# Native-Fortran backend (#85): paths + run limits.
_REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = _REPO_ROOT / "pyAMICA" / "sample_data"
_PARAM_TEMPLATE = SAMPLE_DIR / "input.param"
_DEFAULT_FORTRAN_BIN = _REPO_ROOT / "benchmarks" / "fortran" / "amica15"
_FORTRAN_TIMEOUT = 3600  # seconds; a crashed/hung run must not hang the sweep


# --------------------------------------------------------------------------
# Backend adapters: each returns (ms_per_iter, final_ll) for one fit.
# --------------------------------------------------------------------------
def _torch_available(device: str) -> bool:
    try:
        import torch
    except ImportError:
        return False
    if device == "cpu":
        return True
    if device == "mps":
        return torch.backends.mps.is_available()
    if device == "cuda":
        return torch.cuda.is_available()
    return False


def _share_kw_torch(share):
    # share_start/share_iter tuned so a merge actually fires inside the short
    # benchmark budget (share_iter must be > 6); no-op when share is off.
    return (
        dict(share_comps=True, share_start=5, share_iter=8, comp_thresh=0.99)
        if share
        else {}
    )


def _run_torch(data, device, dtype_str, iters, repeats, n_models=1, share=False):
    import torch

    from pyAMICA.torch_impl import AMICATorchNG

    dtype = torch.float64 if dtype_str == "f64" else torch.float32

    def one(n_iter):
        m = AMICATorchNG(
            n_channels=data.shape[0],
            n_models=n_models,
            n_mix=N_MIX,
            device=device,
            dtype=dtype,
            do_newton=False,
            block_size=BLOCK_SIZE,
            seed=SEED,
            **_share_kw_torch(share),
        )
        if device == "cuda":
            torch.cuda.synchronize()
        t0 = perf_counter()
        m.fit(data, max_iter=n_iter, verbose=False)
        if device == "cuda":
            torch.cuda.synchronize()
        return perf_counter() - t0, float(m.final_ll_)

    one(min(5, iters))  # warmup (kernel compile / context init)
    times, ll = [], float("nan")
    for _ in range(repeats):
        t, ll = one(iters)
        times.append(t)
    return min(times) / iters * 1000.0, ll


def _run_mlx(data, iters, repeats, n_models=1):
    from pyAMICA.mlx_impl import AMICAMLXNG

    def one(n_iter):
        m = AMICAMLXNG(
            n_channels=data.shape[0],
            n_models=n_models,
            n_mix=N_MIX,
            do_newton=False,
            block_size=BLOCK_SIZE,
            seed=SEED,
        )
        t0 = perf_counter()
        m.fit(data, max_iter=n_iter, verbose=False)  # fit's mx.eval syncs
        return perf_counter() - t0, float(m.final_ll_)

    one(min(5, iters))
    times, ll = [], float("nan")
    for _ in range(repeats):
        t, ll = one(iters)
        times.append(t)
    return min(times) / iters * 1000.0, ll


def _run_numpy(data, iters, repeats, n_models=1, share=False):
    from pyAMICA.numpy_impl.core import AMICA as AMICA_NumPy

    # NumPy uses share_int (torch uses share_iter).
    share_kw = (
        dict(share_comps=True, share_start=5, share_int=8, comp_thresh=0.99)
        if share
        else {}
    )

    def one(n_iter):
        m = AMICA_NumPy(
            num_models=n_models,
            num_mix=N_MIX,
            max_iter=n_iter,
            do_newton=False,
            # Match the fixed block_size and skip AMICA_NumPy's do_opt_block
            # auto-tune, which otherwise runs a wall-clock block-size search
            # INSIDE fit() -- unmatched settings + timing pollution vs torch/mlx.
            block_size=BLOCK_SIZE,
            do_opt_block=False,
            use_tqdm=False,
            verbose=False,
            **share_kw,
        )
        t0 = perf_counter()
        m.fit(data)
        # AMICA_NumPy.ll is summed over samples*channels; normalize to the
        # per-sample-per-channel scale the torch/mlx backends report, so the
        # results column is directly comparable (numpy-f64 then matches
        # torch-cpu-f64 to ~5 digits, as the parity suite guarantees).
        ll = float(m.ll[-1]) / (data.shape[0] * data.shape[1]) if m.ll else float("nan")
        return perf_counter() - t0, ll

    one(min(3, iters))
    times, ll = [], float("nan")
    for _ in range(repeats):
        t, ll = one(iters)
        times.append(t)
    return min(times) / iters * 1000.0, ll


# --------------------------------------------------------------------------
# Native-Fortran backend (#85): drives an amica15 compiled from source (see
# benchmarks/fortran/build_amica.sh). Timing comes from amica's OWN per-iter
# stamps in out.txt -- startup-immune (init/PCA/sphering happen before iter 1),
# unlike wrapping the whole process in perf_counter.
# --------------------------------------------------------------------------
def _write_fdt(data: np.ndarray, path: Path) -> None:
    """Write (n_channels, n_samples) EEG as amica's raw float32 .fdt: a
    (data_dim=channels, field_dim=samples) array in column-major (channel-
    fastest) order -- the exact inverse of numpy_impl.data.load_data_file's
    ``reshape(..., order="F")``. Byte-identical to EEGLAB's own .fdt (locked by
    test_write_fdt_roundtrip)."""
    path.write_bytes(np.ascontiguousarray(data).astype("<f4").tobytes(order="F"))


def _write_fortran_param(
    work: Path, nc: int, ns: int, iters: int, threads: int, n_models: int
) -> None:
    """Render input.param for one benchmark fit: start from the committed
    template and override only the benchmark knobs so the algorithm config
    matches the other backends (n_mix=3, pdftype=0, do_newton off, fixed
    block_size). use_min_dll/use_grad_norm are forced OFF so amica runs the full
    ``iters`` budget (matched, fixed-length runs like torch/mlx)."""
    overrides = {
        "files": "files ./bench.fdt",
        "outdir": "outdir ./bench_out/",
        "block_size": f"block_size {BLOCK_SIZE}",
        "do_opt_block": "do_opt_block 0",
        "num_models": f"num_models {n_models}",
        "max_threads": f"max_threads {threads}",
        "num_mix_comps": f"num_mix_comps {N_MIX}",
        "pdftype": "pdftype 0",
        "max_iter": f"max_iter {iters}",
        "num_samples": "num_samples 1",
        "data_dim": f"data_dim {nc}",
        "field_dim": f"field_dim {ns}",
        "do_newton": "do_newton 0",
        "do_sphere": "do_sphere 1",
        "do_mean": "do_mean 1",
        "doPCA": "doPCA 1",
        "pcakeep": f"pcakeep {nc}",
        "use_min_dll": "use_min_dll 0",
        "use_grad_norm": "use_grad_norm 0",
        "write_LLt": "write_LLt 0",
        "do_history": "do_history 0",
        "share_comps": "share_comps 0",
    }
    seen: set[str] = set()
    lines: list[str] = []
    for line in _PARAM_TEMPLATE.read_text().splitlines():
        key = line.split()[0] if line.strip() else ""
        if key in overrides:
            lines.append(overrides[key])
            seen.add(key)
        else:
            lines.append(line)
    lines.extend(v for k, v in overrides.items() if k not in seen)
    (work / "input.param").write_text("\n".join(lines) + "\n")


_ITER_RE = re.compile(
    r"iter\s+\d+\s+lrate\s*=\s*\S+\s+LL\s*=\s*(\S+).*?\(\s*([\d.]+)\s*s,"
)


def _parse_fortran_out(out_txt: Path) -> tuple[list[float], float]:
    """Parse amica's out.txt -> (per_iteration_seconds, final_ll). Each iter
    line ends '( <sec> s, <cum_h> h)'; <sec> is that iteration's compute time."""
    secs: list[float] = []
    last_ll = float("nan")
    for line in out_txt.read_text(errors="ignore").splitlines():
        m = _ITER_RE.search(line)
        if m:
            last_ll = float(m.group(1))
            secs.append(float(m.group(2)))
    return secs, last_ll


def _run_fortran(
    data, iters, repeats, n_models=1, binary=None, threads=None
) -> tuple[float, float]:
    """Time a native amica fit. Per repeat: fresh workdir, write .fdt + param,
    run the binary, read amica's per-iter seconds. ms/iter = MEAN of the timed
    iters (dropping iter 1 as first-touch warmup), reported as the min over
    repeats.

    amica prints per-iteration time to only ~0.01 s (10 ms) resolution, so the
    mean over the timed iters is used. NOTE this floors precision at ~10 ms: when
    every iteration rounds to the same stamp the mean is still that stamp, so run
    at a size whose per-iteration time is well above 10 ms (the 70-ch benchmark
    is ~30-70 ms) for the number to be meaningful. A config whose per-iteration
    time rounds to 0.00 s (below the floor -- only tiny channel/sample counts)
    raises instead of returning a bogus 0.0. Raises on a nonzero exit too -- a
    crashed run must not masquerade as a fast one (mirrors
    benchmark_runtime.time_fortran)."""
    binary = Path(binary or _DEFAULT_FORTRAN_BIN)
    if not (binary.exists() and os.access(binary, os.X_OK)):
        raise RuntimeError(f"native fortran binary not found/executable: {binary}")
    threads = threads or (os.cpu_count() or 4)
    nc, ns = data.shape
    per_iter_ms: list[float] = []
    final_ll = float("nan")
    for _ in range(repeats):
        with tempfile.TemporaryDirectory(prefix="amica_bench_") as td:
            work = Path(td)
            _write_fdt(data, work / "bench.fdt")
            _write_fortran_param(work, nc, ns, iters, threads, n_models)
            (work / "bench_out").mkdir(exist_ok=True)
            env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
            res = subprocess.run(
                [str(binary), "input.param"],
                cwd=work,
                env=env,
                capture_output=True,
                text=True,
                timeout=_FORTRAN_TIMEOUT,
            )
            if res.returncode != 0:
                raise RuntimeError(
                    f"native fortran run failed (exit {res.returncode}).\n"
                    f"stderr:\n{res.stderr[-2000:]}\nstdout:\n{res.stdout[-2000:]}"
                )
            secs, final_ll = _parse_fortran_out(work / "bench_out" / "out.txt")
            if len(secs) < 2:
                raise RuntimeError(
                    f"native fortran produced <2 timed iterations (got {len(secs)}"
                    "); cannot compute per-iteration timing"
                )
            mean_s = statistics.mean(secs[1:])
            if mean_s == 0.0:
                raise RuntimeError(
                    f"native fortran per-iteration time rounds to 0.00 s for "
                    f"{data.shape[0]}ch x {data.shape[1]} samples -- below amica's "
                    "~10 ms out.txt stamp resolution; use a larger channel/sample "
                    "count (the 70-ch benchmark size is well above the floor)"
                )
            per_iter_ms.append(mean_s * 1000.0)
    return min(per_iter_ms), final_ll


def _fortran_available(binary=None) -> bool:
    binary = Path(binary or _DEFAULT_FORTRAN_BIN)
    return binary.exists() and os.access(binary, os.X_OK)


_BACKENDS = {
    "numpy-cpu-f64": ("numpy", {}),
    "torch-cpu-f64": ("torch", {"device": "cpu", "dtype_str": "f64"}),
    "torch-cpu-f32": ("torch", {"device": "cpu", "dtype_str": "f32"}),
    "torch-mps-f32": ("torch", {"device": "mps", "dtype_str": "f32"}),
    "torch-cuda-f64": ("torch", {"device": "cuda", "dtype_str": "f64"}),
    "torch-cuda-f32": ("torch", {"device": "cuda", "dtype_str": "f32"}),
    "mlx-f32": ("mlx", {}),
    "native-fortran-f64": ("fortran", {}),
}


def _available(
    name: str, n_models: int = 1, share: bool = False, fortran_bin=None
) -> bool:
    kind, kw = _BACKENDS[name]
    if kind == "torch":
        return _torch_available(kw["device"])
    if kind == "mlx":
        # AMICAMLXNG supports single- and multi-model (#81); sharing not yet.
        if share:
            return False
        try:
            import mlx.core as mx

            return mx.default_device().type == mx.DeviceType.gpu
        except Exception:
            return False
    if kind == "fortran":
        # amica15 (MPI+OpenMP) is single-model; sharing is not wired here.
        return not share and _fortran_available(fortran_bin)
    if kind == "numpy":
        return True
    return False


def _run_backend(
    name,
    data,
    iters,
    repeats,
    n_models=1,
    share=False,
    fortran_bin=None,
    fortran_threads=None,
):
    kind, kw = _BACKENDS[name]
    if kind == "torch":
        return _run_torch(
            data, iters=iters, repeats=repeats, n_models=n_models, share=share, **kw
        )
    if kind == "mlx":
        return _run_mlx(data, iters=iters, repeats=repeats, n_models=n_models)
    if kind == "fortran":
        return _run_fortran(
            data,
            iters=iters,
            repeats=repeats,
            n_models=n_models,
            binary=fortran_bin,
            threads=fortran_threads,
        )
    return _run_numpy(
        data, iters=iters, repeats=repeats, n_models=n_models, share=share
    )


def _platform_info() -> dict:
    info = {"machine": platform.machine(), "system": platform.system()}
    info["nproc"] = os.cpu_count()  # context for the CPU-scaling sweep (#86)
    try:
        import torch

        info["torch"] = torch.__version__
        if torch.cuda.is_available():
            info["cuda_device"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    try:
        info["loadavg"] = [round(x, 1) for x in os.getloadavg()]
    except OSError:
        pass
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", help="path to the (70, n_samples) real-EEG .npy")
    ap.add_argument("--channels", default="16,32,48,70")
    ap.add_argument("--samples", type=int, default=30000)
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--backends", default="auto")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--report", nargs="+", help="merge these result JSONs into tables")
    ap.add_argument("--n-models", type=int, default=1, dest="n_models")
    ap.add_argument(
        "--share", action="store_true", help="enable component sharing (n_models>1)"
    )
    ap.add_argument(
        "--fortran-bin",
        default=str(_DEFAULT_FORTRAN_BIN),
        dest="fortran_bin",
        help="native amica binary (default benchmarks/fortran/amica15; build it "
        "with benchmarks/fortran/build_amica.sh)",
    )
    ap.add_argument(
        "--fortran-threads",
        type=int,
        default=None,
        dest="fortran_threads",
        help="OMP threads for the native-fortran backend (default: all cores)",
    )
    args = ap.parse_args()

    if args.report:
        _report(args.report)
        return 0

    if not args.data:
        ap.error("--data is required unless --report is given")

    n_models, share = args.n_models, args.share
    config = f"m{n_models}" + ("+share" if share else "")  # e.g. m1, m2, m2+share

    full = np.load(args.data).astype(np.float64)
    channels = [int(c) for c in args.channels.split(",") if int(c) <= full.shape[0]]
    if args.backends == "auto":
        backends = [
            b for b in _BACKENDS if _available(b, n_models, share, args.fortran_bin)
        ]
    else:
        backends = [
            b
            for b in args.backends.split(",")
            if _available(b, n_models, share, args.fortran_bin)
        ]

    info = _platform_info()
    print(f"platform: {info}")
    print(
        f"data {full.shape} | config {config} | channels {channels} | "
        f"samples {args.samples} | iters {args.iters} x {args.repeats} | {backends}"
    )

    rows = []
    for nc in channels:
        data = np.ascontiguousarray(full[:nc, : args.samples])
        for b in backends:
            try:
                ms, ll = _run_backend(
                    b,
                    data,
                    args.iters,
                    args.repeats,
                    n_models,
                    share,
                    fortran_bin=args.fortran_bin,
                    fortran_threads=args.fortran_threads,
                )
                rows.append(
                    {
                        "config": config,
                        "channels": nc,
                        "backend": b,
                        "ms_per_iter": ms,
                        "final_ll": ll,
                    }
                )
                print(f"  [{config}] {nc:3d}ch {b:16s} {ms:9.2f} ms/it   LL={ll:+.5f}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(
                    f"  [{config}] {nc:3d}ch {b:16s} FAILED: {type(exc).__name__}: {str(exc)[:70]}"
                )
                rows.append(
                    {
                        "config": config,
                        "channels": nc,
                        "backend": b,
                        "error": str(exc)[:120],
                    }
                )

    out = {"platform": info, "config": vars(args), "rows": rows}
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
    return 0


def _report(paths):
    """Merge per-platform JSONs into ms/it + LL tables, one block per config."""
    # merged[config][channels][backend] = row
    merged: dict = {}
    plats = []
    for p in paths:
        with open(p) as f:
            d = json.load(f)
        plats.append(d.get("platform", {}))
        for r in d["rows"]:
            if "error" in r:
                continue
            cfg = r.get("config", "m1")  # rows from before the config axis => m1
            merged.setdefault(cfg, {}).setdefault(r["channels"], {})[r["backend"]] = r

    for cfg in sorted(merged):
        by_ch = merged[cfg]
        backends = sorted({b for ch in by_ch.values() for b in ch})
        chans = sorted(by_ch)

        def table(field, fmt):
            hdr = "channels | " + " | ".join(f"{b:>15}" for b in backends)
            print(hdr)
            print("-" * len(hdr))
            for c in chans:
                cells = [
                    f"{format(by_ch[c][b][field], fmt):>15}"
                    if b in by_ch[c] and by_ch[c][b].get(field) is not None
                    else f"{'-':>15}"
                    for b in backends
                ]
                print(f"{c:8d} | " + " | ".join(cells))

        print(f"\n########## config: {cfg} ##########")
        print("=== performance: ms / iteration ===")
        table("ms_per_iter", ".2f")
        print("=== results: converged log-likelihood ===")
        table("final_ll", "+.5f")
    print(f"\nplatforms: {plats}")


if __name__ == "__main__":
    raise SystemExit(main())
