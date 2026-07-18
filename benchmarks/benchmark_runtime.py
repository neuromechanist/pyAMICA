#!/usr/bin/env python
"""Benchmark AMICATorchNG runtime against the Fortran reference binary (issue #59).

Verifies the ".context/plan.md" success criterion "runtime within 2-3x of
Fortran" by timing both implementations on the real sample EEG (32 channels x
30504 frames), with a *matched* algorithm configuration (same iteration budget,
lrate, Newton schedule, mixture count, and PDF type).

Metrics
-------
For each (backend, thread-count, iteration-count) cell we record wall-clock time
over N repeats and the final log-likelihood (a sanity check that both are doing
equivalent work). Two headline numbers are derived:

* **total wall-clock** at a reference iteration count -- the end-to-end runtime a
  user actually waits for (includes fixed overhead: data load, sphering, and for
  Fortran the process spawn + output I/O).
* **marginal per-iteration time** -- the slope of total-time vs iteration-count
  across the grid. This cancels the one-time fixed overhead, isolating the
  steady-state EM cost, which is the fairest cross-implementation comparison.

Honest-measurement caveats (printed in the report banner)
--------------------------------------------------------
* The bundled binary ``amica15mac`` is **x86_64**. On Apple Silicon (arm64) it
  runs under Rosetta 2 emulation while PyTorch runs native, so a local ratio
  flatters NG. For a representative number, run this on a native x86_64/Linux
  host (CI or a compute node) with a native Fortran build.
* Fortran uses OpenMP; PyTorch-CPU uses its own intra-op threads. We control both
  (``OMP_NUM_THREADS`` / ``torch.set_num_threads``) and report each thread count.
* NG computes in float64 for Fortran parity; MPS cannot represent float64, so the
  CPU path is used on Apple Silicon. CUDA is used automatically when present.

Usage
-----
    uv run python benchmarks/benchmark_runtime.py
    uv run python benchmarks/benchmark_runtime.py --iters 50 100 200 --repeats 5
    uv run python benchmarks/benchmark_runtime.py --threads 1 --skip-fortran
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_DIR = REPO_ROOT / "pamica" / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
PARAM_TEMPLATE = SAMPLE_DIR / "input.param"
FORTRAN_BINARY = SAMPLE_DIR / "amica15mac"
PARAMS_JSON = SAMPLE_DIR / "sample_params.json"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# params.json keys AMICATorchNG accepts as constructor kwargs; anything else is
# handled explicitly (below) or does not apply to the backend.
from pamica import AMICA  # noqa: E402
from pamica.torch_impl import AMICATorchNG  # noqa: E402
from pamica.torch_impl.utils import load_eeglab_data  # noqa: E402

import inspect  # noqa: E402

_NG_PARAMS = set(inspect.signature(AMICATorchNG).parameters) - {"n_channels"}
# Passed explicitly to AMICA()/fit() rather than forwarded as a constructor kwarg.
_EXPLICIT = {"lrate", "do_mean", "do_sphere", "do_newton", "seed", "device"}


def set_all_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data_and_params() -> tuple[np.ndarray, dict]:
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Sample data not found at {DATA_FILE}")
    with open(PARAMS_JSON) as f:
        params = json.load(f)
    data = load_eeglab_data(
        str(DATA_FILE),
        data_dim=params["data_dim"],
        field_dim=params["field_dim"][0],
        dtype=np.float32,
    )
    return data, params


def ng_kwargs_from_params(params: dict) -> dict:
    """Map sample_params.json onto AMICATorchNG constructor kwargs (matches the
    validated config in validate_implementations.py)."""
    kwargs = {k: v for k, v in params.items() if k in _NG_PARAMS and k not in _EXPLICIT}
    if "max_decs" in params:
        kwargs["maxdecs"] = params["max_decs"]
    return kwargs


def select_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    # NG runs float64; MPS cannot, so Apple Silicon uses CPU.
    return "cpu"


def time_ng(
    data: np.ndarray,
    params: dict,
    ng_kwargs: dict,
    max_iter: int,
    seed: int,
    device: str,
    n_threads: int,
) -> dict:
    """Run one NG fit and return {time, iters, final_ll}."""
    torch.set_num_threads(n_threads)
    set_all_seeds(seed)
    model = AMICA(
        n_models=1, n_mix=params.get("num_mix", 3), verbose=False, device=device
    )
    t0 = perf_counter()
    model.fit(
        data,
        max_iter=max_iter,
        lrate=params.get("lrate", 0.05),
        do_mean=params.get("do_mean", True),
        do_sphere=params.get("do_sphere", True),
        do_newton=params.get("do_newton", True),
        seed=seed,
        **ng_kwargs,
    )
    elapsed = perf_counter() - t0
    final_ll = model.final_ll_
    return {
        "time": elapsed,
        "iters": len(model.ll_history_),
        "final_ll": float(final_ll) if final_ll is not None else float("nan"),
    }


def _write_fortran_param(work_dir: Path, max_iter: int, n_threads: int) -> None:
    """Copy input.param into work_dir, overriding only the benchmark knobs so the
    algorithm config still matches sample_params.json / the NG run."""
    with open(PARAM_TEMPLATE) as f:
        lines = f.readlines()
    with open(work_dir / "input.param", "w") as f:
        for line in lines:
            key = line.split()[0] if line.strip() else ""
            if key == "files":
                f.write("files ./eeglab_data.fdt\n")
            elif key == "outdir":
                f.write("outdir ./bench_out/\n")
            elif key == "max_iter":
                f.write(f"max_iter {max_iter}\n")
            elif key == "max_threads":
                f.write(f"max_threads {n_threads}\n")
            else:
                f.write(line)


def _scrape_fortran_ll(work_dir: Path) -> float | None:
    """Best-effort: read the last ' iter ... LL = X' line from the timestamped
    out.txt amica writes under outdir. Timing does not depend on this."""
    candidates = list((work_dir / "bench_out").glob("**/*"))
    for out_file in candidates:
        if out_file.is_file() and out_file.name.endswith(".out"):
            candidates.insert(0, out_file)
    last_ll = None
    for out_file in candidates:
        if not out_file.is_file():
            continue
        try:
            text = out_file.read_text(errors="ignore")
        except OSError:
            continue
        for line in text.splitlines():
            if line.strip().startswith("iter") and "LL =" in line:
                parts = line.split()
                idx = parts.index("LL")
                if idx + 2 < len(parts):
                    try:
                        last_ll = float(parts[idx + 2])
                    except ValueError:
                        pass
    return last_ll


def time_fortran(work_dir: Path, max_iter: int, n_threads: int, timeout: int) -> dict:
    """Run one Fortran fit and return {time, iters, final_ll}. Raises on failure
    (no silent fallback -- a crashed run must not masquerade as a fast one)."""
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(DATA_FILE, work_dir / "eeglab_data.fdt")
    _write_fortran_param(work_dir, max_iter, n_threads)
    (work_dir / "bench_out").mkdir(exist_ok=True)

    env = {**os.environ, "OMP_NUM_THREADS": str(n_threads)}
    t0 = perf_counter()
    result = subprocess.run(
        [str(FORTRAN_BINARY), "input.param"],
        cwd=work_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    elapsed = perf_counter() - t0
    if result.returncode != 0:
        raise RuntimeError(
            f"Fortran run failed (exit {result.returncode}).\n"
            f"stderr:\n{result.stderr[-2000:]}\nstdout:\n{result.stdout[-2000:]}"
        )
    return {
        "time": elapsed,
        "iters": max_iter,  # amica runs the full budget (min_dll is ~1e-9)
        "final_ll": _scrape_fortran_ll(work_dir),
    }


def _summarize(runs: list[dict]) -> dict:
    times = [r["time"] for r in runs]
    return {
        "time_mean": statistics.mean(times),
        "time_std": statistics.stdev(times) if len(times) > 1 else 0.0,
        "iters": runs[0]["iters"],
        "final_ll": runs[-1]["final_ll"],
        "per_iter": statistics.mean(times) / runs[0]["iters"],
        "n": len(times),
    }


def _slope_per_iter(cells: dict[int, dict]) -> float | None:
    """Marginal per-iteration seconds = slope of mean-time vs iters across the
    grid (cancels fixed overhead). Uses the widest iteration span available."""
    iters = sorted(cells)
    if len(iters) < 2:
        return None
    lo, hi = iters[0], iters[-1]
    return (cells[hi]["time_mean"] - cells[lo]["time_mean"]) / (hi - lo)


def _binary_arch() -> str:
    try:
        out = subprocess.run(
            ["file", str(FORTRAN_BINARY)], capture_output=True, text=True, timeout=10
        ).stdout
        if "x86_64" in out:
            return "x86_64"
        if "arm64" in out:
            return "arm64"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def environment_banner(device: str) -> dict:
    machine = platform.machine()
    binary_arch = _binary_arch()
    emulated = machine == "arm64" and binary_arch == "x86_64"
    return {
        "host_machine": machine,
        "host_processor": platform.processor() or platform.platform(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "ng_device": device,
        "fortran_binary_arch": binary_arch,
        "fortran_emulated_rosetta": emulated,
    }


def run_benchmark(args) -> dict:
    data, params = load_data_and_params()
    ng_kwargs = ng_kwargs_from_params(params)
    device = args.device or select_device()
    env = environment_banner(device)

    results: dict = {
        "environment": env,
        "config": {
            "data_shape": list(data.shape),
            "iters": args.iters,
            "repeats": args.repeats,
            "threads": args.threads,
            "seed": args.seed,
            "num_mix": params.get("num_mix", 3),
            "do_newton": params.get("do_newton", True),
            "lrate": params.get("lrate", 0.05),
        },
        "ng": {},
        "fortran": {},
    }

    for n_threads in args.threads:
        ng_cells: dict[int, dict] = {}
        for it in args.iters:
            runs = [
                time_ng(data, params, ng_kwargs, it, args.seed, device, n_threads)
                for _ in range(args.repeats)
            ]
            ng_cells[it] = _summarize(runs)
            s = ng_cells[it]
            print(
                f"[NG   t={n_threads} iters={it:>4}] "
                f"{s['time_mean']:.3f}s +/- {s['time_std']:.3f}  "
                f"({s['per_iter'] * 1000:.1f} ms/it, LL={s['final_ll']:.4f})"
            )
        results["ng"][str(n_threads)] = {
            "cells": ng_cells,
            "slope_per_iter": _slope_per_iter(ng_cells),
        }

        if not args.skip_fortran:
            if not FORTRAN_BINARY.exists():
                raise FileNotFoundError(f"Fortran binary not found at {FORTRAN_BINARY}")
            f_cells: dict[int, dict] = {}
            for it in args.iters:
                runs = []
                for r in range(args.repeats):
                    work_dir = RESULTS_DIR / f"fortran_run_t{n_threads}_i{it}_r{r}"
                    runs.append(time_fortran(work_dir, it, n_threads, args.timeout))
                    shutil.rmtree(work_dir, ignore_errors=True)
                f_cells[it] = _summarize(runs)
                s = f_cells[it]
                ll = f"{s['final_ll']:.4f}" if s["final_ll"] is not None else "n/a"
                print(
                    f"[FORT t={n_threads} iters={it:>4}] "
                    f"{s['time_mean']:.3f}s +/- {s['time_std']:.3f}  "
                    f"({s['per_iter'] * 1000:.1f} ms/it, LL={ll})"
                )
            results["fortran"][str(n_threads)] = {
                "cells": f_cells,
                "slope_per_iter": _slope_per_iter(f_cells),
            }

    return results


def render_markdown(results: dict) -> str:
    env = results["environment"]
    cfg = results["config"]
    lines = [
        "## Benchmark: AMICATorchNG vs Fortran (issue #59)",
        "",
        f"- Data: {cfg['data_shape'][0]} ch x {cfg['data_shape'][1]} frames, "
        f"num_mix={cfg['num_mix']}, do_newton={cfg['do_newton']}, lrate={cfg['lrate']}",
        f"- Host: {env['host_machine']} ({env['cpu_count']} CPUs), "
        f"torch {env['torch']}, NG device `{env['ng_device']}`",
        f"- Fortran binary arch: {env['fortran_binary_arch']}"
        + (
            " -- **runs under Rosetta 2 emulation here; ratio flatters NG, "
            "rerun on native x86_64/Linux for a representative number**"
            if env["fortran_emulated_rosetta"]
            else ""
        ),
        f"- Repeats: {cfg['repeats']}, seed {cfg['seed']}",
        "",
        "| threads | iters | NG (s) | Fortran (s) | NG/Fortran | NG ms/it | Fortran ms/it |",
        "|--------:|------:|-------:|------------:|-----------:|---------:|--------------:|",
    ]
    for n_threads in map(str, cfg["threads"]):
        ng = results["ng"].get(n_threads, {}).get("cells", {})
        fort = results["fortran"].get(n_threads, {}).get("cells", {})
        for it in cfg["iters"]:
            ng_c = ng.get(it)
            f_c = fort.get(it)
            if not ng_c:
                continue
            ng_t = f"{ng_c['time_mean']:.3f}"
            ng_pi = f"{ng_c['per_iter'] * 1000:.1f}"
            if f_c:
                f_t = f"{f_c['time_mean']:.3f}"
                f_pi = f"{f_c['per_iter'] * 1000:.1f}"
                ratio = f"{ng_c['time_mean'] / f_c['time_mean']:.2f}x"
            else:
                f_t = f_pi = ratio = "-"
            lines.append(
                f"| {n_threads} | {it} | {ng_t} | {f_t} | {ratio} | {ng_pi} | {f_pi} |"
            )
    # Marginal per-iteration (fixed-overhead-free) summary.
    lines += ["", "**Marginal per-iteration (slope, fixed overhead removed):**"]
    for n_threads in map(str, cfg["threads"]):
        ng_slope = results["ng"].get(n_threads, {}).get("slope_per_iter")
        f_slope = results["fortran"].get(n_threads, {}).get("slope_per_iter")
        if ng_slope is None:
            continue
        parts = [f"threads={n_threads}: NG {ng_slope * 1000:.1f} ms/it"]
        if f_slope:
            parts.append(f"Fortran {f_slope * 1000:.1f} ms/it")
            parts.append(f"ratio {ng_slope / f_slope:.2f}x")
        lines.append("- " + ", ".join(parts))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iters", type=int, nargs="+", default=[50, 100])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, min(8, os.cpu_count() or 1)],
        help="Thread counts to sweep (OMP for Fortran, torch intra-op for NG)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default=None, help="NG device (default: auto)"
    )
    parser.add_argument("--skip-fortran", action="store_true")
    parser.add_argument(
        "--timeout", type=int, default=1800, help="Per-run Fortran timeout (s)"
    )
    args = parser.parse_args()
    # De-dupe threads while preserving order (e.g. cpu_count==1 -> [1,1]).
    args.threads = list(dict.fromkeys(args.threads))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    print(
        f"Benchmark: iters={args.iters}, repeats={args.repeats}, threads={args.threads}"
    )

    results = run_benchmark(args)
    md = render_markdown(results)
    print("\n" + md + "\n")

    (RESULTS_DIR / "benchmark_results.json").write_text(json.dumps(results, indent=2))
    (RESULTS_DIR / "benchmark_report.md").write_text(md)
    print(f"Saved: {RESULTS_DIR / 'benchmark_results.json'}")
    print(f"Saved: {RESULTS_DIR / 'benchmark_report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
