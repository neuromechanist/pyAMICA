#!/usr/bin/env python
"""Device/precision benchmark for AMICATorchNG (issue #63).

Times the natural-gradient EM backend across ``(device, dtype)`` combinations on
the real sample EEG, to characterize the GPU fast path. float64 is the
Fortran-parity default and the safe GPU win (~4.5x on an RTX 4090 vs a 16-thread
CPU). float32 is faster still (~5x CPU / ~10-19x CUDA) and now converges on
full-size data across seeds (issue #75 guarded the one float32-only
divide-by-zero); it is ~7-significant-digit, not float64-parity, so use float64
for Fortran-parity runs. MPS has no float64, so it runs float32 only.

Run on a CUDA host (e.g. via ssh) for real GPU numbers:
    uv run python benchmarks/benchmark_gpu.py
    uv run python benchmarks/benchmark_gpu.py --iters 50 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from pamica.torch_impl.core import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parent.parent / "pamica" / "sample_data"


def _combos() -> list[tuple[str, torch.dtype]]:
    combos = [("cpu", torch.float64), ("cpu", torch.float32)]
    if torch.cuda.is_available():
        combos += [("cuda", torch.float64), ("cuda", torch.float32)]
    if torch.backends.mps.is_available():
        combos.append(("mps", torch.float32))  # MPS lacks float64
    return combos


def _time_fit(data, device, dtype, max_iter, seed) -> tuple[float, float]:
    """Return (seconds, final_ll) for one fit. Synchronizes CUDA so the timing
    captures device compute, not just kernel-launch queueing."""
    model = AMICATorchNG(
        n_channels=data.shape[0],
        n_models=1,
        n_mix=3,
        device=device,
        dtype=dtype,
        do_newton=True,
        seed=seed,
    )
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = perf_counter()
    model.fit(data, max_iter=max_iter, verbose=False)
    if device == "cuda":
        torch.cuda.synchronize()
    assert model.final_ll_ is not None
    return perf_counter() - t0, float(model.final_ll_)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--samples", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    params = json.loads((SAMPLE_DIR / "sample_params.json").read_text())
    data = load_eeglab_data(
        str(SAMPLE_DIR / "eeglab_data.fdt"),
        data_dim=params["data_dim"],
        field_dim=params["field_dim"][0],
    ).astype(np.float64)
    if args.samples:
        data = data[:, : args.samples]

    dev_name = ""
    if torch.cuda.is_available():
        dev_name = torch.cuda.get_device_name(0)
    print(
        f"Host {platform.machine()} | torch {torch.__version__} | "
        f"cuda={torch.cuda.is_available()} {dev_name} | "
        f"data {data.shape[0]}x{data.shape[1]} | {args.iters} iters x {args.repeats}"
    )
    print(f"{'device':>6} {'dtype':>8} {'ms/it':>9} {'speedup':>8}  final_ll")

    baseline = None
    for device, dtype in _combos():
        times = []
        ll = float("nan")
        try:
            # Warmup (untimed): the first CUDA call pays context init + kernel
            # compilation, which would otherwise inflate the first timed repeat.
            _time_fit(data, device, dtype, min(5, args.iters), args.seed)
            for r in range(args.repeats):
                t, ll = _time_fit(data, device, dtype, args.iters, args.seed + r)
                times.append(t)
        except Exception as exc:  # noqa: BLE001 - report, keep going to next combo
            print(f"{device:>6} {str(dtype).split('.')[-1]:>8}   FAILED: {exc}")
            continue
        ms = statistics.mean(times) / args.iters * 1000
        if baseline is None:
            baseline = ms  # cpu/float64 reference
        print(
            f"{device:>6} {str(dtype).split('.')[-1]:>8} {ms:9.1f} "
            f"{baseline / ms:7.2f}x  {ll:.5f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
