"""Fortran-only phase for one seed (CPU-bound, safe to run concurrently with
another seed's CUDA phase or other Fortran-only jobs at reduced thread counts).
Writes W + LL to a fixed per-seed directory for a later, separate NG phase to
pick up.

    uv run python run_fortran_only.py <npy> <seed> <max_iter> <threads> <out_dir> [do_newton]
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIN = REPO / "pamica/sample_data/amica15_linux"
INPUT_PARAM = REPO / "pamica/sample_data/input.param"


def write_fdt(data: np.ndarray, path: Path) -> None:
    path.write_bytes(np.ascontiguousarray(data).astype("<f4").tobytes(order="F"))


def main():
    npy_path = Path(sys.argv[1])
    seed = int(sys.argv[2])
    max_iter = int(sys.argv[3])
    threads = int(sys.argv[4])
    out_dir = Path(sys.argv[5])
    do_newton = int(sys.argv[6]) if len(sys.argv) > 6 else 1
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(npy_path).astype(np.float64)
    nw, field = data.shape

    fdt_path = out_dir / "data.fdt"
    if not fdt_path.exists():
        write_fdt(data, fdt_path)

    (out_dir / "fortran_output").mkdir(parents=True, exist_ok=True)
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
            lines.append(f"pcakeep {nw}")
        elif ln.startswith("use_min_dll"):
            # Force the full max_iter budget instead of Fortran's own
            # early-stopping (matches benchmark_dimsweep.py's convention for
            # fixed-length, matched runs): otherwise Fortran can converge and
            # stop well short of 2000 while AMICATorchNG (no early-stopping
            # equivalent) keeps optimizing past that point, letting weakly-
            # determined components drift to a different, still-valid optimum
            # -- an asymmetry, not real disagreement.
            lines.append("use_min_dll 0")
        elif ln.startswith("use_grad_norm"):
            lines.append("use_grad_norm 0")
        elif ln.startswith("do_newton"):
            lines.append(f"do_newton {do_newton}")
        else:
            lines.append(ln)
    (out_dir / "input.param").write_text("\n".join(lines) + "\n")

    orig = os.getcwd()
    os.chdir(out_dir)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    print(
        f"seed {seed}: starting Fortran ({threads} threads, do_newton={do_newton})...",
        flush=True,
    )
    try:
        r = subprocess.run(
            [str(BIN), "input.param"], capture_output=True, text=True,
            timeout=3600, env=env,
        )  # fmt: skip
    finally:
        os.chdir(orig)
    if r.returncode != 0:
        print(f"seed {seed}: FAILED: {r.stderr[-500:]}", flush=True)
        sys.exit(1)
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
            f"seed {seed}: WARNING could not parse LL from Fortran stdout", flush=True
        )
        fort_ll = float("nan")
    (out_dir / "fortran_ll.txt").write_text(str(fort_ll))
    print(f"seed {seed}: Fortran done, LL={fort_ll:.4f}", flush=True)


if __name__ == "__main__":
    main()
