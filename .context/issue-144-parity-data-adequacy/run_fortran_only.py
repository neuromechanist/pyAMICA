"""Fortran-only phase for one seed (CPU-bound, safe to run concurrently with
another seed's CUDA phase or other Fortran-only jobs at reduced thread counts).
Writes W + LL to a fixed per-seed directory for a later, separate NG phase to
pick up.

    uv run python run_fortran_only.py <npy> <seed> <max_iter> <threads> <out_dir>
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BIN = REPO / "pyAMICA/sample_data/amica15_linux"
INPUT_PARAM = REPO / "pyAMICA/sample_data/input.param"


def write_fdt(data: np.ndarray, path: Path) -> None:
    path.write_bytes(np.ascontiguousarray(data).astype("<f4").tobytes(order="F"))


def main():
    npy_path = Path(sys.argv[1])
    seed = int(sys.argv[2])
    max_iter = int(sys.argv[3])
    threads = int(sys.argv[4])
    out_dir = Path(sys.argv[5])
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
        else:
            lines.append(ln)
    (out_dir / "input.param").write_text("\n".join(lines) + "\n")

    orig = os.getcwd()
    os.chdir(out_dir)
    env = {**os.environ, "OMP_NUM_THREADS": str(threads)}
    print(f"seed {seed}: starting Fortran ({threads} threads)...", flush=True)
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
        float("nan"),
    )
    (out_dir / "fortran_ll.txt").write_text(str(fort_ll))
    print(f"seed {seed}: Fortran done, LL={fort_ll:.4f}", flush=True)


if __name__ == "__main__":
    main()
