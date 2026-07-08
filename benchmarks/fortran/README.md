# Native-Fortran backend for the cross-platform benchmark (epic #84, phase #85)

The `native-fortran-f64` backend in `benchmark_dimsweep.py` times **amica15 compiled
from source** so the Fortran reference has an honest per-iteration number to compare
against the torch / mlx / numpy backends.

## Why build from source (not `amica15mac`)

The bundled `pyAMICA/sample_data/amica15mac` is an **x86_64** binary. On Apple Silicon
it only runs under Rosetta 2, so its timing is not representative of native hardware.
The benchmark therefore builds `amica15` from `pyAMICA/{funmod2,amica15}.f90` on the
native x86 Linux/CUDA host and times *that*. (`amica15mac` is still fine for structural
smoke-testing of the adapter on a Mac; it is just not a timing reference.)

## Host setup (Debian/Ubuntu)

amica15.f90 is an **MPI + OpenMP + LAPACK** program, so it needs an MPI Fortran wrapper
(`mpif90`) plus LAPACK/BLAS, not plain `gfortran`:

```bash
sudo apt-get update
sudo apt-get install -y gfortran libopenmpi-dev openmpi-bin liblapack-dev libblas-dev
```

`nvcc` is **not** required: the torch CUDA backends ship their own CUDA runtime, and the
Fortran build is CPU-only (MPI+OpenMP+LAPACK).

## Build

```bash
bash benchmarks/fortran/build_amica.sh        # -> benchmarks/fortran/amica15
```

The script compiles `funmod2.f90` first (amica15 does `use funmod2`), keeps the `.mod`
in `benchmarks/fortran/build/`, and lifts gfortran's 132-column limit
(`-ffree-line-length-none`). Override the compiler with `FC=...` and the source dir with
`AMICA_SRC=...`. The built binary and `build/` are gitignored.

## Run

amica runs as a single MPI rank (OpenMPI singleton -- no `mpirun` needed); OpenMP width is
`OMP_NUM_THREADS`, which the harness sets from `--fortran-threads` (default: all cores).

```bash
# native-fortran only
uv run python benchmarks/benchmark_dimsweep.py \
  --data benchmarks/data/ds002718_sub-002_eeg70.npy \
  --backends native-fortran-f64 --iters 30

# alongside the CUDA backends on the same host
uv run python benchmarks/benchmark_dimsweep.py \
  --data benchmarks/data/ds002718_sub-002_eeg70.npy \
  --backends native-fortran-f64,torch-cuda-f64,torch-cpu-f64 --out cuda.json
```

Use the **same** `ds002718_sub-002_eeg70.npy` as every other host (fetch recipe in
`benchmarks/README_dimsweep.md`) so the comparison is apples-to-apples.

## How the timing works (startup-immune)

Wrapping the whole process in a timer would fold in MPI init, data load, and PCA/sphering.
Instead the adapter parses amica's own per-iteration stamp from `out.txt`:

```
 iter     2 lrate = 0.05 LL = -3.4685 nd = 0.0257, D = ...  (  0.01 s,   0.0 h)
```

The `( <sec> s, ...)` field is that iteration's compute time (printed after init). The
adapter drops iter 1 (first-touch warmup), takes the median of the rest, and reports the
min across `--repeats`. The final `LL` is read from the same lines and is already on the
per-sample scale the torch backends report (no normalization).

If a run exits non-zero the adapter raises (a crashed run must not masquerade as a fast
one). Settings are matched to the other backends: `num_mix=3`, `pdftype=0`, `do_newton` off,
`block_size=512`, and `use_min_dll`/`use_grad_norm` forced off so it runs the full
`--iters` budget.
