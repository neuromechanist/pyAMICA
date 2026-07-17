# Native AMICA build (dependency-free, portable)

Builds the AMICA Fortran reference (`amica15.f90` + `funmod2.f90`) into a
self-contained binary with **no MPI runtime, no MKL, and no vendor math library**,
so it can be shipped as a release asset and run on any supported platform (macOS,
Linux x64/arm64, Windows x64/arm64) without the user installing a toolchain.

This is epic #165 phase 1. It extends sccn/amica PR #53's vendor-neutral recipe:

| dependency removed | how | file |
|---|---|---|
| Intel MKL | `-cpp` skips the `#ifdef MKL` branches | (build flags) |
| AMD LibM (`vrda_exp`/`vrda_log`) | libm exp/log loops | `vmath_shim.c` |
| Open MPI runtime | single-rank MPI shim | `mpi_single.f90` + `mpi_single.c` |
| ifort-only `random_seed` | portable full-size seed array (PR #53) | `patch_sources.py` |

LAPACK/BLAS remains, satisfied by Apple's Accelerate framework on macOS (a system
framework, always present) and static reference LAPACK/OpenBLAS on Linux/Windows.

## The single-rank MPI shim

`amica15.f90` is an MPI + OpenMP + LAPACK program, but pyAMICA always runs it as
one process. Every MPI collective it uses (`BCAST`, `REDUCE`, `ALLREDUCE`,
`BARRIER`, `GATHER`, `COMM_SPLIT`, ...) is trivial for a single rank: broadcast is
a no-op, reduce/gather is a copy, barrier is a no-op. `mpi_single.f90` supplies
only the named constants (`use mpi`), with the datatype constants set to their
size in bytes; `mpi_single.c` implements the ~11 subroutines as external
procedures, so the generically-typed `BCAST`/`REDUCE`/`GATHER` calls compile under
`-fallow-argument-mismatch` and single-rank `REDUCE`/`GATHER` reduce to
`memcpy(recv, send, count * datatype)`.

## Build

```bash
bash native/build.sh                     # shim build (gfortran, no MPI) -> native/amica15_shim
MPI_MODE=mpi bash native/build.sh        # reference build (mpif90 + real Open MPI)
```

Toolchain (per sccn/amica PR #53): gfortran (+ a C compiler). No `mpif90` needed
for the shim build. macOS: `brew install gcc`; Debian/Ubuntu:
`sudo apt-get install -y gfortran liblapack-dev libblas-dev`.

## Validation

`validate_shim.sh` proves the shim is **mathematically identical to real Open MPI**,
not just "converges to a plausible LL". It builds the same patched source with the
shim and with `mpif90`, pins the RNG seed, runs one iteration on the bundled sample
EEG, and asserts agreement to ~machine epsilon:

```
LL max|diff| = 8.9e-16,  W/A = 1.7e-18,  all params <= 5e-16   (PASS, tol 1e-12)
```

One iteration is the discriminating window: AMICA's optimizer chaotically amplifies
last-bit differences over iterations (cf. #51/#27), so a genuine shim bug would
show as an O(1) iter-1 difference, while the residual here is compile-driver
roundoff (`gfortran` direct vs the `mpif90` wrapper). The shim binary links no
`libmpi` (verified with `otool -L`/`ldd`).
