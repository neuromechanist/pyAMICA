#!/usr/bin/env bash
# Build a native amica15 for the cross-platform benchmark (epic #84, phase #85).
#
# amica15.f90 is an MPI + OpenMP + LAPACK program, so this needs an MPI Fortran
# wrapper (mpif90) plus LAPACK/BLAS -- NOT plain gfortran. The resulting binary
# runs fine as a single MPI rank (OpenMPI singleton); OpenMP width is set at run
# time via OMP_NUM_THREADS. We build from source (not the bundled amica15mac,
# which is x86-under-Rosetta on Apple Silicon) so the reference timing is honest
# on the native x86 host.
#
# Usage:
#   bash benchmarks/fortran/build_amica.sh
#   FC=mpiifort bash benchmarks/fortran/build_amica.sh      # override compiler
#   AMICA_SRC=/path/to/src bash benchmarks/fortran/build_amica.sh
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/../.." && pwd)"
src_dir="${AMICA_SRC:-$repo_root/pyAMICA}"
fc="${FC:-mpif90}"
build_dir="$here/build"
out="$here/amica15"

echo "amica native build"
echo "  compiler : $fc"
echo "  sources  : $src_dir/{funmod2,amica15}.f90"
echo "  output   : $out"

if ! command -v "$fc" >/dev/null 2>&1; then
  cat >&2 <<EOF

ERROR: '$fc' not found. amica15.f90 is an MPI+OpenMP+LAPACK program and needs an
MPI Fortran wrapper plus LAPACK/BLAS. On Debian/Ubuntu:

  sudo apt-get update
  sudo apt-get install -y gfortran libopenmpi-dev openmpi-bin liblapack-dev libblas-dev

Then re-run this script (override the compiler with FC=... if it is not mpif90).
EOF
  exit 1
fi

for f in funmod2.f90 amica15.f90; do
  if [[ ! -f "$src_dir/$f" ]]; then
    echo "ERROR: source not found: $src_dir/$f" >&2
    exit 1
  fi
done

mkdir -p "$build_dir"
# funmod2 first (amica15 does `use funmod2`); -J puts the .mod in build/ so the
# source tree stays clean; -ffree-line-length-none lifts gfortran's 132-col cap.
set -x
"$fc" -O3 -fopenmp -ffree-line-length-none -J"$build_dir" \
  "$src_dir/funmod2.f90" "$src_dir/amica15.f90" \
  -o "$out" -llapack -lblas
set +x

echo
echo "built: $out"
echo "quick check: in a dir holding input.param + its .fdt, run"
echo "  OMP_NUM_THREADS=4 $out input.param"
echo "then benchmark with:"
echo "  uv run python benchmarks/benchmark_dimsweep.py --data DATA.npy \\"
echo "      --backends native-fortran-f64 --iters 30"
