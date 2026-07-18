#!/usr/bin/env bash
# Correctness gate for the single-rank MPI shim (epic #165, phase 1).
#
# Proves the shim is mathematically identical to real Open MPI, not merely
# "converges to a plausible LL". It builds the SAME patched source twice -- once
# with the shim (gfortran, no MPI) and once with real Open MPI (mpif90) -- with
# the clock-based RNG seed pinned to a constant, runs ONE iteration of each on the
# bundled sample EEG, and asserts the two agree to ~machine epsilon. One iteration
# is the discriminating window: AMICA's optimizer chaotically amplifies last-bit
# differences over iterations (cf. #51/#27), so a genuine shim bug shows as an
# O(1) iter-1 difference while compile-driver roundoff stays ~1e-15.
#
# Requires BOTH gfortran and mpif90 (Open MPI). This is a dev/CI check; the
# release build (native/build.sh) needs only gfortran.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "$here/.." && pwd)"
sample="$repo_root/pamica/sample_data"
work="$(mktemp -d)"
trap 'rm -rf "$work"' EXIT

command -v gfortran >/dev/null || { echo "ERROR: gfortran required" >&2; exit 1; }
command -v mpif90  >/dev/null || { echo "ERROR: mpif90 (Open MPI) required" >&2; exit 1; }

os="$(uname -s)"
lapack="${LAPACK_LIBS:-$([[ "$os" == Darwin ]] && echo "-framework Accelerate" || echo "-llapack -lblas")}"
fflags=(-O3 -fopenmp -cpp -ffree-line-length-none -std=legacy -fallow-argument-mismatch)

build() {  # build <mode> <fc> -> $work/amica_<mode>
  local mode="$1" fc="$2" d="$work/b_$1"
  mkdir -p "$d"
  cp "$repo_root/pamica"/{funmod2,amica15,amica15_header}.f90 "$d/"
  # --pin-seed drops the clock term from the RNG seed so the shim and mpif90
  # builds seed identically and can be compared bit-for-bit. Scoped to the seed
  # formula inside patch_sources.py (not a loose whole-file regex).
  python3 "$here/patch_sources.py" --pin-seed "$d/amica15.f90" "$d/amica15_header.f90" >/dev/null
  cc -O3 -c "$here/vmath_shim.c" -o "$d/vmath.o"
  local extra=("$d/vmath.o")
  if [[ "$mode" == shim ]]; then
    "$fc" -O3 -cpp -J"$d" -c "$here/mpi_single.f90" -o "$d/mpi_mod.o"
    cc -O3 -c "$here/mpi_single.c" -o "$d/mpi_c.o"
    extra+=("$d/mpi_mod.o" "$d/mpi_c.o")
  fi
  "$fc" "${fflags[@]}" -J"$d" -I"$d" -c "$d/funmod2.f90" -o "$d/funmod2.o"
  # shellcheck disable=SC2086
  "$fc" "${fflags[@]}" -J"$d" -I"$d" "$d/amica15.f90" "$d/funmod2.o" "${extra[@]}" \
    -o "$work/amica_$mode" -static-libgfortran -static-libgcc $lapack
}

run() {  # run <mode> -> writes $work/o_<mode>
  local mode="$1" d="$work/o_$1"
  mkdir -p "$d"
  ( cd "$work" && cp "$sample/eeglab_data.fdt" .
    sed -e 's#^files .*#files ./eeglab_data.fdt#' -e 's#^max_iter .*#max_iter 1#' \
        -e 's#^writestep .*#writestep 1#' -e 's#^do_newton .*#do_newton 0#' \
        -e "s#^outdir .*#outdir ./o_$mode/#" "$sample/input.param" > "p_$mode.param"
    OMP_NUM_THREADS=1 "$work/amica_$mode" "p_$mode.param" >/dev/null 2>&1 )
}

echo "building shim (gfortran) and reference (mpif90)..."
build shim gfortran
build mpi mpif90
echo "running one iteration of each on real sample EEG..."
run shim
run mpi

python3 - "$work" <<'PY'
import sys, numpy as np
w = sys.argv[1]
tol = 1e-12
worst = 0.0
for f in ("LL", "W", "A", "mu", "sbeta", "alpha", "c", "rho", "gm"):
    a = np.fromfile(f"{w}/o_shim/{f}"); b = np.fromfile(f"{w}/o_mpi/{f}")
    n = min(len(a), len(b))
    d = float(np.abs(a[:n] - b[:n]).max()) if n else 0.0
    worst = max(worst, d)
    print(f"  {f:6s} max|diff| = {d:.3e}")
print(f"worst = {worst:.3e}  (tol {tol:.0e})")
if worst > tol:
    sys.exit("FAIL: shim diverges from real MPI beyond machine epsilon -- shim bug")
print("PASS: single-rank MPI shim is bit-equivalent to Open MPI at iteration 1")
PY
