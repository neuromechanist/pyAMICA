#!/bin/bash
# 5-seed Fortran-vs-NG comparison with Newton disabled (do_newton=0), forced
# full 2000-iteration budget (use_min_dll/use_grad_norm off), on the full
# ds002718 sub-002 recording (70ch, 747750 frames, k=152.6).
#
# Pipelined: each seed's Fortran phase (CPU, 24 threads) runs strictly
# sequentially (splitting threads across concurrent Fortran jobs was
# catastrophically slow in an earlier attempt), but as soon as a seed's
# Fortran finishes, its NG/CUDA phase (GPU) is launched in the background and
# the NEXT seed's Fortran starts immediately -- CPU and GPU work overlap
# since they don't compete for the same resource.
set -e
cd ~/pyAMICA-issue144
NPY=benchmarks/data/ds002718_sub-002_eeg70_full.npy
SCRIPTS=.context/issue-144-parity-data-adequacy

for s in 201 202 203 204 205; do
  echo "=== seed $s: Fortran (24 threads, do_newton=0) ==="
  uv run python -u $SCRIPTS/run_fortran_only.py "$NPY" "$s" 2000 24 /tmp/newton0_seed$s 0
  echo "=== seed $s: launching NG/CUDA in background, moving to next seed's Fortran ==="
  uv run python -u $SCRIPTS/run_ng_only.py "$NPY" "$s" 2000 /tmp/newton0_seed$s 0 \
    > /tmp/newton0_seed${s}_ng.log 2>&1 &
done

echo "=== all Fortran phases launched/done, waiting on remaining background NG jobs ==="
wait
echo "=== ALL DONE ==="
