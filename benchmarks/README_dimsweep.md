# Cross-platform dimension-sweep benchmark (issue #77, epic #74 Phase B)

`benchmark_dimsweep.py` measures **both results (converged log-likelihood) and
performance (ms/iteration)** for every AMICA backend the host supports, sweeping
the channel count on real 70-channel EEG, to answer where an Apple/NVIDIA GPU
actually beats the CPU. Backends: `numpy-cpu-f64`, `torch-cpu-f64/f32`,
`torch-mps-f32`, `torch-cuda-f64/f32`, `mlx-f32` (the MLX backend supports single-
and multi-model but has no component sharing yet, so it is excluded only from the
`--share` configs), and `native-fortran-f64` (the Fortran reference compiled from
source, validated single-model in Phase 1 with component sharing off -- see
`benchmarks/fortran/README.md` to build it).

**`native-fortran-f64` result caveat:** every other backend fixes `seed=42`, so its
`final_ll` is reproducible and directly comparable across backends and repeats. amica seeds
its random init from the wall clock (non-reproducible run-to-run), so the native-Fortran
`final_ll` is drawn from a different basin each invocation and can differ from the others by
more than the fixed-seed backends differ among themselves. Treat that column as a sanity
check (same ballpark), not a fixed-seed parity number; the `ms/iter` timing is unaffected.

## Data (real, not committed)

Real 70-channel EEG from OpenNeuro **ds002718** (Wakeman-Henson faces), subject
sub-002. The data is not committed (`benchmarks/data/` is gitignored); fetch and
extract it once:

```bash
# 1. download one subject's EEGLAB .set (public, no credentials; ~224 MB)
aws s3 cp --no-sign-request \
  s3://openneuro.org/ds002718/sub-002/eeg/sub-002_task-FaceRecognition_eeg.set \
  /tmp/ds002718_sub-002.set

# 2. extract the 70 EEG channels to a (70, n_samples) float64 .npy (needs mne)
uv pip install mne
uv run python - <<'PY'
import mne, numpy as np
raw = mne.io.read_raw_eeglab("/tmp/ds002718_sub-002.set", preload=True, verbose="ERROR")
data = raw.get_data(picks=raw.ch_names[:70]) * 1e6   # first 70 are EEG; V -> uV
np.save("benchmarks/data/ds002718_sub-002_eeg70.npy", data[:, :60000].astype(np.float64))
PY
```

(NEMAR mirrors the same dataset at data.nemar.org / ww2.nemar.org/dataset/ds002718.)

## Run

```bash
# Local (Apple Silicon: cpu / mps / mlx auto-detected)
uv run python benchmarks/benchmark_dimsweep.py \
  --data benchmarks/data/ds002718_sub-002_eeg70.npy --out mac.json

# A CUDA host (skip the CPU backends if its CPU is busy)
uv run python benchmarks/benchmark_dimsweep.py \
  --data benchmarks/data/ds002718_sub-002_eeg70.npy \
  --backends torch-cuda-f64,torch-cuda-f32 --out cuda.json

# Multi-model + component sharing (MLX runs multi-model; auto-excluded only from --share)
uv run python benchmarks/benchmark_dimsweep.py --data DATA --n-models 2 --out mac_m2.json
uv run python benchmarks/benchmark_dimsweep.py --data DATA --n-models 2 --share --out mac_m2share.json

# CPU core-count scaling sweep (#86): run the CPU backends at each thread count
# (GPU backends run once). torch-cpu -> set_num_threads, numpy -> threadpoolctl,
# native-fortran -> OMP_NUM_THREADS. Best on a many-core host (e.g. hallu, 32 cores).
uv run python benchmarks/benchmark_dimsweep.py \
  --data benchmarks/data/ds002718_sub-002_eeg70.npy \
  --backends torch-cpu-f64,numpy-cpu-f64,native-fortran-f64 \
  --threads 4,8,16,32 --out scaling.json

# Merge per-platform JSONs into ms/it + LL tables, one block per config; --threads
# rows add a "CPU scaling" block (threads x backend) with a GPU reference line.
uv run python benchmarks/benchmark_dimsweep.py --report mac.json cuda.json scaling.json ...
```

Findings live in `.context/issue-77/benchmark_findings.md`.
