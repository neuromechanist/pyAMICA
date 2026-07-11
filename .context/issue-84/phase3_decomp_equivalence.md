# Phase 3 (#87): full-decomposition cross-backend equivalence

Runs every backend to a full decomposition (2000 iters) on real EEG, then compares the
recovered independent components across backends. Answers the question the whole epic is
built on: **is pyAMICA (and the native-Fortran build) a drop-in replacement for EEGLAB AMICA
-- do all backends recover the same sources?**

`benchmarks/benchmark_decompose.py` fits each backend, saves total wall-clock + final LL +
the unmixing W and mixing A, and `--compare` Hungarian-matches W across backends (rows
normalized, `|W1 @ W2.T|`, so it is invariant to the ICA sign + permutation ambiguity). Two
figures: a square cross-backend correlation matrix and an MNE IC-scalp-map grid ordered by
back-projected variance (EEGLAB convention).

## Data (the k factor matters)
Full ds002718 sub-002 recording is 747,750 frames (~50 min @ 250 Hz). Phase 3 uses 147,000
frames = **k = 30 at 70ch** (frames / ch^2 = the EEGLAB data-adequacy rule, minimum ~20-30).
numpy is excluded (too slow, not a recommended backend).

## Result 1: pyAMICA is device- and precision-invariant (the headline)

Cross-backend mean Hungarian-matched |corr| @ 70ch, 2000 iters (see
`benchmarks/figures/phase3_equivalence_matrix_70ch.png`):

**Every torch/MLX backend is identical to each other at 1.000** -- torch-cpu (both machines),
torch-cuda, torch-mps, and MLX, across **both f32 and f64** (8 backend/precision/device
combinations). Same decomposition on any device at any precision. This is the definitive
"f32 == f64" and "GPU == CPU" result: pyAMICA gives the same ICs regardless of where or how it
runs.

The two native-Fortran runs (Mac arm64 + Linux x86_64) agree with each other at 0.972 and with
the torch/MLX cluster at ~0.90 -- a different local optimum from a different (clock-seeded)
init, on the weakly-determined components only (see Result 2). Even Fortran-vs-Fortran is not
1.000, confirming it is an init effect, not a platform/backend defect.

## Result 2: equivalence tracks the k factor (data adequacy)

Holding frames = 147k and sweeping channels sweeps k (mlx-f32 vs native-fortran-f64):

| channels | k = frames/ch^2 | mean matched \|corr\| | components > 0.95 |
|---------:|----------------:|---------------------:|------------------:|
|       16 |             574 |            **0.997** |         **16/16** |
|       32 |             144 |                0.974 |             27/32 |
|       48 |              64 |                0.954 |             34/48 |
|       70 |              30 |                0.898 |             20/70 |

When the decomposition is well-determined (high k) **every backend, including Fortran, recovers
identical ICs** (0.997 at k=574). At k=30 -- the rule-of-thumb minimum -- only the strongest
~20/70 components are reproducible; the rest are under-determined and different inits settle
into different, equally valid local optima (AMICA is non-convex). This validates both backend
correctness *and* the data-adequacy rule. Pushing 70ch to k=152 with the full recording (#90,
`.context/issue-90/`) confirms the trend saturates: equivalence climbs to ~0.98 at k~=60 and
plateaus (the two f64 implementations reach 0.995), it does not reach 1.0 because of float32
precision + non-convex local-optimum spread on the weakest components.

## Result 3: total decompose time (2000 iters, 147k frames, 70ch)

| backend | machine | total time |
|---|---|---|
| native-fortran-f64 | hallu (Linux, 32 cores) | **~4.4 min** |
| MLX-f32 | Mac (Apple GPU) | ~5.1 min |
| torch-cuda-f32 / f64 | hallu (RTX 4090) | ~5.6 / ~6.1 min |
| native-fortran-f64 | Mac (arm64) | ~10.1 min |
| torch-cpu-f32 / f64 | Mac (14 cores) | ~27 / ~31 min |
| torch-cpu-f32 / f64 | hallu (32 cores, default threads) | ~29 / **~67 min** |
| torch-mps-f32 | Mac | ~35 min |

Native Fortran on the 32-core Linux host is the fastest end-to-end (~4.4 min), ~2x its own Mac
time; MLX and CUDA follow at ~5-6 min. The GPU finally runs sustained here (a full
decomposition, not the per-iteration blips of Phase 2). Two cautions confirmed from Phase 2:
torch-cpu-f64 on hallu at the default (all-32-core) thread count is the worst CPU number
(~67 min, oversubscription), and torch-MPS is the worst GPU path (~35 min) -- use MLX, never
MPS, on Apple.

## Figures
- `benchmarks/figures/phase3_equivalence_matrix_70ch.png` -- square cross-backend IC-equivalence
  matrix (the 8-backend torch/MLX 1.000 block + the 2 Fortran rows), adaptive color scale so the
  ~0.90 vs 1.00 differences show. `..._16ch.png` is the high-k case where all backends match.
- `benchmarks/figures/phase3_ic_topomaps_70ch.png` -- IC scalp maps, variance-ordered (IC1 =
  highest variance, EEGLAB convention). Each map is the **de-sphered** sensor-space projection
  (the saved A is whitened-space loadings; the harness recomputes the symmetric-ZCA sphere from
  the data and de-spheres, so these are true EEGLAB-style scalp maps), Hungarian-matched and
  sign-aligned; columns are visibly identical down the rows for the well-determined components.

## Caveats / follow-ups
- Electrode positions from the BIDS electrodes.tsv may need a rotation to MNE's head frame for
  the absolute nose-up orientation (does NOT affect equivalence -- every backend shares the
  montage). Tracked informally; refine later.
- Channel subsets use the first N electrodes (spatially clustered) -> #91 (use distributed
  subsets so reduced-channel maps are whole-head).
- Data-size/k sweep at 70ch (full recording, k=152) -> #90 DONE
  (`.context/issue-90/ksweep_findings.md`): equivalence saturates at ~0.98 past k~=60.
- Output-format/convention parity for a true EEGLAB drop-in -> #92.
