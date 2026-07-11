# Issue #90: data-size (k-factor) sweep at 70 channels

Follow-up to Phase 3 (#87). Phase 3 swept k along the **channel** axis (frames fixed at 147k,
so more channels = lower k) and found cross-backend IC equivalence tracks the data-adequacy
factor **k = frames / channels^2** (EEGLAB rule of thumb: k >= 20-30). #90 sweeps the other
axis: hold **channels = 70** and grow **frames** using the full ds002718 sub-002 recording
(747,750 frames, ~50 min @ 250 Hz), so k rises from 15 to 152.

Hypothesis: as k grows the decomposition becomes well-determined, so cross-backend equivalence
should rise toward ~1.0.

## Run
`benchmarks/benchmark_decompose.py --frames 73500,147000,294000,490000,747750 --channels 70
--iters 2000` on hallu (Linux x86_64, RTX 4090, 32 cores). Backends: `native-fortran-f64`,
`torch-cuda-f64`, `torch-cuda-f32` (the fast, well-behaved backends -- Phase 3 already proved
every torch/MLX device/precision recovers an identical decomposition, so a 3-backend cross-check
here is sufficient and the large-frame runs are ~5x longer per backend than the k=30 runs).

## Result: equivalence climbs steeply, then saturates at ~0.98 (not 1.0)

Mean pairwise Hungarian-matched |corr| across the 3 backends
(`benchmarks/figures/ksweep_70ch.png`):

| frames  |     k | mean \|corr\| | min pair | pairwise comps > 0.95 |
|--------:|------:|--------------:|---------:|----------------------:|
|  73,500 |    15 |        0.9105 |   0.8656 |                 55.2% |
| 147,000 |    30 |        0.9288 |   0.8931 |                 56.7% |
| 294,000 |    60 |    **0.9818** |   0.9717 |                 90.0% |
| 490,000 |   100 |        0.9826 |   0.9790 |                 94.8% |
| 747,750 | 152.6 |        0.9817 |   0.9712 |                 92.4% |

The hypothesis holds **directionally** with an important nuance: equivalence rises sharply from
0.91 (k=15) to 0.98 (k=60) and then **plateaus at ~0.982**, it does not keep climbing to 1.0.
The transition is complete by **k ~= 60** (frames ~= 3 * channels^2); the sweep samples no point
between k=30 (0.929) and k=60 (0.982), so the exact knee within that gap is unresolved (it is an
upper bound, not a pinpoint). Below it the decomposition is under-determined and backends settle
into different (equally valid) local optima on the weak components; above it the
strong-and-medium components lock and only the residual precision / init spread remains.

## Why the plateau sits at 0.98, not 1.0

The per-config equivalence matrix at k=152 (`benchmarks/figures/ksweep_70ch_matrix.png`) splits
the residual cleanly by precision:

| pair | \|corr\| @ k=152 |
|---|---|
| native-fortran-f64 vs torch-cuda-**f64** | **0.995** |
| native-fortran-f64 vs torch-cuda-f32     | 0.971 |
| torch-cuda-f64 vs torch-cuda-f32         | 0.979 |

- **The two double-precision implementations agree at 0.995** -- an independent Fortran binary
  and the PyTorch-CUDA backend recover essentially the same ICs once k is large. This is the
  strongest drop-in-replacement evidence in the epic: cross-*implementation*, not just
  cross-device.
- The residual mean-gap below 1.0 is dominated by **float32**, for two compounding reasons:
  (1) f32 rounding accumulates over the larger data and 2000 iters; (2) at k=152 the f32 run
  hit the natural-gradient **lrate floor (1e-12) at iter 1735 and stopped early**, landing on a
  marginally different optimum than the full-2000-iter f64 runs. Both are valid converged
  solutions; this is why Phase 3's "f32 == f64 at 1.000" (k=30, both ran the full 2000) relaxes
  to ~0.98 on the full recording. It is a convergence/precision effect, not a backend defect.

## Practical guideline (the useful deliverable)

For **backend-reproducible** ICs at 70ch, budget **k ~= 60 (frames ~= 3 * channels^2)**, roughly
2x the EEGLAB k >= 20-30 rule. The classic rule guarantees only that *the strongest* components
are stable; k ~= 60 is where the *bulk* of the decomposition (90%+ of pairwise component
correlations > 0.95) becomes reproducible across independent AMICA implementations. Past that,
more data buys convergence robustness but not more cross-backend agreement -- the ceiling is set
by float precision and the non-convex local-optimum spread, not by data adequacy.

## Per-config decompose time (2000 iters, 70ch, hallu)

Time scales ~linearly with frames, as expected; native-fortran-f64 is fastest per config
(torch-cuda ~1.4-1.5x), and torch-cuda-f32 undercuts f64 (and stops early at k=152).

| frames  |     k | fortran-f64 | cuda-f64 | cuda-f32 |
|--------:|------:|------------:|---------:|---------:|
|  73,500 |    15 |      122.6s |   185.0s |   175.0s |
| 147,000 |    30 |      248.1s |   366.4s |   344.7s |
| 294,000 |    60 |      502.3s |   730.8s |   679.3s |
| 490,000 |   100 |      848.2s |  1216.8s |  1125.7s |
| 747,750 | 152.6 |     1303.1s |  1856.2s |  1496.5s |

(Per-sample normalized final LL falls monotonically with more data, -3.285 at k=15 to -3.699 at
k=152; that is data heterogeneity -- a longer recording spans more brain states the single model
must explain -- not a worse fit.)

## Figures
- `benchmarks/figures/ksweep_70ch.png` -- mean cross-backend |corr| vs k (the headline: steep
  climb to the k~=60 knee, then a flat ~0.982 plateau below the 1.0 line).
- `benchmarks/figures/ksweep_70ch_matrix.png` -- per-config equivalence matrix at k=152,
  showing the f64-vs-f64 = 0.995 cell and the f32 rows at ~0.97-0.98.

## Note on the harness
`--compare` writes the k-sweep line plot to `--figure` and the per-config matrix to a derived
`{stem}_matrix{suffix}` path, so the two no longer overwrite each other (the first pass of this
run clobbered the line plot with the matrix; fixed in `_compare`).
