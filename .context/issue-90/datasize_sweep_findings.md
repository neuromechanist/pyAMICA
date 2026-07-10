# Issue #90: Data-size (k-factor) sweep at 70 channels

## Question
Phase 3 (#87) showed cross-backend IC equivalence *falling* as channels grow at a
fixed 147k frames (frames constant -> more channels -> lower data-adequacy factor
`k = frames / channels^2`). This phase isolates the same factor along the other
axis: **hold channels = 70 and increase frames**, so `k` rises, and ask whether
equivalence climbs toward 1.0 as the decomposition becomes better-determined.

## Setup
- Data: ds002718 sub-002, 70-channel EEG, full 747,750-frame recording
  (`benchmarks/data/ds002718_sub-002_eeg70_full.npy`), first-N-frame truncations.
- Frames 73,500 / 147,000 / 294,000 / 490,000 / 747,750 -> **k = 15 / 30 / 60 /
  100 / 152.6** at 70 channels.
- Backends: `native-fortran-f64`, `torch-cuda-f64`, `torch-cuda-f32`, 2000 iters
  each, on the CUDA host (RTX 4090). Command:
  `benchmark_decompose.py --data ..._full.npy --channels 70 --frames 73500,147000,294000,490000,747750 --iters 2000 --backends native-fortran-f64,torch-cuda-f64,torch-cuda-f32`
  then `--compare`.

## Result

| frames | k | mean \|corr\| | min | comps >0.95 |
|---|---|---|---|---|
| 73,500 | 15.0 | 0.892 | 0.838 | 49.5% |
| 147,000 | 30.0 | 0.932 | 0.898 | 61.4% |
| 294,000 | 60.0 | 0.983 | 0.974 | 92.9% |
| 490,000 | 100.0 | 0.983 | 0.980 | 96.2% |
| 747,750 | 152.6 | 0.982 | 0.970 | 92.4% |

![k-sweep](../../benchmarks/figures/equivalence_k90.png)

Cross-backend equivalence rises sharply with `k` and then **saturates at ~0.98
once k >= 60**. Per-backend (largest frames), the two float64 backends agree best
(`native-fortran-f64` vs `torch-cuda-f64` = **0.996**); the float32 path sits
slightly lower (0.97), consistent with its ~7-significant-digit precision. All
three agree on the final log-likelihood to ~3 digits (LL ~ -3.70 at 70ch), and
timing at the largest frames was torch-cuda-f32 1465 s < native-fortran-f64
1822 s < torch-cuda-f64 1858 s.

## Interpretation and caveats

- **The equivalence knee is between k=30 and k=60 for this recording.** Below it
  the decomposition is under-determined and the backends settle into different
  but equally valid local optima (AMICA is non-convex); at and above it they
  recover the same independent components.
- **This threshold is data-specific, not a general law.** Where the knee falls
  depends on the recording's SNR, effective rank, and source structure, so we do
  **not** claim a universal k. The common rule-of-thumb minimum (k ~ 20-30) is a
  lower bound; here full cross-backend equivalence needed roughly k >= 60.
- **The 5-point sweep does not resolve where in [30, 60] the knee sits.**
  Pinning it down would need finer sampling (e.g. k = 40, 50) and, to
  generalize, multiple subjects/datasets.
- **The plateau is ~0.98, not 1.0.** The residual gap is intrinsic estimator
  spread plus the float32 path, consistent with #27: even Fortran does not agree
  with *itself* at 1.0 on under-determined partitions. So ~0.98 with >92% of
  components matched is "the same decomposition," not a defect.

Together with Phase 3 (#87, channel axis), this confirms the **data-adequacy
factor `k` governs cross-backend IC equivalence along both the channel and the
frame axes** for this recording. Backend/precision is not the driver; how
well-determined the decomposition is, is.

## Follow-ups
- #91 (spatially-distributed channel subsets) makes the reduced-channel maps in
  the channel-axis sweep physically meaningful.
- A finer `k in [30, 60]` sweep and a multi-subject replication would localize and
  test the generality of the knee (not blocking; the qualitative result is clear).
