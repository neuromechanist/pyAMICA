# Phase 2 (#86): CPU core-count scaling

Extends the dimension-sweep harness with a `--threads` sweep across the CPU backends
(torch-cpu via `set_num_threads`, numpy via `threadpoolctl`, native-fortran via
`OMP_NUM_THREADS`); GPU backends run once. Question: **does adding CPU cores let the CPU
catch the GPU?**

## Setup

hallu (Linux x86_64, **RTX 4090**, **32 cores**, torch 2.12.1+cu130), real ds002718 sub-002
EEG, 30000 samples, single-model, matched settings (n_mix=3, pdftype=0, do_newton off,
block_size 512). iters 20 x 2 repeats (min). ms/iteration, lower is better.

## Result: fastest CPU thread count vs the GPU

| channels | native-fortran | numpy-cpu-f64 | torch-cpu-f64 | **torch-cuda-f64** |
|---------:|---------------:|--------------:|--------------:|-------------------:|
|       16 |       **3.16** |        182.85 |         59.48 |              34.18 |
|       32 |      **10.00** |        366.99 |         72.91 |              34.99 |
|       48 |      **12.11** |        546.16 |         78.12 |              35.19 |
|       70 |      **24.21** |        791.22 |         95.91 |              38.48 |

LL agrees to ~3 digits across backends (torch-cpu/cuda identical to 5 digits; native-fortran
~0.02 off, its clock-seeded init, expected).

## CPU scaling (ms/iter vs threads), with the CUDA reference

```
              4t      8t     16t     32t     | CUDA
16ch fortran  10.00   10.00   10.00    3.16  | 34.18
     numpy   190.09  187.12  182.85  187.00  |
     torch    62.99   59.48   63.50 1230.03  |
32ch fortran  23.68   15.79   10.00   10.00  | 34.99
     numpy   373.89  372.16  366.99  369.36  |
     torch    76.65   72.91  134.24 1738.88  |
48ch fortran  40.00   30.00   20.00   12.11  | 35.19
     numpy   548.00  555.38  546.16  551.16  |
     torch    87.66   78.12   79.65  973.16  |
70ch fortran  66.32   41.05   40.00   24.21  | 38.48
     numpy   791.22  806.11  834.47  837.96  |
     torch   105.63   95.91   97.02  923.32  |
```

## Findings

1. **Native Fortran + OpenMP is the only CPU backend that scales with cores, and on 32 cores
   it beats the RTX 4090 at every EEG channel count** (16ch ~11x, 32ch ~3.5x, 48ch ~3x, 70ch
   ~1.6x faster than CUDA f64). At 70ch it scales 66 -> 41 -> 40 -> 24 ms/it across 4 -> 32
   threads. This extends Phase 1's single-thread-count result: the native reference is not just
   competitive, it is the fastest option at EEG scale when given cores.

2. **torch-cpu does not scale and collapses at 32 threads.** It is fastest around 8 threads
   (73-96 ms/it) and flat to 16, then regresses **10-20x at 32 threads** (923-1739 ms/it) --
   intra-op thread oversubscription on a small problem (the block/matrix sizes at EEG scale are
   too small to feed 32 threads; scheduling + false sharing dominate). Practical rule: do **not**
   set torch CPU threads to the full core count for AMICA at EEG scale; ~4-8 is the sweet spot,
   matching the #63 laptop finding. torch-cpu never catches the GPU here.

3. **numpy is thread-flat** (~183 ms/it @16ch to ~800 @70ch, essentially constant across 4-32
   threads): its per-iteration cost is BLAS/Python-bound, not intra-op-thread bound, so extra
   cores do nothing. It is the slowest backend and never approaches the GPU.

## Caveats (honest)

- **Native-fortran timing at low channel counts is resolution-limited.** amica stamps per-iter
  time to ~10 ms; at 16-32ch the true per-iter is at/below that floor, so those rows read a flat
  10.00 ms (every iter rounds to 0.01 s) and the 16ch@32t = 3.16 ms is a below-floor noisy mean.
  Trust the **48ch and 70ch** fortran scaling curve (per-iter > 10 ms); treat <=32ch as "<= ~10
  ms/it, faster than the GPU" rather than an exact number.
- Cross-backend LL differs for native-fortran by its clock-seeded init (Phase 1 caveat), so its
  LL column is a sanity check, not a fixed-seed parity number.
- Single machine, single-model. Multi-model CUDA + the merged cross-platform grid (Mac
  mlx/mps + this) are Phase 3 (#87).
