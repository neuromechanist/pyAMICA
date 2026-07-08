# Phase 2 (#86): CPU core-count scaling (cross-platform, f64 + f32)

Adds a `--threads` sweep to the dimension-sweep harness: the CPU backends (torch-cpu via
`set_num_threads`, numpy via `threadpoolctl`, native-fortran via `OMP_NUM_THREADS`) run at
each core count; GPU backends (CUDA, MLX, MPS) run once (thread-independent). Full **channel x
core** grid, both precisions, both machines. Question: **does adding CPU cores let the CPU
catch the GPU, and does f32 scale/work as well as f64?**

Real ds002718 sub-002 EEG, 30000 samples, single-model, matched settings (n_mix=3, pdftype=0,
do_newton off, block_size 512), 20 iters x 2 repeats (min). ms/iteration, lower is better.
Channels swept 16/32/48/70; 70ch shown as the headline below (full grid in the result JSONs).

## hallu -- Intel x86_64, 32 cores, RTX 4090, torch 2.12.1+cu130

ms/iter vs cores, 70 channels:

| backend            |    4c |    8c |   12c |   16c |   24c |
|--------------------|------:|------:|------:|------:|------:|
| native-fortran-f64 |  69.5 |  43.2 |  49.0 |  40.0 |**30.0** |
| torch-cpu-f64      | 105.9 |  91.0 |  92.6 | 142.5 | 212.8 |
| torch-cpu-f32      |  84.6 |  69.5 |  71.8 |  70.9 |  73.0 |
| numpy-cpu-f64      | 794.6 | 810.3 | 871.7 | 855.5 | 866.1 |
| GPU: cuda-f64 = **38.5**   cuda-f32 = **36.2** (flat, run once)                |

## Mac -- Apple Silicon, 14 cores (10P + 4E), MLX + MPS

ms/iter vs cores, 70 channels:

| backend            |    4c |    8c |
|--------------------|------:|------:|
| native-fortran-f64 | 100.0 |**70.0** |
| torch-cpu-f64      | 131.4 | 169.9 |
| torch-cpu-f32      | 111.7 | 144.4 |
| numpy-cpu-f64      | 627.0 | 627.4 |
| GPU: mlx-f32 = **33.4**   mps-f32 = 217.7 (flat, run once)                     |

## f32 works and scales (the correctness assurance)

f32 final LL matches f64 to ~4-5 significant digits at every channel count and backend
(e.g. 70ch: torch-cpu-f32 -3.23606 / cuda-f32 -3.23631 vs f64 -3.23622; 16ch identical to
5 digits). So f32 is numerically correct on both CUDA and Apple (MLX/MPS), not just faster.

f32 also **scales more gracefully than f64 on torch-cpu**: on hallu the f32 path stays ~70 ms
across 8-24 cores while f64 collapses (91 -> 213 ms at 8 -> 24 cores). f32's smaller footprint
and lighter intra-op work avoid the oversubscription cliff. On Mac, torch-cpu (both precisions)
still regresses past 4 cores. cuda-f32 == cuda-f64 (~36 ms) -- the GPU is overhead-bound, not
precision-bound (see below).

## GPU utilization verification (does the GPU actually do the work?)

Each benchmark GPU fit is short (~1-1.5 s), so live `nvidia-smi` shows brief bursts, not
sustained load -- the CPU backends dominate wall-clock. During a sustained 200-iter fit at 70ch
the RTX 4090 hits **98% SM util for f64, ~61% for f32** (both ~37 ms/it; 21-34 MB allocated,
tensors on `device=cuda`). So the GPU genuinely runs the work; it is not a CPU fallback. But at
EEG scale it is **overhead-bound, not FP-throughput-bound**: f32 and f64 land at the same ~36 ms
because per-iteration cost is kernel-launch / Python-loop / sequential-EM overhead, not GPU
compute (f32 only reaches 61% util -- it finishes each kernel fast then waits). This is why more
precision does not help the GPU, and why a fast multi-core CPU can beat it.

## Apple Silicon P-core vs E-core (checked)

The Mac CPU work uses the performance cores, not the efficiency cores: a foreground run and a
background run give the same torch-cpu number (167.1 vs 167.5 ms @70ch/8c), and native-fortran
speeds up from 4 -> 8 -> 12 cores (100 -> 70 -> 60 ms), which is impossible if all threads were
pinned to the 4 E-cores. torch-cpu's poor Mac scaling is intra-op oversubscription, not E-core
placement. Mac core counts are held to 4/8 because past ~8 both effects (oversubscription, and
spilling onto the 4 E-cores beyond the 10 P-cores) muddy the number.

## Findings

1. **Native Fortran + OpenMP is the only CPU backend that scales with cores, and on 32 cores it
   beats the RTX 4090** (70ch: 30 ms @24c vs cuda 38.5 ms; ~11x faster at 16ch). A tight compiled
   loop has no per-iteration launch overhead -- exactly the GPU's bottleneck at this scale.
2. **The RTX 4090 is flat at ~36-38 ms/it regardless of precision or channels** (overhead-bound).
   The f64/consumer-card FP throttle is not the story -- f32 is no faster.
3. **torch-cpu-f64 peaks at ~8 cores then collapses** (10-20x at 24-32c, oversubscription);
   torch-cpu-f32 is faster and scale-stable but still never catches the GPU.
4. **numpy is thread-flat** (BLAS/Python-bound), slowest everywhere.
5. **MLX is the efficiency winner of the whole comparison.** At ~33 ms/it (flat, no tuning) a
   laptop Apple GPU matches the 450 W RTX 4090 (~36-38 ms) and a 32-core Xeon at full core count
   (native-fortran ~30 ms @24c), at a fraction of the power, cost, and effort -- and its LL is
   correct (matches f64 to ~4-5 digits). native-fortran@24c is marginally faster in raw ms/it, but
   only by pinning every core of a much larger, hotter machine. On Apple Silicon, torch-MPS is the
   opposite story (~200 ms, worse than the CPU); use MLX, never MPS. Matches #77.
6. **f32 is correct (LL matches f64 to ~4-5 digits) on CUDA, MLX, and MPS** and scales at least as
   well as f64 -- safe to use for the GPU fast path.

LL agrees to ~3 digits across all backends/platforms (native-fortran ~0.02 off from its
clock-seeded init; torch-cpu/cuda f64 identical to 5 digits).

## Caveats
- Native-fortran timing at <=32ch is at/below amica's ~10 ms stamp resolution (16ch reads a flat
  10.00; the sub-10 ms values are below-floor noisy means). Trust the 48/70ch scaling curve.
- Cross-platform GPU numbers are different machines (RTX 4090 vs Apple), not a controlled compare.
- Single-model, per-iteration throughput. The full 2000-iter decomposition-equivalence across all
  backends and both machines (total time + Hungarian-matched IC figure) is Phase 3 (#87).
