# Issue #63: AMICATorchNG performance findings

Optimizing the per-iteration cost. All numbers are single-model, real sample EEG
(32 channels), Newton on, `do_newton=True`. Correctness constraint: the float64
path must stay within the validated parity tolerances (NumPy 1e-10, families
1e-12); the pow-dedup is provably bit-identical.

## 1. Dead-work removal in the E-step (bit-identical) -- PR #69

`_forward` computed the density derivative `dpdf` every block, but every caller
discards it (the exact-EM M-step uses the score `fp`, not `dpdf`) -- a
`|y|^(rho-1)` power and three `exp`s per block-model of pure waste. And `|y|^rho`
was computed twice (density + the rho-update). `_log_pdf_only` computes only
`log_pdf` and threads `|y|^rho` through for reuse.

| Metric (30504 samples, 1 thread, macOS arm64) | Before | After |
|---|---|---|
| pow time (20 iters, cProfile) | 2.55 s | 1.30 s (-49%) |
| wall time | 255 ms/it | **166 ms/it (-35%)** |

**Bit-identical:** single-model final LL matches to all 15 digits; full torch
suite green.

## 2. block_size default 128 -> 512

Larger blocks give bigger tensor ops (less Python/dispatch overhead, better
threading/GPU use). Matches the Fortran reference (512).

| block_size (1 thread) | ms/it |
|---|---|
| 128 | 168 |
| 512 | 137 (-18%) |
| 2048 | 130 (-23%) |

A single iteration's sufficient stats are block-size-independent to ~1e-8 (test
`test_block_size_independence`); the multi-iteration trajectory shifts ~1e-6 as
it changes (chaotic optimization compounds the 1e-8). All tests specify
block_size explicitly, so the default change is safe (108 passed).

## 3. GPU: CUDA float64 is a clean 2.14x (NVIDIA RTX 4090)

Real device numbers from `hallu` (RTX 4090), full data, 50 iters,
`benchmarks/benchmark_gpu.py`:

| device | dtype | ms/it | vs CPU-f64 | final LL |
|---|---|---|---|---|
| cpu | float64 | 281 | 1.00x | -3.42408 |
| **cuda** | **float64** | **131** | **2.14x** | **-3.42408** (identical) |
| cpu | float32 | 54 | 5.24x | **NaN** |
| cuda | float32 | 35 | 8.05x | **NaN** |

**CUDA float64 is the safe GPU win**: 2.14x and numerically identical to CPU
(-3.42408). The `AMICA` wrapper auto-selects CUDA when present.

## 4. float32 is precision-limited, NOT a free fast path

float32 would be 5-8x faster, but on the full 30504-sample data it **collapses to
NaN around iter 23**: it descends healthily (-3.51 -> -3.44 over 22 iters), then a
mixture component's responsibility mass underflows in float32's ~7-digit range,
the exact-EM `0/0` produces non-finite mu/beta/alpha, and the degenerate-fit
contract (#50) stops it with a clean `nan_ll` (it fails LOUDLY, not silently). It
does converge on smaller slices (4096 samples: -3.234, matching float64). So
float32 is usable for small data / experimentation but not production on
full-size recordings without float32-specific mixture flooring -- tracked as a
follow-up.

## 5. CPU threading is workload-limited

torch intra-op threads barely help (small per-block ops, Python dispatch bound):
~16% at 4 threads, and 8 threads is *slower* than 4 (oversubscription). Guidance,
not a forced default (a library must not mutate global `torch.set_num_threads`):
for CPU runs, ~4 threads is the sweet spot; beyond that returns are negative.

MPS is currently *slower* than CPU on 32-channel data (0.51x) -- dispatch
overhead unamortized on small montages; revisit with newer MPS drivers / larger
channel counts.

## Summary

- **Landed (float64, safe):** pow-dedup (-35%) + block_size 512 (-18%) compound
  to ~-47% CPU; CUDA float64 is a further 2.14x, auto-selected.
- **Documented / follow-up:** float32 stabilization (5-8x potential, currently
  NaNs on full data); MPS performance with newer drivers.
