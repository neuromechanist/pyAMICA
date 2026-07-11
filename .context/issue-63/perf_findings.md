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
threading/GPU use). 512 is a fixed value inside Fortran's `do_opt_block`
auto-tune range (128-1024, step 128); Fortran's *header* default is 128 and it
auto-tunes per host, so 512 is an empirical pick, not a fixed reference value.

| block_size (1 thread) | ms/it |
|---|---|
| 128 | 168 |
| 512 | 137 (-18%) |
| 2048 | 130 (-23%) |

A single iteration's sufficient stats are block-size-independent to ~1e-8 (test
`test_blocking_invariance`); the multi-iteration trajectory shifts ~1e-6 as it
changes (chaotic optimization compounds the 1e-8). No test asserts an exact
trajectory without pinning block_size; the few tests that fit real data on the
default use tolerant (finite/monotone or self-consistency) assertions insensitive
to the ~1e-6 shift, so the default change is safe (full suite green).

## 3. GPU: CUDA float64 is a clean ~4.5x (NVIDIA RTX 4090)

Real device numbers from `the CUDA workstation` (RTX 4090), full data, 50 iters, **warmed**
(first CUDA call pays context init + kernel compilation, ~2-3x inflation if not
excluded), min of 4 repeats, `torch.set_num_threads(16)`:

| device | dtype | ms/it | vs CPU-f64 | final LL |
|---|---|---|---|---|
| cpu (16 thr) | float64 | 172.8 | 1.00x | -3.42408 |
| **cuda** | **float64** | **38.5** | **4.5x** | **-3.42408** |

**CUDA float64 is the safe GPU win**: 4.5x over a 16-thread CPU on the same host,
and it agrees with the CPU LL to 5 significant digits (-3.42408, the benchmark's
print precision; float64 device-to-device reductions can still differ at
~1e-10-1e-13 from summation order). The `AMICA` wrapper auto-selects CUDA when
present. (Measurement caveat: a cold first
CUDA call reads ~131 ms/it -- always warm up before timing GPU.)

## 4. float32 stabilized by a divide-by-zero guard (#75 -- supersedes #70)

**RESOLVED (issue #75).** float32 (~5x CPU / ~10-19x CUDA faster) previously
**diverged to NaN on the full 30504-sample data across every seed, Newton on AND
off** (crashing iter ~9-105), converging only on small slices. Epic #74 Phase A
fixed it with a **one-line guard** (`_get_block_updates`); float32 now converges
across seeds and matches the float64 LL to ~5 significant digits.

**Actual root cause -- it IS a divide-by-zero, per-element (#70 looked at the
wrong granularity).** The mu denominator is `sbeta*sum(ufp/y)` (`ufp=u*fp`). At a
sample sitting on a mixture mean, float32 rounds `y` to *exactly* 0, and the
score `fp(0)=0` for every family, so that sample's `ufp/y` is `0/0 = NaN` -- and
one NaN summand poisons the whole `dmu_d`. #70 watched the *summed* denominators
(`dmu_d`/`dbeta_d`/`dalpha_n`), which read healthy O(1e3) right up to the crash,
and so concluded "not a divide-by-zero" -- but the 0/0 is in the per-sample term
*before* the sum. float64 never rounds `y` to exactly 0, hence float64-only.

**Why summation precision was NOT the sink (diagnostics, #75):** accumulating the
float32 block partials in float64 (and Neumaier compensated summation) did **not**
help -- both still diverged, at the same iterations. And computing the density
`|y|^rho`/`log_pdf` in float64 did **not** help either (nor, per #70, the
responsibility `logsumexp`/`softmax`). Only guarding the `ufp/y` division does.
So the earlier "needs mixed precision (float64 density + accumulation), payoff
shrinks to ~1.5-2x" conclusion is **superseded**: the fix needs no float64 at all,
which is exactly why it also works on MPS (no FP64 on Apple GPUs). float32 stays
non-parity with float64 (~7 sig digits); use float64 for Fortran-parity runs.

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
  to ~-47% CPU; CUDA float64 is a further ~4.5x (warmed, vs 16-thread CPU),
  auto-selected and numerically identical.
- **Landed (float32):** the `ufp/y` divide-by-zero guard (#75) makes full-data
  float32 converge across seeds (~5x CPU / ~10-19x CUDA potential; ~7 sig digits,
  not float64-parity). Unblocks the Apple-GPU roadmap (epic #74): the fix needs no
  float64, so it holds on MPS.
- **Follow-up:** MPS dimension-sweep benchmark to find the CPU/GPU crossover
  (epic #74 Phase B); float64-accumulation mixed precision remains an unneeded
  CPU/CUDA-only option.
