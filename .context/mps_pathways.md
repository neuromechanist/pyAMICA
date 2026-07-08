# MPS / Apple-Silicon GPU acceleration: pathways analysis

Research into whether and how the natural-gradient EM backend (`AMICATorchNG`)
can be accelerated on Apple-Silicon GPUs. Motivated by MPS's growing traction,
and by the #63/#70 findings that MPS was *slower* than CPU on the 32-channel
sample data and that float32 diverges on full-size data.

## The one fact that governs everything: no FP64 on Apple GPUs

Apple-Silicon GPUs have **no native double-precision (FP64) hardware** ([Apple
GPU microarchitecture](https://github.com/philipturner/metal-benchmarks);
[real-world tech](https://www.realworldtech.com/forum/?threadid=217891&curpostid=218072)).
Consequently:

- **PyTorch MPS does not support float64** and still does not as of mid-2026 --
  `Cannot convert a MPS Tensor to float64 dtype ... the MPS framework doesn't
  support float64` ([PyTorch forums](https://discuss.pytorch.org/t/typeerror-cannot-convert-a-mps-tensor-to-float64-dtype-as-the-mps-framework-doesnt-support-float64-please-use-float32-instead/180852),
  [Apple Developer forum, 2026](https://developer.apple.com/forums/thread/797778)).
- **MLX** (Apple's own array framework) supports float64 **only on CPU**; GPU
  arrays must be float32 ([MLX data types](https://ml-explore.github.io/mlx/build/html/python/data_types.html),
  [MLX issue #799](https://github.com/ml-explore/mlx/issues/799)).

`AMICATorchNG` computes in **float64 for Fortran parity**. So the parity path can
never run on an Apple GPU. **Every MPS pathway therefore requires a numerically
stable float32 (or mixed-precision) AMICA** -- exactly the wall #70 hit.

## Pathway A (enabler): stabilize float32 -- DONE (#75)

This was the prerequisite for all Apple-GPU acceleration, and it is **resolved**
(epic #74 Phase A, issue #75). The fix was **not** compensated summation or mixed
precision -- diagnostics on the real data ruled those out: accumulating the block
sufficient statistics in float64 (and Neumaier compensated summation) did **not**
stop the divergence, and neither did computing the density or responsibilities in
float64.

The real cause was a **single per-element divide-by-zero**. The mu-denominator
statistic is `sbeta*sum(ufp/y)` with `ufp = u*fp`; at a sample sitting on a
mixture mean, float32 rounds the scaled activation `y` to *exactly* 0, and the
score `fp(0)=0`, so that term is `0/0 = NaN` and one NaN summand poisons the whole
denominator (float64 never lands `y` on exact 0). Guarding that division
(`ufp / where(y==0, 1, y)`, contributing 0 for the measure-zero sample) makes
full-data float32 converge across seeds and match the float64 LL to ~5 significant
digits. Crucially the guard **needs no float64**, so it holds on MPS.

Consequence: the earlier "Kahan / mixed-precision is the credible route, payoff
shrinks to ~1.5-2x" framing is retired. The float64-accumulation mixed-precision
mode is an *unneeded* CPU/CUDA-only option, not the Apple-GPU door. float32 stays
~7-significant-digit, not float64-parity (use float64 for Fortran-parity runs).

## Pathway B: exploit the large-workload regime (measure the crossover)

MPS's problem on our data is **dispatch overhead**, not raw throughput: for small
tensors, per-operator Metal kernel launches (limited fusion, generic kernels)
dominate, and CPU with Accelerate/oneDNN wins ([elanapearl blog](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/),
[runebook MPS](https://runebook.dev/en/docs/pytorch/mps)). Our 32-channel /
512-block ops are firmly in the small regime (MPS 0.51x).

That overhead is **fixed per op**, so it amortizes as tensors grow. High-channel
montages (128-256 ch), many-model runs, and larger blocks produce much bigger
per-op tensors where MPS can plausibly overtake CPU. **Actionable:** once float32
is stable (Pathway A), benchmark MPS-float32 vs CPU across channel count / block
size / n_models to find the crossover, rather than concluding from 32 channels.
`benchmarks/benchmark_gpu.py` already sweeps devices; add a dimension sweep.

## Pathway C: MLX port (longer-term, higher ceiling)

MLX consistently beats PyTorch MPS on the same hardware (2-3x for LLM inference)
and is Apple-native with a real compiler / lazy graph ([WWDC25](https://developer.apple.com/videos/play/wwdc2025/315/),
[MLX](https://github.com/ml-explore/mlx)); PyTorch MPS remains eager, largely
unfused, and sometimes falls back to CPU ([State of PyTorch HW 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)).
An MLX backend could both cut dispatch overhead and fuse the per-block work.

Cost: a full backend rewrite, still float32-only on GPU (so Pathway A is still
required first), and a new dependency. High effort, uncertain until A lands. Best
seen as a v2 acceleration option, not a near-term step.

## Pathway D: software FP64 emulation -- DEAD END

`metal-float64` emulates FP64 on Apple GPUs, but: it is **custom-Metal-only (not
usable from PyTorch)**, runs at **1/32-1/64 of FP32 throughput** (~18-32x
penalty), and implements **only add/multiply/FMA -- no transcendentals** (AMICA
needs `exp`/`log`/`pow`), and the project is archived/incomplete
([metal-float64](https://github.com/philipturner/metal-float64)). It cannot run
AMICA and would be slower than the CPU even if it could. Ruled out.

## Recommendation

1. **Apple-GPU acceleration is gated on a stable float32 AMICA (Pathway A)** --
   there is no float64 GPU path on Apple hardware, now or on the visible horizon.
2. **Pathway A is DONE (#75)** -- and the fix was a per-element divide-by-zero
   guard, not compensated/mixed precision (those were tried and ruled out). float32
   converges on full data across seeds, needs no float64, so the MPS prerequisite
   is met.
3. **Next: the dimension-sweep MPS benchmark (Pathway B)** to find where MPS
   actually beats CPU (likely high channel counts, not 32), then **MLX (Pathway C)**
   as the higher-ceiling backend.
4. Meanwhile **float64-CUDA (4.5x, bit-safe) remains the production GPU path**;
   float32 is ~7-sig-digit (not float64-parity), for speed / Apple-GPU
   (documented in `issue-63/perf_findings.md`).

Cost/benefit note: on CUDA the float32 work buys little (float64 already works);
its real value is unlocking Apple GPUs *at all*. Prioritize Pathways B/C by how
much the user base runs on Apple Silicon with large montages.
