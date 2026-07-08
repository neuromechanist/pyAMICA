# MPS / Apple-Silicon GPU acceleration: pathways analysis

Research into whether and how the natural-gradient EM backend (`AMICATorchNG`)
can be accelerated on Apple-Silicon GPUs. Motivated by MPS's growing traction,
and by the #63/#70 findings that MPS was *slower* than CPU on the 32-channel
sample data and that float32 diverged on full-size data (the latter now fixed in
#75; see Pathway A).

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
stable float32 AMICA** -- the wall #70 hit, now cleared by #75 (Pathway A).

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

## Pathway B: measure the crossover -- DONE (#77)

MPS's problem is **dispatch overhead**, not raw throughput: for small tensors,
per-operator Metal kernel launches (limited fusion, generic kernels) dominate and
CPU with Accelerate/oneDNN wins ([elanapearl blog](https://elanapearl.github.io/blog/2025/the-bug-that-taught-me-pytorch/),
[runebook MPS](https://runebook.dev/en/docs/pytorch/mps)). The hypothesis was that
this amortizes on larger tensors (128-256 ch), so MPS might overtake CPU there.

**Measured (#77, `benchmarks/benchmark_dimsweep.py`, real 70-channel EEG,
`.context/issue-77/benchmark_findings.md`):** the answer is more decisive than
"MPS eventually wins" -- it splits by *framework*, not just workload size:
- **MLX is the Apple-GPU win: ~15-25 ms/it, flat across 16-70 channels, ~7x over
  torch-CPU, and faster than an RTX 4090 (CUDA ~36 ms) at EEG scale.** MLX's fused
  lazy graph + unified memory beat both PyTorch's dispatch overhead and CUDA's
  kernel-launch overhead when per-op tensors are small.
- **PyTorch-MPS never wins: 162-255 ms/it, at or *worse* than CPU (255 vs 193 at
  70ch), single- and multi-model.** The Apple-GPU acceleration is MLX, not
  `device="mps"`.
- Results agree across cpu/mps/cuda/mlx and f32/f64 to ~3 digits on real data.
- **Multi-model has no GPU path today** (MLX is single-model MVP; MPS loses), so
  the top fast-follow is multi-model MLX; a 128-256 ch sweep would find the
  eventual MLX/CUDA crossover.

## Pathway C: MLX port -- v1 MVP LANDED (#76)

MLX consistently beats PyTorch MPS on the same hardware (2-3x for LLM inference)
and is Apple-native with a real compiler / lazy graph ([WWDC25](https://developer.apple.com/videos/play/wwdc2025/315/),
[MLX](https://github.com/ml-explore/mlx)); PyTorch MPS remains eager, largely
unfused, and sometimes falls back to CPU ([State of PyTorch HW 2025](https://tunguz.github.io/PyTorch_Hardware_2025/)).
An MLX backend could both cut dispatch overhead and fuse the per-block work.

**Status (#76):** `AMICAMLXNG` (`pyAMICA/mlx_impl/core.py`) is a v1 MVP -- single-model,
generalized-Gaussian, natural gradient. It is a **hybrid**: the E/M-step hot path runs on
the GPU in float32 (with the Phase A `ufp/y` guard), while all `mlx.core.linalg` is CPU-only
in MLX 0.32, so `inv(A)`/`slogdet(W)` run on the CPU stream (hoisted to once per iteration --
measured ~42 us/iter vs a ~13 ms GPU E-pass, so not the bottleneck; `mx.eval` placement is).
`lgamma`/`digamma` (absent in MLX) are computed host-side via SciPy on the small `rho` array.
It matches the PyTorch float32 backend's converged LL to ~2e-6 and the NumPy reference stats to
rtol ~1e-4. MLX is an optional dependency (Apple Silicon only), so CI skips these tests. Newton,
the other PDF families, sharing, multi-model, and save/load are fast-follows. Whether it beats
CPU/MPS is Pathway B's question.

Cost (estimated before landing, now borne by the #76 MVP): a backend rewrite,
still float32-only on GPU (Pathway A, #75, provides that), and a new optional
dependency. The MVP is in-tree (see Status above); Newton, the other families,
sharing and multi-model remain fast-follows.

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
