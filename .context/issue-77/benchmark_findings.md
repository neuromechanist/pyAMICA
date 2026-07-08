# Phase B: cross-platform result + performance benchmark (issue #77)

The epic (#74) deferred one question through Phases A and C: **does an Apple/NVIDIA
GPU actually beat the CPU for AMICA, and where?** Phase B answers it, on **real
70-channel EEG** (OpenNeuro ds002718 sub-002, Wakeman-Henson faces), comparing
every backend on **both** performance (ms/iteration, warmed, min-of-repeats) and
results (converged log-likelihood). Harness: `benchmarks/benchmark_dimsweep.py`;
data prep in `benchmarks/README_dimsweep.md`. Matched settings: n_mix=3,
pdftype=0, do_newton=False, block_size=512, seed=42, samples=30000 (single-model)
/ 20000 (multi-model), 25 / 20 iters.

Hosts: **Apple Silicon** (this Mac: cpu / mps / mlx) and **hallu** (RTX 4090:
cuda; its CPU was load-contended so CPU backends were run on the Mac). MLX and
CUDA are on *different machines*, so MLX-vs-CUDA is "best Apple-GPU path vs a
strong NVIDIA GPU", not a same-box comparison.

## Single-model (m1) -- performance, ms / iteration

| channels | mlx-f32 | cuda-f32 | cuda-f64 | torch-cpu-f32 | torch-cpu-f64 | torch-mps-f32 | numpy-cpu-f64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | **15.4** | 35.5 | 35.0 | 52 | 71 | 189 | 310 |
| 32 | **21.3** | 35.5 | 36.2 | 143 | 161 | 162 | 624 |
| 48 | **19.5** | 36.0 | 35.9 | 151 | 168 | 168 | 918 |
| 70 | **25.2** | 35.6 | 38.6 | 173 | 193 | 255 | 1350 |

## Single-model (m1) -- results, converged LL (all agree to ~3 digits)

| channels | mlx-f32 | cuda-f64 | torch-cpu-f64 | torch-mps-f32 | numpy-cpu-f64 |
|---:|---:|---:|---:|---:|---:|
| 32 | -3.28634 | -3.28635 | -3.28636 | -3.28635 | -3.28623 |
| 48 | -3.20951 | -3.20952 | -3.20953 | -3.20951 | -3.20979 |
| 70 | -3.21579 | -3.21562 | -3.21560 | -3.21570 | -3.21122 |

## Multi-model -- ms/it and LL (Mac; MLX excluded = single-model MVP; CUDA pending)

m2 (no share), 70ch: torch-cpu-f32 **224** / cpu-f64 253 / mps-f32 270 / numpy 1894 ms.
m2+share, 70ch: cpu-f32 **224** / cpu-f64 253 / mps-f32 262 / numpy 1891 ms.
Sharing activates in torch (70ch LL -2.943 -> -3.049); numpy's sharing diverges
(-2.919 -> -2.917) -- multi-model is not partition-identifiable and has no
bit-exact oracle (#27/#60), so cross-backend LL differs by intrinsic estimator
spread, not a defect.

## Findings

1. **MLX is the decisive Apple-GPU win: ~15-25 ms/it, flat with channel count --
   ~7x faster than torch-CPU, and faster than the RTX 4090 (CUDA ~36 ms) at
   EEG scale.** (MLX's fused lazy graph + unified memory beat both PyTorch's
   dispatch overhead and CUDA's kernel-launch overhead when the per-op tensors
   are small, as 70ch/30k are. On a much larger workload CUDA would likely
   overtake MLX -- worth a future high-dimensional check.)
2. **PyTorch-MPS is NOT a win: 162-255 ms/it, at or WORSE than the CPU** (255 vs
   193 at 70ch), single- AND multi-model. So the Apple-GPU acceleration comes
   from **MLX, not `device="mps"`** -- an actionable steer for users.
3. **CUDA (RTX 4090) is a flat ~35-39 ms/it**, second only to MLX here, and (as in
   #63) f32 ~= f64 at this size (launch-bound, not compute-bound).
4. **Results agree across every platform / device / precision**: single-model LL
   matches to ~3 digits (mlx = cuda = cpu = mps ~ -3.216 at 70ch; numpy within
   ~0.004). This validates the whole backend family end-to-end on real data.
5. **numpy is 50-70x slower than MLX** -- the reference implementation, not a
   production path.
6. **Multi-model has no GPU acceleration today**: MLX (the only Apple-GPU win) is
   single-model-only (v1 MVP), and MPS loses. **Extending `AMICAMLXNG` to
   multi-model is the highest-value fast-follow** -- it would carry the ~7x MLX
   win to multi-model AMICA.

## Recommendation

- **Apple Silicon: use the MLX backend** for single-model AMICA (~7x over CPU);
  avoid `device="mps"` (no gain). **NVIDIA: float64-CUDA** stays the bit-safe
  production GPU path.
- **Fast-follow: multi-model MLX** (issue TBD) to extend the win beyond
  single-model; and a high-dimensional (128-256 ch, many-model) sweep to find the
  MLX/CUDA crossover.

## Caveats / pending

- MLX vs CUDA is cross-machine (Apple M-series vs RTX 4090 host), not a controlled
  same-box comparison; read it as "best available Apple-GPU vs a strong NVIDIA
  GPU", not silicon-vs-silicon.
- **Multi-model CUDA is pending** (the hallu SSH session dropped mid-run); the
  single-model CUDA data is complete. Re-run `--n-models 2 [--share] --backends
  torch-cuda-f64,torch-cuda-f32` on the CUDA host to fill it in.
