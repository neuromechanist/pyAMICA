# Feature Parity Report: Fortran vs PyTorch AMICA

> **Status (2026-07-04, epic #9 / issue #24 / #32):** the "PyTorch" column in the older tables
> below records the historical numbers of the pre-epic basic backend (`AMICATorch`, Adam/autograd),
> which was **removed in #32**. The current and only PyTorch backend is the natural-gradient EM
> port `AMICATorchNG`, which `AMICA` now wraps directly. It reaches Fortran parity: single-model
> LL ~ -3.40 (vs Fortran -3.4018) and Hungarian component correlation ~0.997 with Newton
> positive-definite (0 fallbacks), and implements Newton + Fortran-style ramping and outlier
> rejection (`do_reject`). Rows below that still read "Missing"/"Partial" describe the removed
> basic backend, not `AMICATorchNG`. Remaining gaps on the NG path: adaptive-PDF selection (#26)
> and full multi-model partition matching (#27, M-step is already bit-exact vs Fortran). See
> `PROGRESS_SUMMARY.md` and ADR `.context/decisions/0001-torch-backend-natural-gradient-em.md`.

## Implementation Status Overview

| Component | Fortran | NumPy | PyTorch | Notes |
|-----------|---------|-------|---------|-------|
| **Core Algorithm** | ✅ | ⚠️ | ✅ | PyTorch more stable than NumPy |
| **GPU Support** | ❌ | ❌ | ✅ | CUDA/MPS/CPU automatic |
| **Output Format** | ✅ | ⚠️ | ✅ | Fortran-style in debug mode |
| **Performance** | ⭐⭐⭐ | ⭐ | ⭐⭐⭐⭐ | PyTorch fastest on GPU |

## Detailed Feature Comparison

### 1. Core AMICA Features

| Feature | Fortran | PyTorch | Status | Notes |
|---------|---------|---------|--------|-------|
| **Multiple Models** | ✅ | ✅ | ✅ Complete | `num_models` parameter |
| **Mixture of Gaussians** | ✅ | ✅ | ✅ Complete | `num_mix` components |
| **Generalized Gaussian PDF** | ✅ | ✅ | ✅ Complete | Shape parameter ρ ∈ [1,2] |
| **Natural Gradient** | ✅ | ✅ | ✅ Complete | Implemented |
| **Learning Rate Adaptation** | ✅ | ⚠️ | 🔧 Partial | Basic decay implemented |
| **Convergence Criteria** | ✅ | ⚠️ | 🔧 Partial | dll and grad_norm checks |

### 2. Optimization Methods

| Feature | Fortran | PyTorch | Status | Notes |
|---------|---------|---------|--------|-------|
| **Natural Gradient** | ✅ | ✅ | ✅ Complete | Via autograd |
| **Newton Method** | ✅ | ✅ | ✅ Complete | With pytorch-minimize |
| **Newton Ramping** | ✅ | ❌ | ❌ Missing | Not yet implemented |
| **Line Search** | ✅ | ⚠️ | 🔧 Partial | In pytorch-minimize |
| **L-BFGS** | ❌ | ✅ | ✅ Bonus | Available via PyTorch |
| **Adam Optimizer** | ❌ | ✅ | ✅ Bonus | More stable |

### 3. Data Preprocessing

| Feature | Fortran | PyTorch | Status | Notes |
|---------|---------|---------|--------|-------|
| **Mean Removal** | ✅ | ✅ | ✅ Complete | `do_mean=True` |
| **Sphering (Whitening)** | ✅ | ✅ | ✅ Complete | `do_sphere=True` |
| **PCA Dimension Reduction** | ✅ | ⚠️ | 🔧 Partial | Basic PCA available |
| **Approximate Sphering** | ✅ | ❌ | ❌ Missing | Not implemented |
| **Data Scaling** | ✅ | ✅ | ✅ Complete | Automatic |

### 4. Advanced Features

| Feature | Fortran | PyTorch | Status | Notes |
|---------|---------|---------|--------|-------|
| **Outlier Rejection** | ✅ | ❌ | ❌ Missing | `do_reject` not implemented |
| **Component Sharing** | ✅ | ❌ | ❌ Missing | `share_comps` not implemented |
| **Adaptive PDF Selection** | ✅ | ❌ | ❌ Missing | Fixed PDF type only |
| **Block Size Optimization** | ✅ | ⚠️ | 🔧 Partial | Simple heuristic |
| **History Tracking** | ✅ | ✅ | ✅ Complete | LL and gradient norms |

### 5. Input/Output

| Feature | Fortran | PyTorch | Status | Notes |
|---------|---------|---------|--------|-------|
| **Binary Data Loading** | ✅ | ✅ | ✅ Complete | `.fdt` files supported |
| **Parameter Files** | ✅ | ✅ | ✅ Complete | JSON format |
| **Fortran-style Output** | ✅ | ✅ | ✅ Complete | Debug mode |
| **Progress Bars** | ❌ | ✅ | ✅ Bonus | tqdm in normal mode |
| **Checkpoint Saving** | ✅ | ✅ | ✅ Complete | `.pth` and `.npy` |
| **Result Loading** | ✅ | ✅ | ✅ Complete | Via `loadmodout()` |

### 6. Numerical Stability

| Feature | Fortran | PyTorch | Status | Notes |
|---------|---------|---------|--------|-------|
| **Min/Max Bounds** | ✅ | ✅ | ✅ Complete | Clamping implemented |
| **Log-space Computation** | ✅ | ✅ | ✅ Complete | For stability |
| **Eigenvalue Thresholds** | ✅ | ✅ | ✅ Complete | `min_eig` parameter |
| **Condition Number Checks** | ✅ | ⚠️ | 🔧 Partial | Basic checks |
| **NaN/Inf Detection** | ✅ | ✅ | ✅ Complete | Automatic in PyTorch |

### 7. Performance & Hardware

| Feature | Fortran | PyTorch | Status | Notes |
|---------|---------|---------|--------|-------|
| **OpenMP Parallelization** | ✅ | ❌ | N/A | PyTorch uses different approach |
| **GPU Support** | ❌ | ✅ | ✅ Bonus | CUDA/ROCm/MPS |
| **Automatic Differentiation** | ❌ | ✅ | ✅ Bonus | No manual gradients |
| **Mixed Precision** | ❌ | ✅ | ✅ Bonus | Available if needed |
| **Device Selection** | N/A | ✅ | ✅ Bonus | Automatic or manual |

## Parameters Comparison

### Fully Supported Parameters

| Parameter | Fortran | PyTorch | Default | Notes |
|-----------|---------|---------|---------|-------|
| `num_models` | ✅ | ✅ | 1 | Number of ICA models |
| `num_mix` | ✅ | ✅ | 3 | Mixture components |
| `max_iter` | ✅ | ✅ | 100 | Maximum iterations |
| `lrate` | ✅ | ✅ | 0.1 | Learning rate |
| `do_mean` | ✅ | ✅ | True | Remove mean |
| `do_sphere` | ✅ | ✅ | True | Apply sphering |
| `do_newton` | ✅ | ✅ | False | Newton optimization |
| `min_dll` | ✅ | ✅ | 1e-9 | Min LL change |
| `min_grad_norm` | ✅ | ✅ | 1e-7 | Min gradient norm |

### Partially Supported Parameters

| Parameter | Fortran | PyTorch | Status | Notes |
|-----------|---------|---------|--------|-------|
| `lratefact` | ✅ | ⚠️ | Basic | Simple decay |
| `newt_start` | ✅ | ⚠️ | Basic | Start iteration |
| `newt_ramp` | ✅ | ❌ | Missing | Not implemented |
| `pcakeep` | ✅ | ⚠️ | Basic | PCA components |
| `block_size` | ✅ | ⚠️ | Basic | Simple heuristic |

### Missing Parameters

| Parameter | Fortran | PyTorch | Priority | Notes |
|-----------|---------|---------|----------|-------|
| `do_reject` | ✅ | ❌ | Medium | Outlier rejection |
| `share_comps` | ✅ | ❌ | Low | Component sharing |
| `do_opt_block` | ✅ | ❌ | Low | Block optimization |
| `kurt_start` | ✅ | ❌ | Low | Kurtosis-based init |
| `comp_thresh` | ✅ | ❌ | Low | Sharing threshold |

## Validation Status

### Convergence Behavior (`backend="ng"`, issue #24)

| Metric | Fortran | PyTorch NG | Match? | Notes |
|--------|---------|-----------|--------|-------|
| **Final LL** | -3.4018 | -3.40 | ✅ | Jacobian-normalized, parity |
| **Component Correlation** | - | ~0.997 | ✅ | Hungarian-matched, clears >0.95 gate |
| **Convergence Rate** | Fast | Fast | ✅ | Similar speed |
| **Newton** | posdef | posdef | ✅ | 0 fallbacks on sample data |

The removed pre-epic basic backend showed the old offset (initial LL ~-46, final -44 to -46;
~0.78 correlation); `AMICATorchNG` is now the sole backend and reaches parity.

### Component Quality (`backend="ng"`)

| Metric | Status | Notes |
|--------|--------|-------|
| **Component Correlation** | ✅ ~0.997 | Real sample data vs Fortran (Hungarian match) |
| **Mixing Matrix Recovery** | ✅ | Ascends to Fortran fixed point |
| **Source Separation** | ✅ real-data | Validated on sample EEG (no synthetic; NO-MOCK policy) |

## Remaining Roadmap (`backend="ng"`)

The epic (#9 / #24) delivered core parity on `AMICATorchNG`. Done on the NG path: natural-gradient
EM at the Fortran fixed point, Newton + Fortran-style ramping (positive-definite), exact-EM mixture
updates, symmetric-ZCA sphere, Jacobian LL, and outlier rejection (`do_reject`). Open items:

#### 1. Adaptive PDF Selection (issue #26)
**Why**: different sources have different distributions; beyond strict Fortran parity (fixed GG PDF)
- [ ] Laplace / Student-t / generalized-Gaussian selection on the NG path
- [ ] Kurtosis-based selection and per-component PDF-fit monitoring

#### 2. Multi-Model Partition Matching (issue #27)
**Why**: full 2-model fit is partition-limited (~0.77) though the M-step is bit-exact vs Fortran
- [ ] Resolve the partition/cross-correlation gap
- [ ] Restore the omitted per-model bias `c` update (no-op only for `n_models=1`)

#### 3. Retire Superseded/Legacy Paths
- [x] Remove `AMICATorchV2` and the basic `AMICATorch`; `AMICATorchNG` is now the sole PyTorch
      backend and `AMICA` wraps it directly (issue #32; also made the basic-backend mixture M-step
      bug #31 moot by deleting the module)
- [ ] Legacy NumPy CLI save/load format mismatch (#30)

## Migration Readiness

### `backend="ng"` at Fortran parity ✅
- Single-model LL ~ -3.40 and component correlation ~0.997 vs Fortran (issue #24)
- NumPy backend carries the same fixes
- Multi-model M-step bit-exact; full-fit partition matching tracked in #27

### Recommendation
1. Use `backend="ng"` for parity-critical work (default for that use case)
2. Keep `validate_implementations.py` (real sample data + Fortran binary) green as source of truth
3. Close out adaptive-PDF (#26) and multi-model (#27) before removing the Fortran binary from the loop

## Testing Checklist

- [x] Basic functionality test
- [x] Convergence test
- [x] GPU/MPS support
- [x] Output format compatibility
- [x] Component correlation vs Fortran (real data, Hungarian match)
- [x] NG backend sufficient-stats vs NumPy reference (bit-exact)
- [ ] Large-scale data test
- [ ] Memory usage comparison
- [ ] Speed benchmarks