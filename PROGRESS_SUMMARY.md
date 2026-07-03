# PyTorch Implementation Progress Summary

## What We've Accomplished

### 1. Core PyTorch Implementation ✅
- Created `AMICATorch` class with GPU/MPS/CPU support
- Implemented natural gradient optimization
- Added Fortran-style debug output mode
- Fixed MPS compatibility issues with eigendecomposition
- Achieved basic convergence matching Fortran pattern

### 2. Validation Framework ✅
- Built comprehensive comparison script (`validate_implementations.py`)
- Implemented component matching using Hungarian algorithm
- Successfully runs both Fortran and PyTorch implementations
- Generates detailed comparison reports

### 3. API and Integration ✅
- Created main `AMICA` class with scikit-learn style API
- Maintained backward compatibility (NumPy version as `AMICA_NumPy`)
- Updated package structure to use PyTorch as default
- Added model save/load functionality

### 4. Documentation ✅
- Created `FEATURE_PARITY.md` with comprehensive comparison
- Updated `CLAUDE.md` with current status and roadmap
- Documented critical missing features and priorities

### 5. Natural-gradient EM backend at Fortran parity ✅ (epic #9 / issue #24)
- Added `AMICATorchNG` (`torch_impl/amica_torch_ng.py`), wired into `AMICA(backend="ng")`
- Root-caused the parity gap: the natural-gradient A-update was transposed / multiplied on the
  wrong side (proven machine-exact); plus exact-EM mixture updates, the digamma rho update, the
  symmetric-ZCA sphere, the output transpose, and the NumPy Jacobian LL
- Newton ported from NumPy and positive-definite (0 fallbacks on the sample data)

## Current Results

### Validation Metrics (`backend="ng"`, issue #24)
- **Log-likelihood**: LL ~ -3.40 (matches Fortran -3.4018; the earlier ~13x scaling gap was an
  artifact of the pre-parity basic backend)
- **Component correlation**: ~0.997 (Hungarian-matched, Newton positive-definite) -- clears the
  >0.95 gate
- **NumPy backend**: carries the same fixes and matches
- **Multi-model** (`n_models>1`): M-step bit-exact vs Fortran; full-fit partition matching is
  partition-limited (~0.77), tracked in #27

The pre-epic basic backend (`backend="torch"`, Adam) still shows the old ~0.78 correlation / ~13x
LL scaling; it is retained but superseded by `backend="ng"` for parity work.

## Next Steps (Prioritized)

### 1. Adaptive PDF selection for `AMICATorchNG` (issue #26)
- Laplace / Student-t / generalized-Gaussian selection (present in the NumPy path, beyond strict
  Fortran parity which uses a fixed GG PDF)

### 2. Multi-model partition matching (issue #27)
- Resolve the full-fit multi-model cross-correlation gap; includes the omitted per-model bias
  `c` update

### 3. Retire the parked/superseded paths (issue #32)
- `AMICATorchV2` is parked (superseded by `AMICATorchNG`); once `backend="ng"` is the de-facto
  default, remove `AMICATorchV2` and promote `backend="ng"` to default, then reassess the basic
  `backend="torch"` path (legacy mixture M-step bug tracked in #31)

## Technical Achievements
- Successfully handles MPS device limitations
- Automatic differentiation working correctly
- Memory efficient batch processing
- Progress bars and debug output modes

## Commits Made
1. Add main AMICA interface using PyTorch backend
2. Update package init to use PyTorch as default
3. Add comprehensive validation script
4. Add feature parity documentation
5. Add Fortran binary for macOS validation
6. Update gitignore for validation output

## Repository Status
- Epic #9 (PyTorch backend Fortran parity) delivered via `AMICATorchNG` / `backend="ng"`
- Remaining follow-ups tracked as issues #26 (adaptive PDF), #27 (multi-model), #30/#31 (legacy)