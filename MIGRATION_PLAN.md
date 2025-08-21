# Migration Plan: NumPy to PyTorch Implementation

## Overview
Replace the NumPy-based AMICA implementation with the PyTorch version to leverage GPU acceleration and automatic differentiation.

## Current Status

### Completed ✅
1. **PyTorch Core Implementation**
   - AMICATorch class with GPU/MPS support
   - Natural gradient and Newton optimizers
   - Gaussian mixture models for ICA
   - Fortran-style output formatting

2. **Testing Framework**
   - Unit tests for components
   - Comparative tests with Fortran
   - Real data testing (no mocks)

3. **Features**
   - Automatic device selection (CUDA/MPS/CPU)
   - Debug mode with detailed output
   - Checkpoint saving
   - Progress monitoring with tqdm

### Known Issues 🔧
1. **MPS Limitations**
   - Some operations fall back to CPU (e.g., eigenvalue decomposition)
   - Workaround: Set `PYTORCH_ENABLE_MPS_FALLBACK=1`

2. **In-place Operations**
   - Some gradient computations need adjustment for MPS
   - Solution: Avoid in-place modifications in forward pass

## Migration Steps

### Phase 1: Fix Current Issues (Week 1)
- [ ] Fix in-place operation errors in log-likelihood computation
- [ ] Handle MPS fallback operations gracefully
- [ ] Ensure numerical stability matches Fortran

### Phase 2: Feature Parity (Week 2)
- [ ] Implement outlier rejection
- [ ] Complete Newton optimization with baralpha
- [ ] Add adaptive PDF selection
- [ ] Implement component sharing

### Phase 3: Replace NumPy Implementation (Week 3)
1. **Update Main Module**
   ```python
   # pyAMICA/__init__.py
   from .torch_impl import AMICATorch as AMICA
   ```

2. **Update CLI**
   - Modify `amica_cli.py` to use PyTorch implementation
   - Keep same interface for backward compatibility

3. **Update Tests**
   - Ensure all existing tests pass with PyTorch
   - Add GPU-specific tests

### Phase 4: Optimization (Week 4)
- [ ] Profile and optimize GPU kernels
- [ ] Implement batched operations
- [ ] Add mixed precision training
- [ ] Optimize memory usage

## Usage After Migration

### Basic Usage
```python
from pyAMICA import AMICA

# Automatically uses GPU if available
model = AMICA(n_channels=32)
model.fit(data, verbose=True)  # Shows Fortran-style output
sources = model.transform(data)
```

### With Debug Output
```python
model = AMICA(n_channels=32)
model.fit(
    data,
    verbose=True,
    debug=True,  # Detailed convergence info
    output_dir='results'
)
```

### Force CPU
```python
model = AMICA(n_channels=32, device='cpu')
```

## Backward Compatibility

### Maintained Features
- Same API as NumPy version
- Compatible file formats
- Same parameter names
- Fortran-style output preserved

### Breaking Changes
- Requires PyTorch installation
- Different random seed behavior
- Minor numerical differences due to GPU computation

## Performance Expectations

### Speed Improvements
- **CPU**: 2-3x faster (automatic differentiation)
- **GPU**: 10-50x faster (parallel computation)
- **Apple Silicon**: 5-20x faster (Metal acceleration)

### Memory Usage
- Higher base memory (PyTorch overhead)
- Better scaling for large datasets
- Automatic gradient checkpointing available

## Testing Strategy

### Before Migration
```bash
# Run all tests with NumPy implementation
pytest pyAMICA/tests/

# Save reference results
python -m pyAMICA.amica_cli sample_params.json --outdir numpy_reference
```

### After Migration
```bash
# Run with PyTorch implementation
PYTORCH_ENABLE_MPS_FALLBACK=1 pytest pyAMICA/tests/

# Compare results
python compare_implementations.py numpy_reference pytorch_results
```

## Rollback Plan

If issues arise:
1. Keep NumPy implementation in `legacy/` directory
2. Add flag to choose implementation:
   ```python
   AMICA(..., backend='numpy')  # or 'pytorch'
   ```
3. Gradual transition with both available

## Documentation Updates

1. **README.md**
   - Add PyTorch requirement
   - Update installation instructions
   - Add GPU setup guide

2. **Examples**
   - Update all examples to show both CPU and GPU usage
   - Add performance benchmarks

3. **API Documentation**
   - Document device parameter
   - Explain debug/verbose modes
   - Add troubleshooting section

## Timeline

- **Week 1**: Fix current issues, ensure stability
- **Week 2**: Achieve feature parity with NumPy/Fortran
- **Week 3**: Replace NumPy implementation
- **Week 4**: Optimization and documentation

## Success Criteria

1. **Functionality**
   - All tests pass
   - Results match Fortran within tolerance
   - No regression in features

2. **Performance**
   - Faster than NumPy on CPU
   - Significant speedup on GPU
   - Memory usage acceptable

3. **Usability**
   - Clear migration guide
   - Comprehensive documentation
   - Smooth user experience