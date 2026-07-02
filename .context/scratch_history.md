# pyAMICA Scratch History

Debugging notes, failed attempts, and lessons. Prevents repeating dead ends.

## Debugging notes

### Likelihood divergence
- Fortran: iter 1 LL ~ -3.51 -> converges ~ -3.44. Early Python: ~1.42e6 -> ~1.51e6.
- Hypothesis: sign flip or missing `log()`; check `-log(p(x))` vs `log(p(x))`.
- Enhanced model produced positive LLs -> traced to wrong GG normalization constant (see research.md).

### NaN issues
- Occurred ~iter 49-50 (Newton start). `RuntimeWarning: invalid value in divide` at
  `dmu = updates['dmu'] / updates['dalpha']` -> `dalpha` near zero. Add epsilon to denominators.
- Gradient norm jumped from ~70 to ~2e5 at Newton start before clipping was added.

### Fortran structure references
- `get_updates_and_likelihood` (~line 1174), `accum_updates_and_likelihood` (~line 1671),
  `update_params` (~line 1878). Fortran separates accumulation; Python merged it.
- `funmod2.f90` holds special-function helpers (gamln/psifun/etc.), not the PDFs; PDF computation is inline in `amica17.f90`. (`fastlog/fastexp/fastpow` are declared but unused stubs in `amica17_header.f90`.)

## Known differences to watch
- Data layout: Fortran column-major vs Python row-major default (handled; verify edge cases).
- PDF computed inline in Python vs separate module in Fortran.
- Block processing simplified in Python vs Fortran's block-size optimization.
- Fortran 1-based vs Python 0-based indexing; OpenMP -> multiprocessing/joblib; BLAS/LAPACK -> scipy.linalg.

## Quick fixes to try
- Epsilon on all divisions: `dmu / (dalpha + 1e-10)`.
- Verify LL sign: possibly `ll = -sum(log(pdf + 1e-10))`.
- Bound values: `clip(value, -1500, 1500)` (Fortran `minlog`).

## Test data
- Sample data: 32 channels, 30504 samples, 32-bit float. Fortran output in `amicaout/`,
  Python in `pyresults/` (load via `loadmodout()`).

## Performance
- Fortran ~0.02s/iter; Python ~0.5s/iter (~25x slower). Matrix ops likely the bottleneck; profile.

## Debugging commands
```bash
python -m pytest pyAMICA/tests/test_sample_data.py::test_sample_data_light -v -s
python -m pyAMICA.amica_cli pyAMICA/sample_data/sample_params.json --verbose --outdir debug_out
# Fortran build (if needed):
gfortran -O3 -fopenmp amica17.f90 funmod2.f90 -o amica -llapack -lblas
```

## Lessons / check first next time
- [ ] Positive LL almost always means a wrong PDF normalization constant.
- [ ] NaN at a phase transition (e.g. Newton start) points at unclipped gradients or zero denominators.
- [ ] Confirm initialization parity (seed, sphering, starting matrices) before chasing algorithmic bugs.
- [ ] Legacy test signatures drift (`load_data_file` dropped a parameter) - update call sites.
