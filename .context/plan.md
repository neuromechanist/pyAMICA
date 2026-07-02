# pyAMICA Development Plan

## Project Overview
**Goal:** Python (PyTorch) implementation of AMICA that reproduces the Fortran binary's results within tolerance.
**Stack:** Python 3.12+, PyTorch (MPS/CUDA/CPU), NumPy/SciPy, Fortran reference.
**Definition of done:** Component correlation >0.95 with Fortran, LL within ~1% of Fortran, no NaN/Inf, runtime within ~2-3x of Fortran.

## Development Tasks
<!-- Status markers: [ ] pending, [~] in progress, [x] complete -->

### Priority 1: Parity blockers
- [~] Fix likelihood sign/scaling vs Fortran (~13x factor; GG normalization corrected, needs revalidation)
- [~] Stabilize Newton optimization (NaN at Newton-start iter; clipping + Fortran-matched ramp added, needs revalidation)
- [ ] Add numerical-stability bounds from Fortran (`mincond=1e-15`, `minlog=-1500`, `maxdble=1e32`, `mineig=1e-15`)
- [ ] Add NaN/Inf checks and epsilon to all divisions (notably `dmu / dalpha`)
- [ ] Ensure identical initialization vs Fortran (seed, sphering/whitening, starting mixing matrix)
- [ ] Raise component correlation with Fortran (~0.46-0.9, run-dependent) to >0.95

### Priority 2: Missing core features
- [ ] Port outlier rejection (`do_reject` / `reject_data`) to the torch backend (exists in legacy NumPy `pyAMICA.py`)
- [ ] Wire up / stabilize Newton for the torch backend (`baralpha`, full Hessian, convergence checks; exists in legacy NumPy)
- [~] Adaptive PDF selection (`do_choose_pdfs`) - in `amica_torch_v2.py`, not yet wired into the default `AMICA` interface
- [ ] Multi-model AMICA - framework exists, needs testing
- [ ] Component sharing

### Priority 3: Testing & validation
- [ ] Real-data test suite exercising the PyTorch backend end-to-end (issue #7; currently unverified and expected to fail)
- [ ] Integration tests comparing against Fortran outputs (`tests/torch_tests/`)
- [ ] Numerical-stability regression tests
- [ ] Edge cases (single channel, single sample)

### Infrastructure / migration
- [ ] Migrate environment from conda `torch-312` to UV; declare the PyTorch stack in `pyproject.toml`
- [ ] Set up CI (see `.rules/ci_cd.md`)
- [ ] Fix legacy test function signatures (`load_data_file` extra parameter)

## Success Criteria
- [ ] Component correlation > 0.95 with Fortran
- [ ] LL convergence within 1% of Fortran
- [ ] No NaN/Inf during optimization
- [ ] Runtime within 2-3x of Fortran
- [ ] Real (non-mock) test suite green

## Notes
- Correctness is defined by parity with the Fortran binary, not by convergence alone.
- Detailed feature status: `../FEATURE_PARITY.md`; migration timeline: `../MIGRATION_PLAN.md`.
