# pyAMICA Instructions

## Project Context
**Purpose:** Python implementation of AMICA (Adaptive Mixture Independent Component Analysis) that reproduces the results of the reference Fortran binary. Targets EEG/EMG source separation with GPU/MPS/CPU support.
**Tech Stack:** Python 3.12+, PyTorch (primary backend, MPS/CUDA/CPU), NumPy/SciPy (legacy backend), matplotlib. Reference implementation is Fortran (`amica17.f90`, `funmod2.f90`).
**Architecture:** The scikit-learn-style `AMICA` interface selects a PyTorch backend via `backend=`. `backend="torch"` (default) wraps the basic `AMICATorch` (`torch_impl/amica_torch.py`), which has Newton disabled and no adaptive PDF. `backend="ng"` wraps `AMICATorchNG` (`torch_impl/amica_torch_ng.py`), the natural-gradient EM port that reaches Fortran parity (Newton, exact-EM mixture updates, symmetric-ZCA sphere, Jacobian LL). The earlier `AMICATorchV2` (`torch_impl/amica_torch_v2.py`) is parked/superseded by `AMICATorchNG`. The legacy NumPy implementation (`pyAMICA.py`, retained as `AMICA_NumPy`) carries the same parity fixes plus baralpha and outlier rejection. Correctness is defined by parity with the Fortran binary, validated by `validate_implementations.py`.

## Architecture Map
```
pyAMICA/
├── amica.py                 # Main scikit-learn-style AMICA interface (PyTorch-backed)
├── __init__.py              # Exposes AMICA (PyTorch) and AMICA_NumPy (legacy)
├── torch_impl/              # PyTorch backend
│   ├── amica_torch.py       #   Core AMICA module (basic; Newton disabled) - backs the default AMICA (backend="torch")
│   ├── amica_torch_ng.py    #   Natural-gradient EM port (AMICATorchNG); Fortran-parity, backs backend="ng"
│   ├── amica_torch_v2.py    #   PARKED/superseded by amica_torch_ng.py (experimental, not wired into AMICA)
│   ├── adaptive_pdf.py      #   Adaptive PDF selection (Laplace/Student-t/GG)
│   ├── newton_optimizer.py  #   Newton optimization with Fortran-style ramping
│   ├── mixture_models.py    #   Mixture-of-densities components
│   ├── optimizers.py        #   Natural-gradient / optimizer utilities
│   ├── fortran_output.py    #   Fortran-style debug/log output
│   └── utils.py             #   Preprocessing (sphering, PCA), device selection
├── pyAMICA.py, amica_*.py   # Legacy NumPy implementation + CLI/data/viz/pdf/newton
├── amica17.f90, funmod2.f90 # Fortran reference source (read-only, for parity)
├── sample_data/             # Sample EEG data + Fortran binary (amica15mac)
└── tests/                   # Tests, incl. tests/torch_tests/ (vs-Fortran parity)

validate_implementations.py  # Runs both backends, Hungarian component matching, reports
```

## Environment Setup
Canonical environment is **UV** (per global standards). The PyTorch stack is declared in
`pyproject.toml` with `uv.lock` pinned; the legacy conda env (`torch-312`) is retired. Migration
history is in `.context/plan.md`.
```bash
uv sync                          # Install dependencies
uv run pytest                    # Run tests
uv run ruff check --fix . && uv run ruff format .
```
MPS note: run with `PYTORCH_ENABLE_MPS_FALLBACK=1` for ops MPS does not yet support. The NG backend
computes in float64 for Fortran parity, which MPS cannot represent, so parity runs use CPU or CUDA
(the `AMICA` wrapper falls back to CPU automatically when a device is not pinned).

## Key Files
- **Main interface:** `pyAMICA/amica.py` (`backend="torch"` default, `backend="ng"` opt-in)
- **Default backend:** `pyAMICA/torch_impl/amica_torch.py` (`AMICATorch`, Newton disabled).
- **Parity backend:** `pyAMICA/torch_impl/amica_torch_ng.py` (`AMICATorchNG`, natural-gradient EM,
  Fortran-parity; ADR `.context/decisions/0001-torch-backend-natural-gradient-em.md`). Parked/superseded: `amica_torch_v2.py` (`AMICATorchV2`)
- **Validation harness:** `validate_implementations.py`
- **Fortran reference binary:** `pyAMICA/sample_data/amica15mac`
- **Sample data:** `pyAMICA/sample_data/`

## Current Status
- PyTorch backend with GPU/MPS/CPU support; the `AMICATorchNG` natural-gradient EM backend now
  matches the Fortran reference (LL ~ -3.40, component correlation ~0.997) with Newton enabled
  and positive-definite (issue #24).
- Validation harness runs both backends and matches components via the Hungarian algorithm.
- Newton and exact-EM updates are implemented in `AMICATorchNG` and the legacy NumPy `pyAMICA.py`
  (both Fortran-faithful); adaptive PDF and multi-model remain experimental (`AMICATorchV2`).
- See `FEATURE_PARITY.md`, `MIGRATION_PLAN.md`, and `PROGRESS_SUMMARY.md` for detailed roadmaps.

## Known Issues (parity blockers)
The core parity blockers were RESOLVED by issue #24 (natural-gradient A-update transpose fix +
exact-EM mixture updates + digamma rho update + symmetric-ZCA sphere + Jacobian LL). Both the
`AMICATorchNG` backend and the legacy NumPy `pyAMICA.py` now ascend to Fortran's solution
(LL ~ -3.40, Hungarian-matched component correlation ~0.997 with Newton, > 0.95 gate cleared).
Root-cause writeup and machine-precision repro: `.context/issue-24/root_cause_Aupdate.py` +
`drift_localization.md`.

Multi-model (`n_models>1`): the per-block sufficient statistics and model responsibilities are
bit-exact vs Fortran (one seeded step matches unmixing to ~1e-15, including `gm`) and the sphere
matches to ~1e-13; NG<->NumPy multi-model sufficient-stat parity is a suite test. A full 2-model fit
reaches a comparable LL (NG -3.355 vs Fortran -3.345) but a lower Hungarian cross-correlation
(~0.77). Two contributors, being separated in issue #27: (1) intrinsic partition ambiguity --
multi-model AMICA has many near-degenerate partitions (unlike single-model's unique solution), and
NG is self-consistent (cross-corr 1.0 across block sizes); (2) a genuine code gap -- the per-model
bias `c` update is omitted (only a no-op for `n_models=1`), whereas Fortran moves `c` per model when
`v` is non-uniform. So the multi-model gap is NOT purely partition ambiguity; the `c` update is an
open, fixable suspect.

Remaining, non-blocking (tracked as follow-up issues):
1. Newton stability was fixed for `AMICATorchNG` (posdef, 0 fallbacks on the sample data). The
   separate experimental `AMICATorchV2` is parked/superseded and scheduled for removal (#32).
2. The `AMICATorchNG` backend still lacks the adaptive-PDF selection present in the NumPy path
   (a feature beyond Fortran parity, which uses a fixed GG PDF; #26); outlier rejection IS
   implemented (`do_reject`).
3. Full multi-model partition matching and the per-model bias `c` update (#27).
4. Legacy `backend="torch"` bugs: mixture M-step LL descent (#31), CLI save/load format (#30).

## Development Workflow
1. **Check context:** `.context/plan.md` for current tasks and priorities.
2. **Understand deeply:** `.context/ideas.md` (design), `.context/research.md` (Fortran-vs-Python parity).
3. **Branch:** `gh issue develop <issue-number>` (create an issue first, except minor fixes).
4. **Code:** Follow patterns in `.rules/`.
5. **Test:** Real data only (sample EEG + Fortran binary); see `.rules/testing.md`.
6. **Document failures:** Log dead ends in `.context/scratch_history.md`.
7. **Commit:** Atomic, <50 chars, no emojis, no AI attribution.
8. **PR + review:** Run `/review-pr` and address all findings (`.rules/code_review.md`).

## [CRITICAL] Core Principles
- **NO MOCKS:** Validate against real sample data and the Fortran binary, never fabricated data. Details: `.rules/testing.md`.
- **No technical debt carried forward:** Address ALL PR review findings; replace, don't deprecate. Details: `.rules/code_review.md`.
- **Numerical parity is the spec:** Correctness means matching Fortran output within tolerance, not merely "converging".

## [NEVER DO THIS]
- Never use mocks, stubs, or fake/synthetic data as the basis for correctness tests.
- Never use `pip`, `conda`, or `virtualenv`; use UV for Python.
- Never commit secrets, `.env` files, or credentials.
- Never leave empty catch blocks or silent failures (this codebase already has NaN-suppression risks).
- Never add backward-compatibility shims; replace directly.
- Never add a TODO without a linked issue.
- Never use emojis in commits, PRs, or code.

## [REFERENCE] Rules Directory
- `.rules/testing.md` - NO MOCK policy, coverage
- `.rules/python.md` - UV, ruff, ty
- `.rules/git.md` - Commit/branch conventions
- `.rules/code_review.md` - PR review toolkit and checklist
- `.rules/ci_cd.md` - GitHub Actions setup
- `.rules/documentation.md` - Docs conventions
- `.rules/self_improve.md` - Learning/pattern extraction
- `.rules/serena_mcp.md` - Serena MCP code intelligence (when available)

## Context Files
- `.context/plan.md` - Current tasks, phases, priorities
- `.context/research.md` - Fortran-vs-Python parity analysis (data structures, subroutines, divergence causes)
- `.context/ideas.md` - PyTorch design decisions and library options
- `.context/scratch_history.md` - Debugging notes, failed attempts, lessons
- `.context/decisions/` - Architecture Decision Records (copy `0000-template.md` to start one)

## Project Docs (top-level)
- `FEATURE_PARITY.md` - Feature comparison and implementation roadmap
- `MIGRATION_PLAN.md` - NumPy to PyTorch migration timeline
- `PROGRESS_SUMMARY.md` - Achievements and validation metrics snapshot
- `README.md` - Overview and quick start

---
Remember: parity with the Fortran reference is the definition of done. Check `.rules/` for detailed guidance.
