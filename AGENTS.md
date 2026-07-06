# pyAMICA Instructions

## Project Context
**Purpose:** Python implementation of AMICA (Adaptive Mixture Independent Component Analysis) that reproduces the results of the reference Fortran binary. Targets EEG/EMG source separation with GPU/MPS/CPU support.
**Tech Stack:** Python 3.12+, PyTorch (primary backend, MPS/CUDA/CPU), NumPy/SciPy (legacy backend), matplotlib. Reference implementation is Fortran (`amica17.f90`, `funmod2.f90`).
**Architecture:** The scikit-learn-style `AMICA` interface wraps `AMICATorchNG` (`torch_impl/amica_torch_ng.py`), the natural-gradient EM port that reaches Fortran parity (Newton, exact-EM mixture updates, symmetric-ZCA sphere, Jacobian LL). This is the single PyTorch backend: the earlier Adam/autograd backends (`AMICATorch`, `AMICATorchV2`) and their mixture/optimizer/PDF helper modules were removed in issue #32 as superseded. The legacy NumPy implementation (`pyAMICA.py`, retained as `AMICA_NumPy`) carries the same parity fixes plus baralpha and outlier rejection. Correctness is defined by parity with the Fortran binary, validated by `validate_implementations.py`.

## Architecture Map
```
pyAMICA/
├── amica.py                 # Main scikit-learn-style AMICA interface (wraps AMICATorchNG)
├── __init__.py              # Exposes AMICA (PyTorch) and AMICA_NumPy (legacy)
├── torch_impl/              # PyTorch backend
│   ├── amica_torch_ng.py    #   Natural-gradient EM port (AMICATorchNG); Fortran-parity, sole backend
│   └── utils.py             #   Preprocessing (sphering, PCA), device selection
├── pyAMICA.py, amica_*.py   # Legacy NumPy implementation + CLI/data/viz/pdf/newton
├── amica17.f90, funmod2.f90 # Fortran reference source (read-only, for parity)
├── sample_data/             # Sample EEG data + Fortran binary (amica15mac)
└── tests/                   # Tests, incl. tests/torch_tests/ (vs-Fortran parity)

validate_implementations.py  # Runs both implementations, Hungarian component matching, reports
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
- **Main interface:** `pyAMICA/amica.py` (thin wrapper over `AMICATorchNG`)
- **PyTorch backend:** `pyAMICA/torch_impl/amica_torch_ng.py` (`AMICATorchNG`, natural-gradient EM,
  Fortran-parity; ADR `.context/decisions/0001-torch-backend-natural-gradient-em.md`). This is the
  only PyTorch backend; the basic `AMICATorch`/`AMICATorchV2` paths were removed in #32.
- **Validation harness:** `validate_implementations.py`
- **Fortran reference binary:** `pyAMICA/sample_data/amica15mac`
- **Sample data:** `pyAMICA/sample_data/`

## Current Status
- PyTorch backend with GPU/MPS/CPU support; the `AMICATorchNG` natural-gradient EM backend now
  matches the Fortran reference (LL ~ -3.40, component correlation ~0.997) with Newton enabled
  and positive-definite (issue #24).
- Validation harness runs both implementations (NG + NumPy) and matches components via the Hungarian
  algorithm.
- Newton and exact-EM updates are implemented in `AMICATorchNG` and the legacy NumPy `pyAMICA.py`
  (both Fortran-faithful); adaptive PDF (#26) and full multi-model matching (#27) remain open.
- See `FEATURE_PARITY.md`, `MIGRATION_PLAN.md`, and `PROGRESS_SUMMARY.md` for detailed roadmaps.

## Known Issues (parity blockers)
**Single-model parity: DONE (#24).** The natural-gradient A-update transpose fix (plus exact-EM
mixture updates, digamma rho update, symmetric-ZCA sphere, Jacobian LL) brought both `AMICATorchNG`
and the legacy NumPy `pyAMICA.py` to Fortran's solution (LL ~ -3.40, Hungarian-matched component
correlation ~0.997, > 0.95 gate cleared; root cause in `.context/issue-24/`). Also resolved: Newton
stability (posdef, 0 fallbacks), backend consolidation (#32/#31), NumPy CLI save/load format (#30),
and NG save/load persistence (#36).

**Open (non-blocking, tracked):**
- **Adaptive-PDF selection (#26):** `AMICATorchNG` uses a fixed generalized-Gaussian PDF (Fortran
  parity) and lacks the per-source Laplace/Student-t/GG selection the NumPy path has (beyond Fortran
  parity; the NumPy path is the validation oracle). Outlier rejection (`do_reject`) IS implemented.
- **Multi-model (#27):** per-block sufficient stats are bit-exact vs Fortran, but a full 2-model fit
  matches LL yet not Fortran's partition. The per-model bias `c` update (Fortran `update_c`:
  `c[i,h] = sum_t v_h*x / sum_t v_h`, the responsibility-weighted data-space mean) is now ported to
  both `AMICATorchNG` and NumPy `pyAMICA.py`, guarded to a no-op for `n_models=1` (keeps single-model
  parity bit-exact). A controlled A/B (same config/seed, `c` toggled) shows it lifts the 2-model
  Hungarian cross-corr by only ~0.011 with LL comparable (~-3.376 vs -3.375), so the omission was a
  minor contributor:
  the dominant residual gap is **intrinsic partition ambiguity** (NG is self-consistent, cross-corr
  1.0 across block sizes), and `>0.95` vs Fortran is not reachable via `c` alone. See
  `.context/issue-27/multimodel_c_update.md`.

## Development Workflow
1. **Check context:** `.context/plan.md` for current tasks and priorities.
2. **Understand deeply:** `.context/ideas.md` (design), `.context/research.md` (Fortran-vs-Python parity).
3. **Branch:** `gh issue develop <issue-number>` (create an issue first, except minor fixes).
4. **Code:** Follow patterns in `.rules/`.
5. **Test:** Real data only (sample EEG + Fortran binary); see `.rules/testing.md`.
6. **Document failures:** Log dead ends in `.context/scratch_history.md`.
7. **Commit:** Atomic, <50 chars, no emojis, no AI attribution.
8. **PR + review:** Run `/review-pr` and address all findings (`.rules/code_review.md`).
9. **Merge:** CI green first (see below), then **squash merge** (`gh pr merge <n> --squash --delete-branch`).

## [CRITICAL] Core Principles
- **NO MOCKS:** Validate against real sample data and the Fortran binary, never fabricated data. Details: `.rules/testing.md`.
- **No technical debt carried forward:** Address ALL PR review findings; replace, don't deprecate. Details: `.rules/code_review.md`.
- **Numerical parity is the spec:** Correctness means matching Fortran output within tolerance, not merely "converging".
- **Squash merge every PR** by default; use a regular merge commit **only** for PRs coming from an epic branch (to preserve the epic's per-phase history). Never merge until CI is green.

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
