# pyAMICA Instructions

## Project Context
**Purpose:** Python implementation of AMICA (Adaptive Mixture Independent Component Analysis) that reproduces the results of the reference Fortran binary. Targets EEG/EMG source separation with GPU/MPS/CPU support.
**Tech Stack:** Python 3.12+, PyTorch (primary backend, MPS/CUDA/CPU), NumPy/SciPy (legacy backend), matplotlib. Reference implementation is Fortran (`amica17.f90`, `funmod2.f90`).
**Architecture:** The scikit-learn-style `AMICA` interface wraps `AMICATorchNG` (`torch_impl/core.py`), the natural-gradient EM port that reaches Fortran parity (Newton, exact-EM mixture updates, symmetric-ZCA sphere, Jacobian LL). This is the single PyTorch backend: the earlier Adam/autograd backends (`AMICATorch`, `AMICATorchV2`) and their mixture/optimizer/PDF helper modules were removed in issue #32 as superseded. The legacy NumPy implementation (`numpy_impl/core.py`, retained as `AMICA_NumPy`) carries the same parity fixes plus baralpha and outlier rejection. Correctness is defined by parity with the Fortran binary, validated by `validate_implementations.py`.

## Architecture Map
```
pyAMICA/
├── amica.py                 # Main scikit-learn-style AMICA interface (wraps AMICATorchNG)
├── __init__.py              # Exposes AMICA (PyTorch), AMICA_NumPy (legacy), numpy_impl, torch_impl
├── torch_impl/              # PyTorch backend
│   ├── core.py              #   Natural-gradient EM port (AMICATorchNG); Fortran-parity, sole backend
│   └── utils.py             #   Preprocessing (sphering, PCA), device selection
├── numpy_impl/              # Legacy NumPy reference (topic-named modules, issue #34)
│   ├── core.py              #   AMICA_NumPy; newton.py, pdf.py, data.py, load.py, viz.py, utils.py, cli.py
│   └── ...
├── amica17.f90, funmod2.f90 # Fortran reference source (read-only, for parity)
├── sample_data/             # Sample EEG data + Fortran binary (amica15mac)
└── tests/                   # Tests, incl. tests/torch_tests/ (vs-Fortran parity)

validate_implementations.py  # Runs both implementations, Hungarian component matching, reports
```
Module names are topic-based (`core`/`newton`/`pdf`/`data`/... under `numpy_impl/`,
`core`/`utils` under `torch_impl/`); the old `pyAMICA.py`/`amica_*.py`/`amica_torch_ng.py`
prefixes were dropped in issue #34. The public import surface is stable:
`from pyAMICA import AMICA, AMICA_NumPy, AMICATorchNG`.

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
- **PyTorch backend:** `pyAMICA/torch_impl/core.py` (`AMICATorchNG`, natural-gradient EM,
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
- Newton and exact-EM updates are implemented in `AMICATorchNG` and the legacy NumPy `numpy_impl/core.py`
  (both Fortran-faithful). Adaptive PDF (#26) is DONE (all five `pdftype` families + ext-Infomax
  switcher); full multi-model matching (#27) is validated by distributional equivalence.
- See `FEATURE_PARITY.md`, `MIGRATION_PLAN.md`, and `PROGRESS_SUMMARY.md` for detailed roadmaps.

## Known Issues (parity blockers)
**Single-model parity: DONE (#24).** The natural-gradient A-update transpose fix (plus exact-EM
mixture updates, digamma rho update, symmetric-ZCA sphere, Jacobian LL) brought both `AMICATorchNG`
and the legacy NumPy `numpy_impl/core.py` to Fortran's solution (LL ~ -3.40, Hungarian-matched component
correlation ~0.997, > 0.95 gate cleared; root cause in `.context/issue-24/`). Also resolved: Newton
stability (posdef, 0 fallbacks), backend consolidation (#32/#31), NumPy CLI save/load format (#30),
NG save/load persistence (#36), and the degenerate-fit contract (#50: the `AMICA` wrapper marks a
degenerate fit unusable via `converged_`/`stop_reason_` and refuses `transform`/`get_*`/`save`,
instead of returning NaN sources).

**Adaptive-PDF selection: DONE (#26).** `AMICATorchNG` now supports all five `amica15.f90`
source-density families via `pdftype`: 0 generalized Gaussian (default, unchanged), 2 Gaussian,
3 logistic, 4 sub-Gaussian cosh+, and the extended-Infomax adaptive switcher (`pdftype=1`, which
flips each source between super-Gaussian code 1 and sub-Gaussian code 4 by kurtosis sign on the
`kurt_start`/`num_kurt`/`kurt_int` schedule). Key correction to the earlier "no oracle" finding: the
validation binary is `amica15mac` = `amica15.f90` (now copied into `pyAMICA/`), which *does*
implement the families; the repo's `amica17.f90` is a later GG-only trim, and the reference binary
was never amica17. The fixed families are bit-exact vs the literal Fortran `z0`/`fp` (~1e-15) and
converge to the binary within ~0.005 LL (Newton-matched). The dynamic `do_choose_pdfs` switch is
dead code even in amica15 (the moment buffers are never accumulated), so the auto-switcher has no
bit-exact oracle and is validated by real-data LL. `pdftype=0` stays the default and is
byte-for-byte unchanged. See `.context/decisions/` and `tests/torch_tests/test_ng_pdf_families.py`.

**Open (non-blocking, tracked):**
- **Multi-model (#27): VALIDATED by distributional equivalence.** Multi-model AMICA is not
  partition-identifiable, so exact partition parity with Fortran is the wrong acceptance bar (the
  `>0.95` cross-corr in #27's title asks the algorithm to be more identifiable than it is). The right
  test is whether the two implementations sample the same distribution over solutions. On an
  N=20-each ensemble (real sample EEG, `n_models=2`), the NG-vs-Fortran partition cross-corr
  distribution is **statistically equivalent to Fortran's own run-to-run distribution** (Mann-Whitney
  p=0.97, TOST equivalent within ±0.05; within-Fortran/within-NG/between all ~0.63-0.64). The
  single-run ~0.64 cross-corr is intrinsic estimator spread, not a defect -- Fortran agrees with
  *itself* at 0.63. See `.context/issue-27/multimodel_distributional_equivalence.md` (+ figure).
  Supporting: per-block sufficient stats are bit-exact vs Fortran; the per-model bias `c` update
  (Fortran `update_c`: `c[i,h] = sum_t v_h*x / sum_t v_h`) is ported to both backends, guarded to a
  no-op for `n_models=1` (single-model parity stays bit-exact), see
  `.context/issue-27/multimodel_c_update.md`.
- **Best-iterate safeguard (#51): DONE.** NG's multi-model LL was ~0.02 lower and ~13x more variable
  than Fortran because `fit` returned the *last* EM iterate under the non-monotone lrate schedule; the
  variance was driven by late Newton-fallback overshoots (one seed peaked at -3.357 then crashed to
  -3.545 in its final iterations). `AMICATorchNG.fit` now returns the highest-LL iterate (`keep_best`,
  default on; `final_ll_` reports the returned iterate's LL, `ll_history` stays the true trajectory).
  At matched 100-iter budget this cuts the LL sd from 12.7x to 2.0x Fortran's; the residual ~0.009
  mean gap is convergence speed, not a worse optimum (at 200 iters NG reaches Fortran's exact mean
  -3.3541, at 300 it exceeds it -- the M-step is bit-exact vs Fortran). Single-model #24 parity stays
  byte-for-byte (monotone => no restore). Inactive under `do_reject`. See ADR 0003 and
  `.context/issue-51/`.

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
