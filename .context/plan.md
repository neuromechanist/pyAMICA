# pyAMICA Development Plan

## Project Overview
**Goal:** Python (PyTorch) implementation of AMICA that reproduces the Fortran binary's results within tolerance.
**Stack:** Python 3.12+, PyTorch (MPS/CUDA/CPU), NumPy/SciPy, Fortran reference.
**Definition of done:** Component correlation >0.95 with Fortran, LL within ~1% of Fortran, no NaN/Inf, runtime within ~2-3x of Fortran.

## Development Tasks
<!-- Status markers: [ ] pending, [~] in progress, [x] complete -->

### Priority 1: Parity blockers — DONE (#24)
- [x] Fix likelihood sign/scaling vs Fortran (the ~13x factor was a pre-parity basic-backend
      artifact; NG uses Jacobian LL, single-model LL ~ -3.40 vs Fortran -3.4018)
- [x] Stabilize Newton optimization (ported from NumPy, positive-definite, 0 fallbacks on sample data)
- [x] Add numerical-stability bounds from Fortran (`mincond`, `minlog`, `maxdble`, `mineig` present;
      regression tests still owed, see Priority 3)
- [x] Add NaN/Inf checks (degenerate-fit contract #50 refuses NaN output instead of suppressing it)
- [x] Ensure identical initialization vs Fortran (symmetric-ZCA sphere, seed, starting mixing matrix)
- [x] Raise component correlation with Fortran to >0.95 (now ~0.997, Hungarian-matched)

### Priority 2: Missing core features
- [x] Port outlier rejection (`do_reject` / `reject_data`) to the torch backend (done in `AMICATorchNG`)
- [x] Wire up / stabilize Newton for the torch backend (done in `AMICATorchNG`, posdef, issue #24)
- [x] Adaptive PDF selection for `AMICATorchNG` (issue #26). Corrected the "no oracle" finding:
      the reference binary is `amica15mac` = `amica15.f90` (now copied into `pyAMICA/`), which
      implements five `pdtype` density families; the repo's `amica17.f90` is a later GG-only trim.
      Ported all five (0 GG default, 2 Gaussian, 3 logistic, 4 sub-G cosh+, 1 super-G cosh-) plus the
      extended-Infomax kurtosis auto-switcher (`pdftype=1`). Fixed families are bit-exact vs the
      literal Fortran `z0`/`fp` and converge within ~0.005 LL of the binary; the dynamic switcher is
      dead code in the binary (no bit-exact oracle) so it is LL-validated. `pdftype=0` default
      unchanged. Tests in `tests/torch_tests/test_ng_pdf_families.py`.
- [x] Multi-model AMICA per-model bias `c` update (issue #27): ported to both backends, guarded
      no-op for `n_models=1`; controlled A/B shows +0.011 cross-corr, gap is intrinsic partition
      ambiguity (see `.context/issue-27/multimodel_c_update.md`).
- [x] Degenerate-fit contract (issue #50): the `AMICA` wrapper no longer marks a degenerate fit
      (`stop_reason` nan_ll/singular_ll) as usable. `fit` sets `is_fitted_` only on a converged fit
      and exposes `converged_`/`stop_reason_`; `transform`/`get_mixing_matrix`/`get_unmixing_matrix`/
      `save` raise a clear degenerate error (mirroring `state_dict`) instead of returning NaN sources.
- [x] Best-iterate safeguard (issue #51): `AMICATorchNG.fit` returns the highest-LL iterate
      (`keep_best`, `final_ll_`), not the last, so a late Newton-fallback overshoot no longer leaves
      the model below a peak it reached. Root cause was return-last, not a bad basin (the sole
      variance-driving seed peaked at -3.357 then crashed to -3.545 in its final iterations).
      Single-model #24 parity stays bit-exact (monotone => no restore). See ADR 0003,
      `.context/issue-51/`.
- [x] Component sharing (issue #60): `share_comps` multi-model reassignment ported to
      `AMICATorchNG` (de-sphered cosine-similarity merge, `share_start`/`share_iter`/`comp_thresh`
      schedule + A-freeze). OFF by default so single-model (#24)/default multi-model (#27) parity is
      byte-for-byte. No bit-exact oracle (reference `Spinv2` metric is dead code, like #26);
      behavior-validated. See `tests/torch_tests/test_ng_sharing.py`.

### Priority 3: Testing & validation
- [x] Real-data test suite exercising the PyTorch backend end-to-end (issue #7) - Phase 1, issue #10:
      suite collects cleanly and passes (35 passed, 6 xfailed for documented parity/algorithm
      issues, 0 errors); see `.context/phase1_baseline.md`
- [x] Integration tests comparing against Fortran outputs (`tests/torch_tests/`) - Phase 1, issue #10
- [ ] Numerical-stability regression tests (mincond/minlog/maxdble/mineig)
- [ ] Edge cases (single channel, single sample)
- [ ] `AMICA.save`/`load` and `plot_components` coverage (issue #15)
- [ ] Performance benchmark: NG runtime vs Fortran binary (verify the 2-3x criterion)

### Infrastructure / migration
- [x] Migrate environment from conda `torch-312` to UV; declare the PyTorch stack in `pyproject.toml`
      (Phase 1, issue #10: `torch`, `pytest`, `pytest-cov` added to `pyproject.toml`/`uv.lock`,
      `.python-version` pinned to 3.12 for torch/MPS compatibility)
- [x] Set up CI (see `.rules/ci_cd.md`): ruff lint/format, pytest (excluding slow/Fortran-binary
      parity), build + clean-env import matrix on Python 3.12/3.13; typos check. Green on `main`.
- [x] Fix legacy test function signatures (`load_data_file` extra parameter) - Phase 1, issue #10

## Success Criteria
- [x] Component correlation > 0.95 with Fortran (~0.997, single model)
- [x] LL convergence within 1% of Fortran (LL ~ -3.40 vs -3.4018)
- [x] No NaN/Inf during optimization (degenerate-fit contract #50)
- [ ] Runtime within 2-3x of Fortran (unmeasured; benchmark owed)
- [x] Real (non-mock) test suite green

## Notes
- Correctness is defined by parity with the Fortran binary, not by convergence alone.
- Detailed feature status: `feature_parity.md`; migration record: `migration_plan.md`.

## Release Readiness (JOSS track) — started 2026-07-10
Endgame ordering: finish the three open benchmark issues -> documentation ->
transfer to github.com/sccn -> JOSS paper. The repo will move to
`github.com/sccn/pyAMICA` (full transfer, preserving issues/PRs/history) and docs
will be hosted at `eeglab.org/pyAMICA`.

### Phase R1: Benchmark completion (#90, #91, #92)
- [~] #90 Data-size (k-factor) frames sweep at 70ch. Code + data staged on hallu
      (branch `feature/issue-90-datasize-sweep`, origin-pushed; based on current main).
      Full 747,750-frame npy present (`ds002718_sub-002_eeg70_full.npy`). CUDA sweep
      (torch-cuda-f64/f32) + native-fortran-f64 launched on hallu 2026-07-10:
      frames 73.5k/147k/294k/490k/747.75k -> k=15/30/60/100/152, 2000 iters,
      out=`benchmarks/results_k90_hallu`. Probe: largest frames ~1.1 s/it (f64) ->
      ~36 min/run; CUDA sweep ETA ~2.5 h. Remaining: finish runs -> `--compare`
      cross-backend |corr| vs k figure + report -> PR.
- [ ] #91 Spatially-distributed channel subsets: replace `full[:nc]` first-N slicing
      in `benchmark_dimsweep.py`/`benchmark_decompose.py` with farthest-point sampling
      over real electrode 3D coords (whole-head 16/32/48ch montages). Local; formalize
      `mne` as a viz/benchmark extra (not in core env). Prereq for #92 reduced-montage.
- [ ] #92 EEGLAB drop-in output parity: variance-ordered ICs (back-projected variance,
      IC1=highest), `loadmodout15`/`pop_runamica`-readable output, sign/scale
      conventions, documented MATLAB+EEGLAB round-trip. MATLAB R2025b and EEGLAB both
      present locally (`~/Documents/git/eeg/eeglab`).

### Phase R2: Documentation (MkDocs Material, per /project:init-project)
Use the init-project docs templates verbatim where possible
(`~/.claude/plugins/cache/research-skills/project/0.5.0/templates/config/mkdocs.yml`
+ `github/workflows/docs.yml`), adapted for pyAMICA:
- Material theme (light/dark palette toggle, navigation.tabs/sections/indexes/top/
  instant, search.suggest/highlight, content.code.copy, toc.integrate).
- Plugins: `search`, `mkdocstrings` (python) with **`docstring_style: numpy`** (repo
  uses numpy docstrings, NOT the template's `google` default), `git-revision-date-localized`.
- Add a `docs` optional-dependency extra (mkdocs-material, mkdocstrings[python],
  mkdocs-git-revision-date-localized-plugin) — the `docs.yml` workflow runs
  `uv sync --extra docs` then `uv run mkdocs build`. Currently only an `mlx` extra exists.
- **Hosting:** GitHub Pages via `docs.yml` (build+deploy on push to main). Served at
  `eeglab.org/pyAMICA` because `sccn/pyAMICA` *project* Pages inherit the sccn org
  Pages custom domain (`eeglab.org`) at the `/pyAMICA` subpath. `site_url:
  https://eeglab.org/pyAMICA/`, relative links; stages at
  `neuromechanist.github.io/pyAMICA/` pre-transfer.
- [ ] De-WIP `README.md` (drop the "do not rely on this" disclaimer; `uv` install;
      quickstart; backend-selection guide MLX/CUDA/CPU + f32/f64; results table).
- [ ] mkdocs.yml + docs/ skeleton (Home, Getting Started, User Guide, API Reference
      via mkdocstrings, Development, Changelog) + `docs` extra + `docs.yml` workflow.
- [ ] Community health: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `CITATION.cff`,
      issue/PR templates (JOSS-expected).

### Phase R3: Transfer to github.com/sccn (before JOSS submission)
- [ ] GitHub repo transfer (preserves issues/PRs/stars/history; auto-redirects old
      URLs). Post-transfer: update badge/repo URLs (README, CITATION.cff, paper.md),
      wire up the `eeglab.org/pyAMICA` docs deploy target.

### Phase R4: JOSS paper (/manuscript:manuscript-writing)
- [ ] `paper.md` (~1000 words) + `paper.bib`: summary, statement of need (GPU +
      cross-platform AMICA with Fortran parity; drop-in for EEGLAB AMICA), comparison
      vs EEGLAB AMICA / Picard / FastICA, backend + parity results, acknowledgments;
      `repository -> github.com/sccn/pyAMICA`.
