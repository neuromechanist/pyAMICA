# Changelog

Release notes are also published on the
[GitHub releases page](https://github.com/sccn/pyAMICA/releases).

## Unreleased

- Native Fortran run engine (`AMICANative`), the fourth backend alongside NumPy,
  PyTorch and MLX. It runs the AMICA Fortran reference itself and returns an
  `AmicaOutput` with the usual accessors, so it is the parity oracle the Python
  backends are checked against. The reference is now built dependency-free (a
  single-rank MPI shim removes the Open MPI runtime, on top of sccn/amica PR
  \#53's no-MKL recipe; proven identical to real Open MPI at machine epsilon) and
  released as a self-contained binary for macOS arm64, Linux x64/arm64 and Windows
  x64 (Windows arm64 runs the x64 binary via emulation until a native toolchain
  exists, issue #173). The binary is resolved for the host and downloaded from the
  release on first use (SHA-256 verified); `python -m pamica.native` installs it
  explicitly, or set `PAMICA_NATIVE_BINARY` to a local build (epic #165).
- Fixed `loadmodout` reading `W`, `sbeta` and `rho` in the wrong byte order:
  it used C order where the writer, genuine Fortran output and EEGLAB's
  `loadmodout15.m` all use column-major (F order). The consequence was that
  `AmicaOutput.W` came back transposed, silently corrupting genuine Fortran
  output and everything derived from it (`A`, `svar`, `origord`), and
  `sbeta`/`rho` were scrambled whenever `num_mix > 1` (the default). A
  write-then-read round trip cancels the error, so no self-consistency test
  could catch it; the fix is pinned by recomputing the bundled Fortran
  fixture's own reported log-likelihood from the loaded parameters (an external
  oracle). The writer's multi-model `W` layout, which interleaved models and was
  not EEGLAB-readable, is corrected to genuine Fortran (model axis slowest);
  single-model output is byte-identical to before. `AmicaOutput` gains a
  supported `sources(X, model=0)` accessor (the loaded-fit counterpart of the
  live model's `transform`) so downstream source derivations no longer hand-roll
  the sphere/unmixing composition (#159). Migration note: a *multi-model*
  `amicaout` directory written by an earlier pamica (whose `W` used the old
  model-interleaved layout) must be regenerated with `write_amica_output`, not
  just re-loaded; there is no version marker to detect the old layout (genuine
  Fortran output carries none either), and the pre-fix multi-model `W` was never
  in the correct convention regardless. Single-model directories are unaffected
  (byte-identical before and after).
- Separation-quality metrics (`pamica.metrics`): `mir` (Mutual Information
  Reduction, in nats) measures how much mutual information a fitted unmixing
  removes from the data. A direct port of `getMIR.m` from bigdelys/pre_ICA_cleaning
  (Apache-2.0; see `THIRD_PARTY_NOTICES.md`), verified against the original at
  1.7e-15 relative on the bundled sample EEG (#134).
- `pairwise_mi` and `block_diagonal_order` (`pamica.metrics`): the pairwise
  mutual-information matrix between fitted sources, plus a greedy
  nearest-neighbour-chain ordering that clusters dependent components near the
  diagonal. A clean-room reimplementation: the reference (`minfojp.m` in
  postAmicaUtility) is GPL-2.0-or-later and pamica is BSD-3-Clause, so its
  source was never read. Agrees with that reference at r=0.9887 on identical
  signals (#135).
- `LLt` output parity with the Fortran reference: both backends now write the
  per-timepoint, per-model log-likelihood file that the reference binary
  produces on every run, and `loadmodout` reads it with the correct column-major
  layout (it previously used C order, scrambling `Lht`/`Lt`). Verified
  bit-exactly in both directions against EEGLAB's real `loadmodout15.m`. Under
  `do_reject`, rejected samples are written as exactly `0.0`, matching Fortran:
  those zeros are load-bearing, since its `load_rej` reconstructs the rejection
  mask from them (#155).
- `AMICATorchNG`/`AMICA` gain `mir()`/`pmi()` accessors that compose the
  fitted unmixing the documented way (`get_unmixing_matrix(model_idx) @
  sphere` for MIR, `transform(X, model_idx)` for PMI) and delegate to
  `pamica.metrics.mir`/`pairwise_mi`, so callers no longer hand-compose the
  transform themselves. `fit()` also accepts `mir_step` (default `0`, off) to
  record MIR waypoints during training in `mir_history_` as
  `(iteration, mir_nats, variance)`; like `ll_history_`, it is a true
  trajectory that a `keep_best` restore does not rewrite. PCA reduction
  (`pcakeep`/`pcadb`) is rejected up front with a named error, since it
  leaves the sphere rank-deficient and MIR's log-Jacobian undefined (#137).
- Visualization module (`pamica.viz`): `plot_pmi_heatmap` and
  `plot_model_probability`, backend-agnostic views over `AmicaOutput` that return
  a `Figure` (and accept an optional `ax`/`axes`) rather than mutating pyplot
  global state, plus `read_eeglab_set_metadata` for the sample rate pamica
  itself has no notion of. Both plots are verified against the MATLAB reference:
  the smoothed model probability matches `smooth_amica_prob` at r=0.9886, and
  `pairwise_mi` matches `minfojp` at r=0.9887 (#136).
- Fixed `numpy_impl.pdf.compute_pdf` using `gammaln` where the generalized
  Gaussian needs `gamma`, which made the returned density negative for every
  `rho` outside the special-cased 1 and 2 (it integrated to -8.82 at the default
  `rho0=1.5`). Affected `numpy_impl.viz.plot_pdf_fits`; the fit path was never
  affected, as it uses its own log-space implementation (#136).

## 0.1.2

Outlier-rejection parity in the NumPy backend, repo-wide type-checking, and the
full validation-evidence documentation.

- NumPy backend outlier rejection: the Fortran `do_reject` outlier-rejection path
  is ported to `AMICA_NumPy` via the same `good_idx` mechanism as the PyTorch
  backend, so the NumPy reference now drops per-sample outliers on the
  `rejstart`/`rejint`/`maxrej` schedule (#123).
- Rejection robustness: a non-finite log-likelihood is now distinguished from an
  over-aggressive `rejsig`, so an over-tight rejection threshold fails with a
  clear message instead of a silent non-finite result (#127).
- Type checking enforced: repo-wide `ty` diagnostics fixed (496 to 0) and `ty`
  added to CI alongside a pre-commit config (ruff + ty) (#124, #125).
- Documentation: the validation guide is expanded into a full evidence page,
  source-density bit-exactness, cross-platform device/precision invariance
  (cross-backend equivalence matrix and IC topomaps), the EEGLAB drop-in
  round-trip, and the other validated behaviors (#108).

## 0.1.1

Validation-methodology and correctness fixes since 0.1.0.

- Amari distance: a second, permutation- and scale-invariant unmixing-matrix
  comparison metric (Amari, Cichocki & Yang 1996) alongside Hungarian-matched
  correlation, used throughout the Fortran-parity validation (#120).
- Multi-model equivalence test: switched to a valid run-level permutation test
  that respects the dependence among the 40 runs' pairwise correlations, instead
  of a pseudoreplicated Mann-Whitney/TOST (#115).
- Parity and performance tables added to the paper, with the full results,
  native-Fortran CPU core-scaling rows, and per-run detail in the docs (#112).
- Type-safety fixes in `validate_implementations.py` (`run_fortran_amica`
  return type, `load_eeglab_data` dtype annotation) (#118).
- JOSS draft-PDF build workflow, `.zenodo.json` with ROR-based citation
  metadata, and an MLX backend API reference page (#110, #105, #107).
- Corrected a stale float32-speedup claim and added a funding acknowledgement
  (#114).

## 0.1.0

First public release.

- PyTorch natural-gradient EM backend (`AMICATorchNG`) at Fortran parity on real
  EEG (single-model log-likelihood ~ -3.40, Hungarian-matched component
  correlation ~ 0.997).
- Backends: CPU, NVIDIA GPU (CUDA), and Apple GPU (MLX); float64 for parity,
  float32 for speed.
- All five source-density families, mixture of ICA models, Newton updates,
  component sharing, and outlier rejection.
- EEGLAB drop-in output: `write_amica_output` writes the `loadmodout15` format,
  and `variance_order` gives the EEGLAB back-projected-variance component order.
- Spatially-distributed channel-subset selection and a data-size (k-factor)
  cross-backend equivalence sweep for the benchmarks.
- scikit-learn-style `AMICA` interface, save/load, and a documentation site.
