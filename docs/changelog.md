# Changelog

Release notes are also published on the
[GitHub releases page](https://github.com/sccn/pyAMICA/releases).

## Unreleased

- Visualization module (`pyAMICA.viz`): `plot_pmi_heatmap`, `plot_model_probability`,
  and `plot_topo_pdf`, backend-agnostic views over `AmicaOutput` that return a
  `Figure` (and accept an optional `ax`/`axes`) rather than mutating pyplot
  global state, plus `read_eeglab_set_metadata` for the sample rate and channel
  positions pyAMICA itself has no notion of (#136).

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
