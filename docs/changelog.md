# Changelog

Release notes are also published on the
[GitHub releases page](https://github.com/sccn/pyAMICA/releases).

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
