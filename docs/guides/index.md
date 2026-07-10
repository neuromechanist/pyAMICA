# User Guide

The user guide covers how to run pyAMICA in practice and how its results relate
to the reference implementation.

- **[Backends & Devices](backends.md)** — the available compute backends
  (PyTorch natural-gradient EM, optional MLX, legacy NumPy), device selection
  (CUDA / CPU / MPS), float32 vs float64, and performance guidance on real EEG.
- **[Validation & Parity](validation.md)** — how correctness is defined as
  parity with the Fortran reference, the validation harness, and how
  cross-backend equivalence depends on data adequacy.
