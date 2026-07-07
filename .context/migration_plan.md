# Migration Plan: NumPy to PyTorch (COMPLETE)

**Status: complete.** This file is kept as a historical record. The NumPy-to-PyTorch migration
landed via the natural-gradient EM backend (`AMICATorchNG`); the PyTorch path is the default and
reaches Fortran parity. The NumPy implementation is retained as an oracle and CLI, not as a
fallback. Current status lives in `../AGENTS.md` (Known Issues) and `progress_summary.md`.

## What the migration delivered

- **Default backend is PyTorch.** `from pyAMICA import AMICA` wraps `AMICATorchNG`. The legacy
  reference is exposed separately as `AMICA_NumPy` and is not selected via a `backend=` flag; there
  is no `legacy/` directory and no runtime backend switch.
- **Feature parity.** Natural gradient + Newton (positive-definite, with Fortran-style ramping),
  exact-EM mixture updates, symmetric-ZCA sphere, Jacobian LL, outlier rejection (`do_reject`), all
  five `pdftype` density families (#26), and multi-model support with the per-model bias `c` update
  (#27). Single-model results match the Fortran fixed point (LL ~ -3.40, component correlation
  ~0.997).
- **Device support.** Automatic CUDA / MPS / CPU selection. The NG backend computes in float64 for
  Fortran parity, which MPS cannot represent, so parity runs use CPU or CUDA; the `AMICA` wrapper
  falls back to CPU automatically when a device is not pinned. Run with
  `PYTORCH_ENABLE_MPS_FALLBACK=1` for ops MPS does not support.
- **Environment.** Migrated from the conda `torch-312` env to UV; the stack is declared in
  `pyproject.toml` with `uv.lock` pinned and `.python-version` set to 3.12.
- **Validation.** `validate_implementations.py` runs both backends on real sample EEG plus the
  Fortran binary and matches components via the Hungarian algorithm. No mocks or synthetic data.

## Not carried over from the original plan

- **`backend='numpy'` runtime switch / `legacy/` dir / gradual dual-backend transition.** Dropped.
  The two implementations are separate public classes, not a runtime toggle.
- **Adam / L-BFGS / autograd "bonus" optimizers.** The basic `AMICATorch` and `AMICATorchV2`
  (Adam/autograd) backends were removed in #32; `AMICATorchNG` (natural-gradient EM) is the only
  PyTorch backend.

## Still open (tracked outside this plan)

- Performance benchmark vs Fortran (the "2-3x runtime" criterion is unmeasured).
- Component sharing (`share_comps`), the last unimplemented Fortran feature.
- Edge-case + numerical-stability regression tests; `save`/`load` and `plot_components` coverage
  (issue #15).
