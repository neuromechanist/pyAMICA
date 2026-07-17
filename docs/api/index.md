# API Reference

The public import surface is stable:

```python
from pyAMICA import AMICA, AMICA_NumPy, AMICATorchNG
```

- **[`AMICA`](amica.md)** — the main scikit-learn-style interface. Wraps the
  PyTorch natural-gradient EM backend. Start here.
- **[`AMICATorchNG`](torch-backend.md)** — the PyTorch natural-gradient EM
  backend (Fortran parity). The `AMICA` interface delegates to this class.
- **[`pyAMICA.metrics`](metrics.md)** — separation-quality metrics (`mir`,
  `pairwise_mi`, `block_diagonal_order`) as free functions over plain arrays.
  Also reachable as `AMICA.mir`/`AMICA.pmi` on a fitted model.
- **[`pyAMICA.viz`](viz.md)** — backend-agnostic plots over a written
  `amicaout` directory.
- **[`AMICA_NumPy`](numpy-backend.md)** — the legacy NumPy reference
  implementation, retained as an oracle and for its command-line interface.

The optional Apple-Silicon GPU backend is imported separately and is not part of
the default import surface:

```python
from pyAMICA.mlx_impl import AMICAMLXNG  # requires the `mlx` extra
```

- **[`AMICAMLXNG`](mlx-backend.md)** — the optional Apple-GPU (MLX) backend; the
  fastest option on Apple Silicon (float32).
