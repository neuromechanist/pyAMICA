# MLX backend (AMICAMLXNG)

The optional Apple-Silicon GPU backend. It runs the natural-gradient EM E/M-step
on the Apple GPU in float32 (Apple GPUs have no float64), with the small
per-iteration linear algebra on MLX's CPU stream. It supports single- and
multi-model generalized-Gaussian (`pdftype=0`) natural-gradient AMICA and is the
fastest option on Apple hardware; see [Backends & Devices](../guides/backends.md)
for the performance comparison.

MLX is an optional dependency (Apple Silicon only), so it is imported separately
and is not part of the default `import pyAMICA` surface:

```python
from pyAMICA.mlx_impl import AMICAMLXNG  # requires the `mlx` extra
```

Install it with `uv pip install mlx` or the `mlx` extra (`pip install pyAMICA[mlx]`).
Because it computes in float32, use the [PyTorch backend](torch-backend.md) on
CUDA/CPU for float64 Fortran-parity runs.

::: pyAMICA.mlx_impl.AMICAMLXNG
