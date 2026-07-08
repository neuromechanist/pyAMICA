"""MLX backend for pyAMICA (Apple-Silicon GPU).

Mirrors ``torch_impl`` but targets Apple's MLX array framework. MLX is an
*optional* dependency (Apple Silicon only); importing this subpackage requires
``mlx`` to be installed, so the top-level ``pyAMICA`` package never imports it
eagerly. Install with ``uv pip install mlx`` (or the ``mlx`` extra).

The backend (:class:`AMICAMLXNG`) runs the natural-gradient EM E/M-step on the
Apple GPU in float32 (Apple GPUs have no FP64), with the small per-iteration
linear algebra on MLX's CPU stream (issue #76, epic #74 Phase C). It is a v1
MVP: single-model, generalized-Gaussian (``pdftype=0``), natural gradient.
"""

from .core import AMICAMLXNG

__all__ = ["AMICAMLXNG"]
