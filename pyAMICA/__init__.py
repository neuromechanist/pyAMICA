"""
pyAMICA: Python implementation of Adaptive Mixture ICA algorithm.

This package provides a Python implementation of the Adaptive Mixture ICA (AMICA)
algorithm for blind source separation using adaptive mixtures of independent
component analyzers.

Main Features:
- GPU acceleration (CUDA/ROCm/MPS)
- Multiple source models
- Mixture of Generalized Gaussians
- Natural-gradient EM with Newton (Fortran parity)
- Data preprocessing (mean removal, sphering)
"""

from .version import __version__
from .amica import AMICA
from .torch_impl import AMICATorchNG
from . import metrics
from . import numpy_impl
from . import torch_impl

# Legacy NumPy reference implementation (topic-named modules under numpy_impl/,
# issue #34); AMICA_NumPy is its scikit-learn-style interface.
from .numpy_impl import AMICA as AMICA_NumPy

__all__ = [
    "AMICA",
    "AMICATorchNG",
    "AMICA_NumPy",
    "metrics",
    "numpy_impl",
    "torch_impl",
    "__version__",
]
