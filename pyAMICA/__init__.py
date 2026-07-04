"""
pyAMICA: Python implementation of Adaptive Mixture ICA algorithm.

This package provides a Python implementation of the Adaptive Mixture ICA (AMICA)
algorithm for blind source separation using adaptive mixtures of independent
component analyzers.

Main Features:
- GPU acceleration (CUDA/ROCm/MPS)
- Multiple source models
- Mixture of Generalized Gaussians
- Natural gradient optimization
- Automatic differentiation
- Data preprocessing (mean removal, sphering)
"""

from .version import __version__
from .amica import AMICA
from .torch_impl import AMICATorchNG
from . import amica_utils
from . import amica_data
from . import amica_newton
from . import amica_pdf
from . import amica_viz

# Legacy NumPy implementation (deprecated)
from .pyAMICA import AMICA as AMICA_NumPy

__all__ = [
    "AMICA",
    "AMICATorchNG",
    "AMICA_NumPy",
    "amica_utils",
    "amica_data",
    "amica_newton",
    "amica_pdf",
    "amica_viz",
    "__version__",
]
