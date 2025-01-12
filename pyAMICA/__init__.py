"""
pyAMICA: Python implementation of Adaptive Mixture ICA algorithm.

This package provides a Python implementation of the Adaptive Mixture ICA (AMICA) 
algorithm for blind source separation using adaptive mixtures of independent 
component analyzers.

Main Features:
- Multiple source models
- Different PDF types
- Newton optimization
- Component sharing
- Outlier rejection
- Data preprocessing (mean removal, sphering)
"""

from .version import __version__
from .pyAMICA import AMICA
from . import amica_utils
from . import amica_data
from . import amica_newton
from . import amica_pdf
from . import amica_viz

__all__ = [
    'AMICA',
    'amica_utils',
    'amica_data',
    'amica_newton',
    'amica_pdf',
    'amica_viz',
    '__version__'
]
