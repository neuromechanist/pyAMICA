"""
PyTorch implementation of AMICA (Adaptive Mixture ICA).

This module provides a GPU-accelerated implementation of AMICA using PyTorch,
with support for CUDA, ROCm, and Apple Silicon (MPS) backends.
"""

from .core import AMICATorchNG, PDFTYPE_NAMES
from .utils import setup_device, check_numerical_stability

__all__ = [
    "AMICATorchNG",
    "PDFTYPE_NAMES",
    "setup_device",
    "check_numerical_stability",
]
