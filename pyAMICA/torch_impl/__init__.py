"""
PyTorch implementation of AMICA (Adaptive Mixture ICA).

This module provides a GPU-accelerated implementation of AMICA using PyTorch,
with support for CUDA, ROCm, and Apple Silicon (MPS) backends.
"""

from .amica_torch_ng import AMICATorchNG
from .utils import setup_device, check_numerical_stability

__all__ = [
    "AMICATorchNG",
    "setup_device",
    "check_numerical_stability",
]
