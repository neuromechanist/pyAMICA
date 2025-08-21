"""
PyTorch implementation of AMICA (Adaptive Mixture ICA).

This module provides a GPU-accelerated implementation of AMICA using PyTorch,
with support for CUDA, ROCm, and Apple Silicon (MPS) backends.
"""

from .amica_torch import AMICATorch
from .mixture_models import GaussianMixtureICA
from .optimizers import NaturalGradientOptimizer, NewtonOptimizer
from .utils import setup_device, check_numerical_stability

__all__ = [
    'AMICATorch',
    'GaussianMixtureICA',
    'NaturalGradientOptimizer',
    'NewtonOptimizer',
    'setup_device',
    'check_numerical_stability'
]