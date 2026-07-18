"""MNE-Python compatibility layer for pamica (issue #139).

This subpackage is an *optional* entry point: it imports ``mne`` at module
import time, so it is intentionally NOT imported by :mod:`pamica.__init__`.
``import pamica`` therefore never requires MNE; reach the wrapper explicitly::

    from pamica.mne_compat import AMICAICA

Install the dependency with ``pip install pamica[mne]`` (same lazy-extra
pattern as the optional ``mlx`` backend).
"""

from .core import AMICAICA

__all__ = ["AMICAICA"]
