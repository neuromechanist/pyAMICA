"""Legacy NumPy reference implementation of AMICA.

Topic-named modules (``core``, ``newton``, ``pdf``, ``data``, ``load``, ``viz``,
``utils``, ``cli``), renamed from the old ``pyAMICA.py``/``amica_*.py`` sprawl in
issue #34. The scikit-learn-style :class:`~pyAMICA.numpy_impl.core.AMICA` is also
exposed at the package root as ``pyAMICA.AMICA_NumPy``.
"""

from .core import AMICA

__all__ = ["AMICA"]
