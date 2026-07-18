"""Single source of truth for the package version.

``__version__`` is read from the installed distribution metadata, which
setuptools fills from ``pyproject.toml`` at build time. Deriving it (rather than
hardcoding a second copy here) means ``pamica.__version__`` can never drift from
the packaged version, and ``scripts/sync_version.py`` only has to bump one place.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("pamica")
except PackageNotFoundError:  # a source tree with no installed distribution
    __version__ = "0.0.0+unknown"
