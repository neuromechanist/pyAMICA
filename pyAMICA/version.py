from os.path import join as pjoin

# Format expected by setup.py and doc/source/conf.py: string of form "X.Y.Z"
_version_major = 0
_version_minor = 1
_version_micro = ''  # use '' for first of series, number for 1 and above
_version_extra = 'dev'
# _version_extra = ''  # Uncomment this for full releases

# Construct full version string from these.
_ver = [_version_major, _version_minor]
if _version_micro:
    _ver.append(_version_micro)
if _version_extra:
    _ver.append(_version_extra)

__version__ = '.'.join(map(str, _ver))

CLASSIFIERS = ["Development Status :: 3 - Alpha",
               "Environment :: Console",
               "Intended Audience :: Science/Research",
               "License :: OSI Approved :: MIT License",
               "Operating System :: OS Independent",
               "Programming Language :: Python",
               "Topic :: Scientific/Engineering"]

# Description should be a one-liner:
description = "pyAMICA: Python implementation of Adaptive Mixture ICA algorithm"
# Long description will go up on the pypi page
long_description = """
AMICA (Adaptive Mixture ICA) is an advanced blind source separation algorithm
that uses adaptive mixtures of independent component analyzers. This implementation provides:

- Multiple source models
- Different PDF types
- Newton optimization
- Component sharing
- Outlier rejection
- Data preprocessing (mean removal, sphering)

For more information, visit: http://github.com/neuromechanist/pyAMICA
"""

NAME = "pyAMICA"
MAINTAINER = "Seyed Yahya Shirazi"
MAINTAINER_EMAIL = "shirazi@ieee.org"
DESCRIPTION = description
LONG_DESCRIPTION = long_description
URL = "http://github.com/neuromechanist/pyAMICA"
DOWNLOAD_URL = ""
LICENSE = "MIT"
AUTHOR = "Seyed Yahya Shirazi"
AUTHOR_EMAIL = "shirazi@ieee.org"
PLATFORMS = "OS Independent"
MAJOR = _version_major
MINOR = _version_minor
MICRO = _version_micro
VERSION = __version__
PACKAGE_DATA = {'pyAMICA': [pjoin('data', '*')]}
REQUIRES = ["numpy", "scipy", "matplotlib", "tqdm", "json5"]
PYTHON_REQUIRES = ">= 3.10"
