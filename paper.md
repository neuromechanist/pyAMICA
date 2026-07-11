---
title: 'pyAMICA: GPU-accelerated Adaptive Mixture Independent Component Analysis in Python with Fortran parity'
tags:
  - Python
  - PyTorch
  - independent component analysis
  - blind source separation
  - EEG
  - neuroscience
authors:
  - name: Seyed Yahya Shirazi
    orcid: 0000-0000-0000-0000
    corresponding: true
    affiliation: 1
affiliations:
  - name: Swartz Center for Computational Neuroscience, Institute for Neural Computation, University of California San Diego, USA
    index: 1
date: 11 July 2026
bibliography: paper.bib
---

# Summary

Independent Component Analysis (ICA) is a standard method for separating
electroencephalography (EEG) and electromyography (EMG) recordings into
maximally independent sources, which isolates brain, muscle, and artifact
activity for downstream analysis. Adaptive Mixture ICA (AMICA) generalizes single-model
ICA to a mixture of ICA models with adaptive source densities, and produces the
most dipolar component decompositions of EEG among widely used algorithms, meaning
its components are best modeled by a single equivalent current dipole and are
therefore the most physiologically interpretable [@delorme2012independent]. Its reference implementation,
however, is a Fortran program parallelized with the Message Passing Interface
(MPI) and distributed as a compiled binary driven from MATLAB/EEGLAB, which is
difficult to install, runs only on the CPU, and is not usable from a Python
scientific workflow.

`pyAMICA` is a Python implementation of AMICA that reproduces the reference
Fortran results within numerical tolerance while running on the CPU, NVIDIA GPUs
(CUDA), and Apple GPUs (Apple's MLX array framework [@mlx2023]). It is built on
PyTorch [@paszke2019pytorch], NumPy [@harris2020array], and SciPy
[@virtanen2020scipy], exposes a scikit-learn-style estimator, and writes results
in the exact binary format that EEGLAB's AMICA loader reads. A single-model
`pyAMICA` run is byte-identical to a native AMICA run and needs no manual
re-interpretation; multi-model output round-trips through the same EEGLAB loader
with a self-consistent layout. Correctness is defined as parity with the Fortran
reference and is validated on real EEG against the reference binary.

# Statement of need

AMICA yields components that are well suited to equivalent-dipole source
localization and to automated classification of independent components
[@piontonachini2019iclabel], and it separates EEG more effectively than most
alternatives [@delorme2012independent; @palmer2012amica]. The reference
implementation is the original Fortran code: it depends on a Message Passing
Interface (MPI) toolchain and precompiled binaries that are awkward to build
across platforms, it cannot use a GPU, and it is invoked as an external process
from MATLAB rather than called from Python. As neuroimaging analysis has moved
toward Python, for example MNE-Python [@gramfort2013meg], an AMICA that runs
natively in Python and on a GPU, and that is validated to reproduce the Fortran
reference numerically, is needed for modern pipelines and for studying the
algorithm in an open codebase.

General-purpose Python ICA implementations do not fill this gap. `scikit-learn`
and `MNE-Python` provide FastICA [@hyvarinen2000independent] and Infomax
[@bell1995information; @lee1999independent], and Picard [@ablin2018faster]
provides a faster maximum-likelihood ICA, but none implement AMICA's mixture of
models, adaptive generalized-Gaussian source densities, or Newton updates, and so
they do not reproduce AMICA decompositions. `pyAMICA` targets researchers who
need AMICA specifically: EEG/EMG analysts who want AMICA-quality decompositions
inside a Python pipeline, users of GPU hardware who want faster runs than the
CPU-only binary, and methodologists who need a transparent reference
implementation to inspect and build on.

# Implementation and validation

`pyAMICA` provides a natural-gradient [@amari1998natural] expectation-maximization
backend that ports the full AMICA algorithm: exact-EM mixture updates, a
positive-definite Newton step [@palmer2008newton], symmetric
zero-phase-component-analysis (ZCA) sphering, the five
source-density families of the reference (generalized Gaussian, Gaussian,
logistic, sub-Gaussian, and the extended-Infomax kurtosis switcher), a mixture of
ICA models, and component sharing across models. On real sample EEG, the
single-model solution matches the Fortran reference to a log-likelihood of
approximately -3.40 (reference -3.4018) with a Hungarian-matched component
correlation of approximately 0.997. The fixed source-density families are
bit-exact against the literal Fortran score and derivative expressions (to about
$10^{-15}$). Because a mixture of ICA models is not partition-identifiable, the
multi-model case is validated by distributional equivalence: the
implementation-versus-reference cross-correlation distribution is statistically
indistinguishable from the reference's own run-to-run spread (Mann-Whitney
$p = 0.97$).

Across hardware, all backends agree on the log-likelihood to at least three
significant digits on real EEG. On Apple Silicon the MLX backend is the fastest
option (roughly 15-25 ms per iteration, about seven times faster than a
multithreaded CPU and faster than an NVIDIA RTX 4090 at EEG channel counts),
while double-precision CUDA is the bit-reproducible path on NVIDIA hardware and
runs about 4.5 times faster than a 16-thread CPU. A data-size sweep shows that
cross-backend component equivalence rises with the number of frames per channel
and plateaus near 0.98 once the decomposition is well determined. A companion
validation harness runs both the Python and Fortran implementations on the same
real EEG and matches components with the Hungarian algorithm. Correctness tests
use only real sample EEG and the reference binary; synthetic data is never used
as the basis for a correctness claim, so parity is always measured against the
reference.

# State of the field

`pyAMICA` complements rather than replaces EEGLAB [@delorme2004eeglab] and its
Fortran AMICA plugin: it reads and writes the same output format, so results move
between the two, while adding GPU support and a Python API. Other Python AMICA
reimplementations have appeared [@esmaeili2025amica; @herforth2026pyamica], which
also target MNE-Python pipelines and GPU execution. `pyAMICA` is distinguished by
defining correctness as quantitative parity with the Fortran reference and
validating it accordingly (component correlation of about 0.997, source-density
score functions bit-exact to about $10^{-15}$, and a distributional-equivalence
test for the non-identifiable multi-model case), by writing byte-identical EEGLAB
output, and by an MLX backend for Apple GPUs.

# Acknowledgements

We thank Jason Palmer, Ken Kreutz-Delgado, and Scott Makeig for developing AMICA
and making the reference implementation available, and the EEGLAB community for
the tools and sample data used to validate this work.

# References
