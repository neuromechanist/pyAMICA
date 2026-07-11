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
    orcid: 0000-0001-5557-259X
    corresponding: true
    affiliation: 1
  - name: Arnaud Delorme
    orcid: 0000-0002-0799-3557
    affiliation: "1, 2"
  - name: Scott Makeig
    orcid: 0000-0002-9048-8438
    affiliation: 1
affiliations:
  - name: Swartz Center for Computational Neuroscience, Institute for Neural Computation, University of California San Diego, USA
    index: 1
  - name: Centre de Recherche Cerveau et Cognition (CerCo), CNRS, University of Toulouse, France
    index: 2
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
(CUDA), and Apple GPUs (Apple's MLX array framework [@mlx2023]). It runs in
double precision for bit-level agreement with the reference and adds
single-precision (float32) execution, which the CPU-only reference binary does
not offer, for 5-19x-faster runs where bit-exact parity is not required; making
float32 AMICA converge reliably is what unlocks the GPU speedups, and it is
required on Apple GPUs, which have no float64. It is built on
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
implementation is the original Fortran code: it depends on an MPI toolchain and
precompiled binaries that are awkward to build across platforms, it cannot use a
GPU, and it is invoked as an external process
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
ICA models, and component sharing across models.

Correctness is measured against the reference binary on real sample EEG
(Table 1). For a single model the solution reproduces Fortran's log-likelihood
and component structure, and the source-density score functions and per-block
sufficient statistics are bit-exact. A mixture of ICA models is not
partition-identifiable, so exact partition parity is the wrong bar; the
multi-model case is validated instead by distributional equivalence, where
pyAMICA's ensemble of solutions is statistically indistinguishable from Fortran's
own run-to-run spread. The single-run partition cross-correlation of ~0.64 is
intrinsic estimator spread, not a shortfall, because Fortran agrees with itself at
0.63.

| Regime | Metric | Result |
|---|---|---|
| Single-model | Log-likelihood (`pyAMICA` vs Fortran) | $-3.40$ vs $-3.4018$ |
| Single-model | Component correlation (Hungarian-matched) | $0.997$ |
| Single-model | Score functions and sufficient statistics | bit-exact ($\sim\!10^{-15}$) |
| Multi-model | Partition cross-corr, single run (`pyAMICA`; Fortran vs itself) | $0.64$; $0.63$ |
| Multi-model | Solution-ensemble equivalence ($N\!=\!20$ each) | Mann-Whitney $p\!=\!0.97$; TOST equivalent |

  : Parity of `pyAMICA` with the Fortran reference on real sample EEG.

All backends agree on the log-likelihood to at least three significant digits on
real EEG. On Apple Silicon the MLX backend is the fastest option and stays flat
with channel count; PyTorch-MPS is not a win (at or worse than the CPU), and
double-precision CUDA is the bit-reproducible path on NVIDIA hardware (Table 2). A
data-size sweep further shows cross-backend component equivalence rising with
frames per channel and plateauing near 0.98 once the decomposition is
well-determined, where two independent double-precision implementations (native
Fortran and PyTorch-CUDA) agree at 0.995. Single precision is
seven-significant-digit rather than bit-exact, so double precision remains the
default for Fortran-parity runs.

| Backend (device) | Precision | ms / iteration |
|---|---|---:|
| MLX (Apple GPU) | float32 | 25 |
| CUDA (NVIDIA RTX 4090) | float64 | 39 |
| PyTorch CPU | float64 | 193 |
| PyTorch MPS (Apple GPU) | float32 | 255 |
| NumPy (reference) | float64 | 622 |

  : Throughput on real 70-channel EEG (ms per iteration). CPU, MPS, and MLX on
Apple Silicon; CUDA on an NVIDIA RTX 4090 (a separate host); CUDA float32 is
comparable (~36 ms).

A companion validation harness runs both implementations on the same real EEG and
matches components with the Hungarian algorithm; correctness tests use only real
sample EEG and the reference binary, never synthetic data. The full per-channel
performance tables, the cross-backend log-likelihood-agreement and data-size
sweeps, and the multi-model ensemble figure are in the documentation
(<https://eeglab.org/pyAMICA/guides/validation/>).

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

We thank Jason Palmer and Ken Kreutz-Delgado, co-developers of AMICA, for the
reference implementation, and the EEGLAB community for the tools and sample data
used to validate this work.

# References
