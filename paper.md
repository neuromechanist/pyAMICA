---
title: 'pamica: GPU-accelerated Adaptive Mixture Independent Component Analysis in Python with Fortran parity'
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

Independent Component Analysis (ICA) is a widely applied method for separating electroencephalographic and magnetoencephalographic (EEG/MEG) recordings into maximally independent sources
that isolate brain, muscle, and artifact activities for downstream analysis [@makeig1995independent; @vigario1997independent; @iversen2019megeeg]. Adaptive Mixture ICA (AMICA) [@palmer2012amica] generalizes single-model ICA to a mixture of such models with adaptive source densities,
and produces both the least dependent and the most dipolar (and thus most physiologically interpretable) component decompositions of EEG among the algorithms benchmarked by @delorme2012independent.
Its reference implementation, written by Jason Palmer, is a Fortran program parallelized with the Message Passing Interface (MPI) and distributed as a compiled binary callable from MATLAB/EEGLAB,
which is difficult to install, runs only on the central processing unit (CPU), and is not usable from a Python scientific workflow.

`pamica` is a Python implementation of AMICA that reproduces the reference Fortran results within numerical tolerance while running on the CPU, NVIDIA graphics processing units (GPUs, via CUDA), and Apple GPUs (Apple's MLX array framework [@mlx2023]).
It is a complete reimplementation built on PyTorch [@paszke2019pytorch], NumPy [@harris2020array],
and SciPy [@virtanen2020scipy], not a wrapper around the Fortran binary,
and exposes a scikit-learn-style estimator under a BSD-3-Clause license.
In double precision it reproduces the reference score-function algebra to floating-point round-off;
it also runs in single precision (float32),
unavailable in the CPU-only binary and required for Apple GPUs, which have no float64 and host the fastest backend (MLX).
`pamica` writes output in the binary format that EEGLAB's AMICA loader reads:
a single-model output file is byte-identical in layout to a native AMICA file,
and multi-model output round-trips through the same loader. Correctness is defined as parity with the Fortran reference for the single-model case and, because multi-model AMICA is not partition-identifiable, as a similar distribution of solutions for the multi-model case;
both are validated on real EEG against the reference binary.
The software is at <https://github.com/sccn/pAMICA> (archived at doi:10.5281/zenodo.21312148).

# Statement of need

AMICA decompositions of neuroelectromagnetic data are well suited to equivalent-dipole source localization and automated component classification [@piontonachini2019iclabel].
Yet its reference implementation is MATLAB-only Fortran, an increasing obstacle as neuroimaging analysis moves toward Python, for example MNE-Python [@gramfort2013meg]:
an AMICA that runs natively in Python and on a GPU, and that is validated to reproduce the Fortran reference numerically, is needed for modern pipelines.

General-purpose Python ICA implementations do not fill this gap. `scikit-learn` and `MNE-Python` provide FastICA [@hyvarinen2000independent] and Infomax [@bell1995information; @lee1999independent],
while Picard [@ablin2018faster] offers faster-converging maximum-likelihood ICA;
none implement AMICA's mixture of models, adaptive generalized-Gaussian source densities, or Newton updates,
so they do not reproduce AMICA decompositions. `pamica` targets EEG and MEG analysts who want AMICA-quality decompositions inside a Python pipeline,
users of GPU hardware who want faster runs than the CPU-only binary, and methodologists who need a transparent reference implementation to build on.

# Implementation and validation

`pamica` provides a natural-gradient [@amari1998natural] expectation-maximization (EM) backend that ports the full AMICA algorithm:
exact-EM mixture updates, a positive-definite Newton step [@palmer2008newton],
symmetric zero-phase-component-analysis (ZCA) sphering, the five source-density families of the reference (generalized Gaussian, Gaussian,
logistic, sub-Gaussian, and the extended-Infomax kurtosis switcher), a mixture of ICA models, and component sharing across models.
It also computes mutual information reduction (MIR) and pairwise mutual information (PMI), separation-quality metrics useful for benchmarking ICA algorithms [@delorme2012independent].

`pamica`'s conformity with the reference binary is measured with two complementary metrics: Hungarian-matched component correlation
and the Amari distance [@amari1996new], a relabeling- and scale-invariant unmixing-matrix metric that needs no assignment step.
Both implementations were run for the AMICA heuristic default of 2000 iterations with Newton off (`do_newton=0`) and otherwise-default parameters
(settings transcribed between `pamica`'s JSON and Fortran's native text format).
Newton acceleration is disabled here to isolate the algorithm from initialization: the Newton update speeds convergence, but once enabled it lets independently seeded runs settle a few under-determined components into different, equally likely optima, whereas a matched initialization recovers agreement ([documentation](https://eeglab.org/pAMICA/guides/validation/)).
The single-model comparison uses a well-determined external recording (OpenNeuro ds002718, $k\approx153$, where $k$ = frames over squared channel count), well past the ~60 threshold where cross-backend agreement plateaus, together with the bundled 32-channel sample ($k\approx30$); Table 1 gives each metric's dataset.
Score functions and per-block sufficient statistics are exact to floating-point resolution against the literal Fortran expressions on the bundled sample.
A mixture of ICA models is not partition-identifiable, so exact partition parity is the wrong bar for the multi-model case;
it is instead assessed by whether the two implementations sample a similar distribution of solutions, across ensembles of 20 runs each (\autoref{fig:ensemble}).
A permutation test finds no evidence that cross-implementation agreement is worse than Fortran's own run-to-run agreement.
Multi-model log-likelihood distributions still differ slightly at a matched iteration budget
(`pamica` needs about twice as many iterations to reach Fortran's mean), so full-likelihood similarity is not yet claimed.

| Regime | Metric (dataset) | Result (mean) |
|---|---|---|
| Single-model | Log-likelihood gap to Fortran (ds002718, $k\approx153$) | within ~0.0005 of $-3.6993$ |
| Single-model | Hungarian-matched component correlation (ds002718, $k\approx153$) | 0.998 |
| Single-model | Amari distance (bundled 32-channel sample, $k\approx30$) | 0.006 |
| Single-model | Score functions and sufficient statistics (bundled sample) | exact, $\sim\!10^{-15}$ |
| Multi-model | Component correlation, single run: `pamica`-Fortran; Fortran-Fortran (bundled) | 0.65; 0.64 (sd 0.05) |
| Multi-model | Amari distance, single run: `pamica`-Fortran; Fortran-Fortran (bundled) | 0.163; 0.174 (sd 0.02) |
| Multi-model | Ensemble agreement, cross-implementation $-$ within-Fortran, 20 runs each (bundled) | correlation $+0.011$ ($p=0.96$); Amari $-0.011$ ($p>0.999$) |

: Parity of `pamica` with the Fortran reference. The two single-model conformity metrics use different recordings, hence the two data-adequacy ratios $k$: the correlation headline uses a well-determined external recording ($k\approx153$), while the Amari distance and score-function checks use the bundled 32-channel sample ($k\approx30$). Multi-model agreement is distributional, since a mixture of models is not partition-identifiable; the ensemble row is the mean difference between cross-implementation and within-Fortran agreement, with a run-level permutation $p$-value. Values are means (sd, standard deviation) over matched components or, for multi-model, over within/cross-implementation run pairs (190/400).

![Multi-model solution-ensemble partition-correlation distributions (panel A) and log-likelihood distributions (panel B) for 20 `pamica` and 20 Fortran fits of the sample EEG; dashed lines mark each distribution's mean.
The within-Fortran, within-`pamica`, and between-implementation correlation distributions overlap,
so the single-run correlation reflects the estimator's intrinsic run-to-run spread rather than a gap to the reference.
Panel B's apparent separation is a 0.009 log-likelihood gap on a ~0.035 axis.\label{fig:ensemble}](docs/assets/figures/multimodel-ensemble.png){ width=100% }

All backends converge to the same single-model log-likelihood on real EEG (maximum pairwise difference ~0.003).
On Apple Silicon, MLX is the fastest backend and flat with channel count (Table 2); PyTorch-MPS is never a win.
Double-precision CUDA is the reproducible NVIDIA path; native Fortran scales with CPU cores and, with enough cores pinned,
can beat the GPU on a larger, hotter host, though it does not match Apple's MLX on laptop hardware.
A data-size sweep ([documentation](https://eeglab.org/pAMICA/guides/validation/)) shows cross-backend component agreement rising with frames per channel and plateauing near 0.98 once the decomposition is well-determined,
where two independent double-precision implementations agree at a mean of 0.995;
single-precision runs agree with float64 to four to five significant digits, so float64 stays the default for parity.

| Backend (device) | Precision | ms / iteration |
|---|---|---:|
| MLX (Apple GPU) | float32 | 25 |
| CUDA (NVIDIA RTX 4090) | float64 | 39 |
| Native Fortran (Intel Core i9-13900K, 24 cores) | float64 | 30 |
| PyTorch CPU (Apple Silicon) | float64 | 193 |
| Native Fortran (Apple Silicon, 8 cores) | float64 | 70 |
| PyTorch MPS (Apple GPU) | float32 | 255 |
| NumPy (reference, Apple Silicon) | float64 | 622 |

: Single-model throughput on real 70-channel EEG (`n_mix`=3,
`pdftype`=0, `block_size`=512; warm, minimum of repeated runs).
CPU, MPS, and MLX on Apple Silicon; CUDA on a separate NVIDIA RTX 4090 (float32 comparable, ~36 ms).
The two native-Fortran rows are from a separate core-count sweep ([documentation](https://eeglab.org/pAMICA/guides/validation/)) at each backend's plateau;
the other CPU rows use platform-default threads and are not core-matched to Fortran.
Unlike the correctness comparison, this benchmark uses external data (OpenNeuro ds002718, one subject so far) and specific GPU hardware.

The correctness harness never uses synthetic data;
the multi-model and score-function checks need no external download (bundled sample only).
The full performance tables, per-run Amari-distance detail, data-size sweep,
and step-by-step reproduction commands are in the [documentation](https://eeglab.org/pAMICA/guides/validation/).

# State of the field

`pamica` complements rather than replaces the reference Fortran AMICA used with EEGLAB [@delorme2004eeglab]: it uses the same output format as that Fortran version,
adds a Python API with GPU support, and runs the reference Fortran itself through a bundled dependency-free native build (no Intel Math Kernel Library or MPI runtime).
Two other Python AMICA reimplementations have appeared [@esmaeili2025amica; @herforth2026pyamica],
both of which provide MNE-Python-compatible objects; `pamica` offers a scikit-learn-style array API, byte-identical EEGLAB I/O, an MLX backend for Apple GPUs, and an optional MNE-Python wrapper.
What sets `pamica` apart is the depth of its Fortran-parity validation (Table 1): source-density score functions bit-exact to floating-point resolution against the literal Fortran expressions, and a distributional-similarity framework for the non-identifiable multi-model case.

# Acknowledgements

We thank Jason Palmer and his advisor Ken Kreutz-Delgado, co-developers of AMICA, for the reference implementation.
We also thank the EEGLAB community for the tools and sample data used to validate this work.
Two of the authors are original developers of the methods `pamica` builds on: S.M. co-developed the AMICA algorithm [@palmer2012amica] and A.D. is a lead developer of EEGLAB [@delorme2004eeglab].
This work was supported by The Swartz Foundation (Old Field, NY) to the Swartz Center for Computational Neuroscience and by National Institutes of Health grant R01-NS047293 (to A.D. and S.M.).

# References

