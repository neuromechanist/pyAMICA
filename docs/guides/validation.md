# Validation & Parity

**Correctness in pyAMICA is defined as parity with the reference Fortran binary,
not merely as convergence.** A run is correct when it reproduces the Fortran
output within numerical tolerance.

## The validation harness

`validate_implementations.py` runs the implementations on real sample EEG,
matches components across implementations with the Hungarian algorithm, and
reports log-likelihood and per-component correlation. It always uses real sample
data and the Fortran binary, never synthetic data.

## Single-model parity

On real sample EEG the natural-gradient backend reaches Fortran's solution:

- Log-likelihood ~ -3.40 (Fortran ~ -3.4018).
- Hungarian-matched component correlation ~0.997, clearing the >0.95 gate.

The fixed source-density families are bit-exact against the literal Fortran
score/derivative expressions (~1e-15), and the backend converges to the binary's
solution within ~0.005 log-likelihood.

## Multi-model equivalence

Multi-model AMICA is not partition-identifiable, so exact partition parity with
Fortran is the wrong acceptance bar. The right test is whether the two
implementations sample the same distribution over solutions. Running an ensemble
of `N = 20` fits per implementation on the real sample EEG (`n_models = 2`, 3
mixture components, 100 iterations, matched schedule), the pyAMICA-vs-Fortran
partition cross-correlation distribution is statistically equivalent to Fortran's
own run-to-run distribution:

| Distribution (pairwise Hungarian-matched \|corr\|) | Mean | SD | Range |
|---|---:|---:|---|
| within-Fortran (Fortran vs Fortran) | 0.634 | 0.042 | [0.567, 0.772] |
| within-pyAMICA (pyAMICA vs pyAMICA) | 0.644 | 0.046 | [0.537, 0.798] |
| between (pyAMICA vs Fortran) | 0.638 | 0.047 | [0.525, 0.938] |

![Multi-model solution-ensemble cross-correlation distributions for pyAMICA and Fortran.](../assets/figures/multimodel-ensemble.png){ width=640 }
/// caption
Pairwise Hungarian-matched component correlation for 20 pyAMICA and 20 Fortran
multi-model fits of the sample EEG. The within-Fortran, within-pyAMICA, and
between-implementation distributions overlap: the estimators sample the same
solution space.
///

- **Mann-Whitney** (one-sided, H1: between worse than within-Fortran):
  $p = 0.97$, so there is no evidence that cross-implementation agreement is worse
  than Fortran's own run-to-run agreement.
- **TOST** equivalence of the means within a $\pm 0.05$ margin: equivalent
  (mean difference $+0.004$).

The single-run cross-correlation of ~0.64 is therefore intrinsic estimator
spread, not a shortfall: Fortran agrees with *itself* at 0.63. The per-block
sufficient statistics and one M-step are bit-exact against Fortran (~$10^{-15}$),
so the update equations are correct; a small residual in the log-likelihood
*distribution* (pyAMICA $-3.374 \pm 0.040$ vs Fortran $-3.354 \pm 0.003$) is an
optimizer-quality effect, not a model-correctness defect.

## Data adequacy and cross-backend equivalence

Whether backends recover the *same* components depends on how well-determined the
decomposition is, captured by the data-adequacy factor:

$$k = \frac{\text{frames}}{\text{channels}^2}$$

As `k` grows, cross-backend component equivalence rises toward 1.0; at the
rule-of-thumb minimum (`k` around 20-30) only the strongest components are
backend-reproducible, while the rest are under-determined and settle into
different but equally valid local optima (AMICA is non-convex).

### Data-size sweep: equivalence versus k

Holding channels fixed at 70 and increasing the number of frames (so `k` rises),
on real EEG (ds002718 sub-002), cross-backend IC equivalence climbs sharply and
then saturates once the decomposition is well-determined:

![Cross-backend IC equivalence versus the data-adequacy factor k at 70 channels.](../assets/figures/kfactor-equivalence.png){ width=640 }
/// caption
Mean Hungarian-matched cross-backend |correlation| versus $k = \text{frames} /
\text{channels}^2$ (70 channels, 2000 iterations, native-Fortran and PyTorch-CUDA
float64/float32 backends). Equivalence saturates at ~0.98 once $k \geq 60$.
///

| frames | k | mean \|corr\| | components >0.95 |
|---|---|---|---|
| 73,500 | 15 | 0.911 | 55.2% |
| 147,000 | 30 | 0.929 | 56.7% |
| 294,000 | 60 | 0.982 | 90.0% |
| 490,000 | 100 | 0.983 | 94.8% |
| 747,750 | 152 | 0.982 | 92.4% |

!!! note "The threshold is data-specific"
    For this recording the equivalence knee falls **between k=30 and k=60**; below
    it the backends settle into different (equally valid) local optima, above it
    they recover the same components. Where that knee sits depends on the data
    (signal-to-noise ratio, effective rank, source structure), so this is not a
    universal value of `k`. The plateau is ~0.98 rather than 1.0 because of
    intrinsic estimator spread and the float32 path, not a backend defect.

### Why the plateau sits at ~0.98, not 1.0

At the largest data size (k=152) the residual below 1.0 splits cleanly by
precision. The two double-precision implementations, an independent native
Fortran binary and the PyTorch-CUDA backend, agree at 0.995:

| Pair (at k=152) | \|corr\| |
|---|---:|
| native-Fortran f64 vs PyTorch-CUDA f64 | 0.995 |
| native-Fortran f64 vs PyTorch-CUDA f32 | 0.971 |
| PyTorch-CUDA f64 vs PyTorch-CUDA f32 | 0.979 |

This is cross-*implementation* agreement, not just cross-device. The residual gap
is dominated by the float32 path (rounding accumulated over 2000 iterations, plus
an early stop when the natural-gradient learning rate hit its floor), which is a
convergence/precision effect rather than a backend defect.

## Performance across backends

Throughput on real EEG (OpenNeuro ds002718 sub-002; `n_mix=3`, `pdftype=0`,
`block_size=512`, warmed, min-of-repeats). CPU, MPS, and MLX were measured on
Apple Silicon; CUDA on a separate NVIDIA RTX 4090 host, so MLX-versus-CUDA reads
as "best Apple-GPU path versus a strong NVIDIA GPU", not a same-box comparison.

### Single-model, ms/iteration

| channels | MLX f32 | CUDA f32 | CUDA f64 | torch-CPU f32 | torch-CPU f64 | torch-MPS f32 | NumPy f64 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 15.4 | 35.5 | 35.0 | 52 | 71 | 189 | 142 |
| 32 | 21.3 | 35.5 | 36.2 | 143 | 161 | 162 | 287 |
| 48 | 19.5 | 36.0 | 35.9 | 151 | 168 | 168 | 426 |
| 70 | 25.2 | 35.6 | 38.6 | 173 | 193 | 255 | 622 |

MLX is the fastest option on Apple Silicon and stays roughly flat with channel
count (~7x over torch-CPU). PyTorch-MPS is *not* a win (at or worse than CPU), so
use MLX rather than `device="mps"` on Apple hardware. CUDA float32 and float64 are
near-identical here (launch-bound at this size). NumPy is the reference
implementation, not a production path.

### Multi-model (n_models=2), ms/iteration

| channels | MLX f32 | torch-CPU f32 | torch-MPS f32 | NumPy f64 |
|---:|---:|---:|---:|---:|
| 32 | 38 | 187 | 291 | 869 |
| 70 | 45 | 224 | 270 | 928 |

The Apple-GPU win extends to multi-model: MLX ~38-45 ms/iteration, ~5x over
torch-CPU, with MPS still losing.

### Cross-backend log-likelihood agreement (single-model)

Every backend converges to the same log-likelihood to ~3 significant digits on
real EEG, across device and precision, confirming the whole backend family
end-to-end:

| channels | MLX f32 | CUDA f64 | torch-CPU f64 | torch-MPS f32 | NumPy f64 |
|---:|---:|---:|---:|---:|---:|
| 32 | -3.28634 | -3.28635 | -3.28636 | -3.28635 | -3.28620 |
| 48 | -3.20951 | -3.20952 | -3.20953 | -3.20951 | -3.21019 |
| 70 | -3.21579 | -3.21562 | -3.21560 | -3.21570 | -3.21315 |
