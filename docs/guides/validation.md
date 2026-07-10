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
implementations sample the same distribution over solutions. On an ensemble of
real sample EEG runs, the pyAMICA-vs-Fortran partition cross-correlation
distribution is statistically equivalent to Fortran's own run-to-run
distribution.

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
| 73,500 | 15 | 0.892 | 49.5% |
| 147,000 | 30 | 0.932 | 61.4% |
| 294,000 | 60 | 0.983 | 92.9% |
| 490,000 | 100 | 0.983 | 96.2% |
| 747,750 | 152 | 0.982 | 92.4% |

!!! note "The threshold is data-specific"
    For this recording the equivalence knee falls **between k=30 and k=60**; below
    it the backends settle into different (equally valid) local optima, above it
    they recover the same components. Where that knee sits depends on the data
    (signal-to-noise ratio, effective rank, source structure), so this is not a
    universal value of `k`. The plateau is ~0.98 rather than 1.0 because of
    intrinsic estimator spread and the float32 path, not a backend defect.
