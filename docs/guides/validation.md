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

!!! info "Data-size sweep in progress"
    A quantitative data-size sweep at 70 channels (holding channels fixed and
    increasing frames so `k` rises) is being finalized to chart cross-backend
    equivalence versus `k`. The results figure and table will be added here.
