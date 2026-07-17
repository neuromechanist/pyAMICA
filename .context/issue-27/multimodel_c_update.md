# Issue #27: per-model bias `c` update + 2-model partition re-measurement

## What was done
Ported the update gated by Fortran's `update_c` flag (numerator accumulation at
amica17.f90:1423-1429, division at :1899-1901) into both `AMICATorchNG` and the
legacy NumPy `pamica.py`:

- `c[i,h] = dc_numer[i,h] / dc_denom[i,h]`, with
  `dc_numer[i,h] = sum_t v_h(t) * x(i,t)` (sphered-data space) and
  `dc_denom[i,h] = sum_t v_h(t) = dgm[h]`.
- `c` is the per-model, per-channel responsibility-weighted **data-space** mean.
  The E-step centers each model's data before unmixing: `b = W(x - c)`
  (Fortran subtracts `wc = W c`). The Python E-step now computes
  `b = (X - c).T @ W` (data-space subtraction, equivalent, keeps `c`'s
  semantics identical to Fortran).
- Guarded to a no-op for `n_models = 1`: with `v == 1` the update collapses to
  the (zero) mean of mean-removed data, and skipping it keeps single-model
  parity **bit-exact** (issue #24). Without the guard a ~1e-13 float-sum
  residual would perturb the otherwise machine-exact single-model trajectory.
- Replaced the old `dc = sum(g)` accumulator (gradient-style bias that was
  accumulated but never applied — `c` was frozen at 0) with the data-space
  `dc_numer`. `transform()` in both backends now unmixes as `W(x - c)`.
- Dead-model containment: a model with zero total responsibility (`dgm[h]==0`)
  would give `0/0`; the guard keeps that model's PRIOR `c` instead of writing a
  NaN. A NaN `c` would poison the next iteration's cross-model `softmax` for
  every model (unlike `log(gm[h])=-inf`, which `softmax` tolerates), so this
  mirrors the existing mu/beta/rho non-finite guards in the same method.

## Controlled re-measurement (real sample EEG, 2 models, 100 iters)
`scratchpad/measure_multimodel_xcorr.py`: Fortran binary (num_models=2) run to
convergence, then `AMICATorchNG` (n_models=2) run twice with an otherwise
identical config/seed — once with the `c` update ON, once forced OFF (c
re-zeroed each iteration to reproduce the pre-fix trajectory). Metric: stacked
`2*NW`-component Hungarian |correlation| of the per-model unmixing rows vs
Fortran (same orientation as the validated single-model harness).

| run              | final LL | mean cross-corr | min cross-corr |
|------------------|---------:|----------------:|---------------:|
| NG, c OFF (pre)  |  -3.3754 |          0.6306 |         0.2647 |
| NG, c ON  (fix)  |  -3.3762 |          0.6415 |         0.2729 |
| Fortran          |  -3.3596 |            1.000 |          1.000 |

**delta mean cross-corr = +0.011** (LL comparable, essentially unchanged).

## Conclusion
The omitted `c` update was a genuine, fixable contributor to the multi-model
gap, but a **minor** one: with all else held fixed it lifts the 2-model
partition cross-correlation by only ~0.011. The dominant residual gap is
**intrinsic partition ambiguity** (mixture-of-ICA has many near-degenerate
partitions; NG is self-consistent, cross-corr 1.0 across block sizes). The
`>0.95` target is not reachable via the `c` fix alone. This matches issue #27's
prior expectation; the fix is retained because it is Fortran-faithful and
correct (unit-validated: `c` equals the responsibility-weighted data mean and
the two models center differently), and it removes the last known
non-ambiguity discrepancy in the multi-model M-step.

Absolute cross-corr magnitudes depend on run configuration and random init
(Fortran uses its own RNG; NG uses seed=0), so the ~0.63 baseline here is not
identical to the ~0.77 figure previously noted under a different setup. The
**controlled A/B delta** (+0.011, same config/seed, only `c` toggled) is the
config-independent result.
