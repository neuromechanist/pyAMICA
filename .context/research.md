# pyAMICA Research & Parity Analysis

Fortran-vs-Python parity analysis. Fortran reference: `amica17.f90` (~3900 lines),
`amica17_header.f90` (declarations), `funmod2.f90` (function modules).

## Core Data Structures (Fortran -> Python)
| Fortran | Python | Status |
|---|---|---|
| A (mixing) | `self.A` | done |
| W (unmixing) | `self.W` | done |
| c (bias), mean | `self.c`, `self.mean` | done |
| sphere (S) | `self.sphere` | done |
| gm (model weights) | `self.gm` | done |
| alpha, mu, sbeta, rho | `self.alpha`, `self.mu`, `self.beta`, `self.rho` | done |
| comp_list, LL, nd | `self.comp_list`, `self.ll`, `self.nd` | done |
| lambda, kappa, sigma2 (Newton) | `compute_newton_direction()` | partial |
| baralpha (Newton) | `self.baralpha` in legacy `pyAMICA.py` | done in NumPy; missing in torch backend |

## Core Subroutines (Fortran -> Python)
| Fortran | Python | Status |
|---|---|---|
| get_data | `load_data_file()`, `preprocess_data()` | done |
| get_updates_and_likelihood | `_get_updates_and_likelihood()` | partial |
| accum_updates_and_likelihood | merged into `_get_updates_and_likelihood()` | merged |
| update_params | `_update_parameters()` | done |
| get_unmixing_matrices | `get_unmixing_matrices()` | done |
| identify_shared_comps | `identify_shared_components()` | done |
| reject_data | `_reject_outliers()` in legacy `pyAMICA.py` | done in NumPy; missing in torch backend |
| determine_block_size | `determine_block_size()` | done |
| write_output / write_history | `save_results()` | done / partial |

## Missing / incomplete in the PyTorch backend
The legacy NumPy `pyAMICA.py` implements outlier rejection, `baralpha`, and Newton; the PyTorch
backend (the parity-focused path) does not yet. Items below refer to the torch backend.
1. Outlier rejection (`do_reject`, `reject_data`) - present in NumPy, absent in torch.
2. Adaptive PDF selection (`do_choose_pdfs`) - partial, in `amica_torch_v2.py`, not wired into the default interface.
3. Full Newton (`baralpha`, complete Hessian) - present in NumPy; torch Newton (`amica_torch_v2.py`) is unstable and not wired into the default interface.
4. Parallel processing (Fortran uses OpenMP).
5. Numerical-stability safeguards (eigenvalue/condition bounds, overflow/underflow protection).

## Critical divergence causes
1. **Likelihood scaling/sign.** Fortran LL ~ -3.44/sample; early Python LL was positive and large.
   Enhanced PyTorch produced positive LLs (mathematically impossible).
2. **Numerical stability.** Python hits NaN during optimization; Fortran uses bounds
   `mincond=1e-15`, `minlog=-1500`, `maxdble=1e32`, `mineig=1e-15`.
3. **Update equations.** Possible gradient/accumulation differences; mixture-component handling.
4. **Data format.** Fortran column-major vs Python row-major (handled, watch edge cases).
5. **Precision.** Fortran double throughout; Python mixes float32/float64.

## 2025-08-20 deep analysis
### Log-likelihood normalization fix
Generalized Gaussian PDF: `p(x; rho, beta) = (rho / (2*beta*Gamma(1/rho))) * exp(-(|x/beta|)^rho)`,
so `log p = log(rho) - log(2) - log(beta) - log(Gamma(1/rho)) - (|x/beta|)^rho`.
```python
# INCORRECT (produced positive LL):
log_norm = torch.lgamma(1.0 + 1.0/rho) + torch.log(torch.tensor(2.0))
# CORRECT:
log_norm = torch.log(torch.tensor(2.0)) + torch.lgamma(1.0/rho) - torch.log(rho)
```
Location: `pyAMICA/torch_impl/adaptive_pdf.py`.

### Newton ramping
Fortran ramps conservatively at Newton start (lrate 0.05 -> 0.10 at iter 51, doubling, capped ~0.5).
Fixes applied: Fortran-matched ramp, off-by-one iteration-counter fix, gradient clipping
(max_norm=5.0 when Newton active). Remaining: NaN still occurs at the Newton-start iteration and
the enhanced model may still produce positive LL; both need revalidation after the latest fixes.

### Observed LL scale (per sample)
Fortran -3.41, PyTorch basic -44.56 (~13x vs Fortran), PyTorch enhanced post-fix -114.58 (~34x).
The "~13x" figure quoted elsewhere refers to the basic backend; the enhanced model diverges further.
A normalization/computation difference remains to resolve.

### Component correlation
Mean ~0.46 (best 0.68, worst 0.24) in the analyzed run; target >0.95. Likely causes: initialization
mismatch, precision, optimization-path divergence, sphering/whitening differences.

## Next steps for parity
1. Match Fortran initialization exactly (seed, sphering, starting matrices).
2. Step through Fortran likelihood and compare intermediate values.
3. Add the numerical-stability bounds and epsilon guards.
4. Complete Newton (line search for stability) and outlier rejection.
