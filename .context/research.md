# pyAMICA Research & Parity Analysis

Fortran-vs-Python parity analysis. Fortran reference: `amica17.f90` (~3900 lines),
`amica17_header.f90` (declarations), `funmod2.f90` (function modules).

> **2026-07-02 - Issue #21 root cause (M-step diverges).** The NG backend's first-order
> M-step uses the density derivative `dpdf = p'(y)` where Fortran uses the score
> `fp = rho*sign(y)*|y|^(rho-1)`; since `p' = -fp*p`, the log-likelihood *descends* from
> iteration 1. Plus gradient-style vs exact-EM mu/beta, PCA vs symmetric-ZCA sphere, and a
> validation-metric artifact (sphered-space `W` vs basis-invariant `W@S`). Full analysis,
> evidence, and fix recipe: [`issue-21/root_cause.md`](issue-21/root_cause.md); runnable
> proof: `issue-21/reproduce_root_cause.py`.

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

## 2026-07-02 design review (torch backend vs Fortran)

Root cause: the default torch backend (`torch_impl/amica_torch.py`, and `amica_torch_v2.py`)
reframes AMICA as "minimize NLL with Adam over reparameterized tensors," but AMICA is a
fixed-point EM / natural-gradient method with closed-form sufficient-statistic updates. Adam on
the reparameterized surface follows a different trajectory and converges to a different fixed
point, so component correlation with Fortran is essentially coincidental. Decision to rewrite:
see `.context/decisions/0001-torch-backend-natural-gradient-em.md`. The NumPy `pyAMICA.py` is a
faithful port and is the spec.

### Concrete bugs (fixable independently of the rewrite)
1. **Swapped mixture factorization** - `amica_torch.py:176-233`.
   `_compute_gg_log_pdf_vectorized` sums log-PDF over sources (`.sum(dim=0)`, line 233) *before*
   the mixture `logsumexp` over k, so all sources are forced to share one mixture label per time
   point. Correct AMICA is per-source mixture reduction then sum over sources
   (Fortran `amica17.f90:1313-1360`): `sum_t sum_i log sum_k alpha_{k,i} f_k(y_{i,t})`. The v2
   adaptive path (`amica_torch_v2.py:236-247`) does it right; the default backend and v2
   non-adaptive fallback do not.
2. **alpha collapsed to a scalar** - `amica_torch.py:187`, `amica_torch_v2.py:259`.
   `torch.log(alpha_k.mean() + eps)` discards the per-source mixture weights `alpha_{k,i}`.
3. **LL normalization (the "~13x" scale gap)** - Fortran reports
   `LL = LLtmp2 / (num_samples * nw)` (`amica17.f90:1866`; `nw` = n_sources = 32 for the sample
   data). Torch divides by `n_samples` only (`amica_torch.py:205`). Fix: divide by
   `n_samples * nw`.
   **CORRECTION (2026-07-02, verified against source):** an earlier version of this note claimed
   Fortran omits `log|det W|` from the reported LL and that the fix should drop the logdet term.
   That is WRONG. Fortran's printed LL INCLUDES both the unmixing log-determinant and the
   sphering log-determinant: `amica17.f90:975-980` computes `Dtemp(h) = sum_i log|R(i,i)|` from the
   QR of `W(:,:,h)` (= `log|det W(h)|`), and `amica17.f90:1273` seeds the per-timepoint accumulator
   `Ptmp(:,h) = Dsum(h) + log(gm(h)) + sldet` (with `Dsum = log|det W|`, `sldet = log|det sphere|`)
   BEFORE the per-source density loop; this flows unchanged into `LL(iter)` at `:1866` (the printed
   value). So the reported LL MUST include `log|det W|` and `sldet`. The autograd backends also need
   `log|det W|` in the training objective for stability (without it W inflates degenerately); the
   closed-form natural-gradient backend gets it for free via the `I - <g y^T>` update (the identity
   term is exactly the gradient of `log|det W|`), but must still ADD `log|det W| + sldet` to the
   REPORTED LL value.
4. **Newton double-steps** - `newton_optimizer.py:221-262` mutates `A` in place, then the main
   loop also calls Adam `optimizer.step()` on the same iteration -> NaN at `newt_start`. Port the
   NumPy Newton (`pyAMICA.py:666-694`) instead.
5. **`pinv` in the hot path** - `amica_torch.py:149` recomputes `W = pinv(A.T).T` every forward.
   Track `W` directly.
6. **float32 default** - Fortran is double throughout; validate in float64.

### Performance / memory / scaling
- Python loops over sources/mix/models in the hot path (`adaptive_pdf.py:122,237,349`;
  `compute_log_likelihood`) serialize into many tiny GPU/MPS kernels. Vectorize with broadcasting
  over `(model, mix, source, block)`.
- No sample blocking: the torch path materializes all-sample intermediates and retains the
  autograd graph, so peak memory grows with recording length (OOM risk on 128-256 ch data).
  Fortran blocks over samples (`block_size` 128-1024) and accumulates sufficient statistics =
  O(block) streaming memory.
- Autograd is unnecessary: every AMICA update is a closed-form function of the responsibilities
  `z`, `v` and simple moments. Dropping it removes graph-retention memory and enables streaming.

## Next steps for parity
1. Fix the three math bugs above (mixture factorization, per-source alpha, LL normalization).
2. Match Fortran initialization exactly (seed, sphering, starting matrices) for trajectory parity.
3. Rewrite the E-step / M-step as vectorized natural-gradient updates with block accumulation
   (ADR 0001); parameterize `W` directly; validate in float64.
4. Port Newton verbatim from NumPy (H from sigma2/kappa/lambda); delete the autograd Newton.
5. Add the numerical-stability bounds and epsilon guards; complete outlier rejection.

## Multi-model parity (how to validate a non-identifiable estimator)
Single-model ICA is identifiable, so parity = Hungarian cross-corr ~0.997 vs Fortran. Multi-model
AMICA is **not partition-identifiable** (many near-degenerate partitions), so exact partition
parity is the wrong bar. The correct test is **distributional equivalence**: run ensembles of both
implementations and check whether NG-vs-Fortran agreement is statistically indistinguishable from
Fortran's own run-to-run agreement. It is (N=20 each, TOST-equivalent within ±0.05; even Fortran vs
itself is only ~0.63). Full method, results, figure, and the multi-model acceptance criteria:
`.context/issue-27/multimodel_distributional_equivalence.md`. The per-model bias `c` update that
this built on: `.context/issue-27/multimodel_c_update.md`.
