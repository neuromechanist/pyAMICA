# Issue #21 root cause: NG-backend M-step diverges (dpdf vs score, gradient vs exact-EM)

**Date:** 2026-07-02. **Status:** root cause proven; full >0.95 re-port in progress.

## TL;DR

Issue #21 frames the ~0.68 correlation as a *subtle fixed-point gap* (blaming
initialization, iteration count, or the E-step). That is wrong. **The NG backend's
natural-gradient M-step diverges from iteration 1.** With `lrate` held at Fortran's 0.05
(where Fortran *ascends* -3.51 -> -3.44), NG's log-likelihood goes monotonically *down*:
-3.51, -3.54, -3.58, -3.93, -4.83, ... The Phase-4 lrate ratchet only *masked* this by
collapsing `lrate` to ~1e-6 and freezing parameters at a bad point (the "-3.477 plateau"
prior sessions recorded). The initial LL matches Fortran (-3.512 vs -3.513), so the E-step,
data scaling, and init are fine; only the M-step *update direction* is broken.

This overturns:
- the earlier note "Newton is posdef 50/50 / the cause is outside the update equations"
  (memory `amica-ng-fixed-point-gap`, now corrected);
- AGENTS.md's premise that the NumPy `pyAMICA.py` "is the faithful spec" -- it shares the
  bug, which is why `test_ng_backend.py::test_sufficient_stats_match_numpy_reference`
  passes (it only asserts NG == NumPy, never NG == Fortran).

## Bug 1 (critical): score `fp` vs density derivative `dpdf`

The first-order updates -- natural-gradient `g`/`dA`, `dmu`, `dbeta` -- use the density
derivative `dpdf = p'(y)` (2nd return of `amica_torch_ng._log_pdf_and_deriv`), but Fortran
uses the **score** `fp = rho*sign(y)*|y|^(rho-1)`:

- `amica17.f90:1467` `fp = rho*sign(y)*|y|^(rho-1)`
- `:1485` `ufp = u*fp` (u = v*z, the model-weighted mixture responsibility)
- `:1493` `g(i) += sbeta_j * ufp`  (natural-gradient score)
- `:1532` `dmu_numer = sum(ufp)`

Since `p'(y) = -fp(y)*p(y)` for the generalized Gaussian, substituting `dpdf` for `fp`
**flips the update sign** -> descent. (Phase 4 already fixed exactly this for the *Newton*
curvature terms, `pyAMICA.py:726` "use the score fp ... NOT dpdf", but never fixed the
first-order terms.)

**Proof:** monkeypatching `_log_pdf_and_deriv` to return `(_log_pdf, _score)` turns
monotonic descent into monotonic ascent tracking Fortran (i0=-3.512, i1=-3.501, i5=-3.484)
and makes Newton positive-definite (0 fallbacks vs 68/68 before). See
`reproduce_root_cause.py`.

## Bug 2: gradient-style vs exact-EM mu/beta

NG uses `mu += lrate*dmu/dalpha`, `beta *= sqrt(1+lrate*dbeta)`. Fortran uses exact-EM
closed forms with **no lrate** (`amica17.f90:1978,1993`):

    mu    = mu + dmu_numer/dmu_denom          dmu_numer=sum(ufp), dmu_denom=sbeta*sum(ufp/y)
    sbeta = sbeta*sqrt(dbeta_numer/dbeta_denom) dbeta_numer=sum(u), dbeta_denom=sum(ufp*y)
    alpha = dalpha_numer/sum_j dalpha_numer     (NG already does this correctly)

With only Bug 1 fixed, LL ascends then explodes (overshoot); porting the exact-EM forms too
gives stable convergence and Newton posdef every iteration.

## Further confirmed faithfulness gaps (needed for full >0.95)

- **Natural-gradient space/sign.** NG builds `dA = X @ (v*g)` (DATA-space `sum x g^T`, then
  ADD). Fortran builds `dWtmp = g^T b` (SOURCE-space `sum g_t b_t^T`, b=Wx, `:1592`) and
  applies `A += -lrate*A*(I - <g b^T>/dgm)` (SUBTRACT, `:1800-1917`).
- **Sphere.** Fortran S is symmetric ZCA `V diag(1/sqrt(eval)) V^T` (verified: S == that to
  5e-6; S symmetric to 2e-17; `:480-481`, `do_approx_sphere=True` path). NG/NumPy compute
  PCA `D^-1/2 V^T` (approx=True) or `V D^-1/2` (approx=False) -- neither symmetric, and the
  Python `do_approx_sphere` semantics are inverted vs Fortran. Also `sample_params.json`
  sets `do_approx_sphere=false` while the gate test `_fresh_ng()` uses the default True.
- **rho update** omits Fortran's `1/psi(1+1/rho)` digamma factor (`:2013-2014` vs
  `amica_torch_ng.py:768`).
- **c update** differs (Fortran `c = sum(v*x)/sum(v)`, data-space, `:1900`; NG
  `c += lrate*dc/dgm`). Negligible for 1 model + do_mean (c ~ 0).
- **fit() lrate ratchet** halves on every LL decrease; Fortran reduces only after
  `max_decs` consecutive decreases.

## Measurement artifact (separate from the algorithm)

`validate_implementations.py::compare_results` correlates raw `W` rows, i.e. the
*sphered-space* unmixing, but NG and Fortran sphere in different bases, so even an identical
decomposition scores low. The basis-invariant metric is the total spatial filter `W@S`
(Fortran's own mixing is `pinv(W*S)`, `loadmodout15.m:257`). The harness should compare
`W @ sphere` rows, not `W` rows.

## Update: fixed-point test + transpose bug (per-iteration parity grind)

Seeding the corrected M-step with Fortran's *converged* `amicaout` solution (raw `W`, `S`,
`mean`, `mu`, `sbeta`, `rho`, `alpha`) and running one M-step (fixed-point test) established:

- **The corrected M-step is faithful.** Fortran's solution is a *stable* fixed point of the
  natural gradient: seeded LL = -3.40186 (== Fortran -3.402), 15 steps at lrate=0.05 hold
  -3.4019, and the stationarity condition `<g b^T>/dgm == I` holds to max 0.005. So score fp
  + exact-EM mu/beta + source-space natural gradient are correct.

- **W/A transpose bug (major, new).** Seeding required `A = inv(W_fortran.T)`: NG's internal
  `W` is the *transpose* of the true unmixing. `_forward` (`b = X.T @ W`) uses it correctly,
  but `transform` (`W @ X`), `get_unmixing_matrix` (returns `self.W`), and the validation
  comparison are transposed. At Fortran's solution, `corr(self.W.T, Fortran W) = 0.9997`
  vs `corr(self.W, Fortran W) ~ 0.5`. **Fix: report/compare `self.W.T`** (and fix `transform`
  to `self.W.T @ x`). This shared-with-NumPy transpose is a dominant cause of the low
  correlation number, independent of LL. (`amica_torch_ng.py:475` forward vs `:994` transform
  vs `:1001` get_unmixing.)

- **Newton is the remaining blocker.** At lrate ramping to 1.0, the Newton step diverges even
  seeded at the fixed point (0 fallbacks -- it takes the Newton direction, then overshoots).
  Root: the 2x2 solve `H[i,k] = (sk1*dA[i,k] - dA[k,i])/(sk1*sk2 - 1)` amplifies the small
  residual `dA = I - <g b^T>/dgm` for source pairs whose `sk1*sk2` is barely > 1 (near-marginal
  positive definiteness). Fortran is stable at lrate=1.0. Needs: match Fortran's Newton
  curvature exactly at the fixed point (so the residual and marginal pairs vanish), and/or a
  gentler ramp. Natural-gradient alone (no Newton) reaches ~-3.47 (vs Fortran's pre-Newton
  -3.44) with ~0.5 correlation, so Newton is required for >0.95.

## Remaining work

The confirmed fixes remove the divergence and make Newton fire, but a residual per-iteration
co-adaptation drift still leaves the natural-gradient phase at ~-3.47 vs Fortran's -3.44 (and
Newton overshoots at lrate=1.0 from that off-track point). Closing it needs a per-iteration
comparison against Fortran ground truth (re-run the binary with `do_history=1`/`histstep=1`,
seed the Python model from Fortran's own per-iteration state + sphere, and diff a single
M-step to localize the drift). Fix recipe, ordered:

1. First-order M-step re-port: score `fp` + source-space NG + exact-EM mu/beta/alpha + rho
   psi-factor + c update.
2. Symmetric ZCA sphere; correct `do_approx_sphere` semantics.
3. Revalidate Newton finalization against Fortran's baralpha/denominator terms (the base
   class's "denominators cancel to dgm" simplification is algebraically correct -- verified --
   but must be rechecked once the first-order params are on-track).
4. Fortran-faithful lrate control (hold 0.05 in the NG phase, ramp to `newtrate` under Newton,
   ratchet only after `max_decs`).
5. Apply the same fixes to `pyAMICA.py`; re-baseline `test_ng_backend.py` tests against
   Fortran, not the (buggy) NumPy reference.
6. Fix the validation-harness metric to compare `W@sphere`.
