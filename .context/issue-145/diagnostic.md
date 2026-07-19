# Issue #145 diagnostic - near-singular-Hessian hypothesis REFUTED

## What was tested
Instrumented `AMICATorchNG` M-step to log, per iteration, per model: Hessian
conditioning (`min` over off-diagonal source pairs of `p_i*p_k - 1`, where
`p_i = sigma2_i*kappa_i`; a pair is near-singular as `p_i*p_k -> 1+`), the
accepted Newton step magnitude (`maxHoff`, `max_a_step`), and per-component
density params. Ran seed 42 (the collapser) and seed 7 (the matcher) to 2000
iters, CUDA float64, `do_newton=1`, on the full 70ch/k=152 data. Trajectories +
scripts on hallu: `/mnt/local/taskB/diag/{s42,s7}_traj.npz`,
`analyze_newton_145.py`.

## Hypothesis (refuted)
Weak components' Hessians graze singular (`p_i*p_k -> 1+`), the Newton
denominator `-> 0`, so a ~1e-9 per-step difference from Fortran is amplified by
`1/(prod-1)` into a basin-jumping step that is still technically posdef (accepted,
not a fallback).

## Evidence it is REFUTED
- **Conditioning never approaches the singular boundary.** `min(p_i*p_k)-1` goes
  2.24 (iter 50) -> 0.68 (iter 2000) for seed 42, 2.27 -> 0.59 for seed 7. The
  count of off-diagonal pairs in the marginal zone `(1, 1.02]` is **0 at every
  iteration** for both seeds. Nothing grazes singular.
- **No step blowup.** `maxHoff` stays ~1e-3..5e-2, `max_a_step` ~1e-3..3e-2, all
  run. `n_newton_fallbacks = 0` because the Hessian is comfortably posdef, not
  because it is marginally posdef.
- **The collapsed components are not the ill-conditioned ones.** Collapsed torch
  rows (seed 42, corr<0.9 vs F_d03): [0, 11, 26, 35, 42, 43, 52], with final
  `p_i*p_k` = 1.6-2.7 (well clear of 1) and 0 iterations near-singular. Fraction
  of iters (>=50) where the globally worst-conditioned component is a collapsed
  component: **0.003**.
- **Seeds 42 and 7 have nearly identical aggregate trajectories** (LL,
  conditioning, step magnitudes match at every logged iteration) yet seed 42
  collapses 7 components and seed 7 collapses 0 - on the *same* intrinsically-weak
  components (0, 26, 35, 42, 43...). The difference is which basin those weak
  components settle into, not any measured numerical pathology.

## Reframed finding
The `do_newton=1` divergence is **basin selection on the ~7 weakest
(under-determined) components**, via well-conditioned Newton steps. Torch's
extended optimization settles them into a different configuration than Fortran's,
at an **equal-or-higher LL** (seed 42 collapser final LL -3.69759 > seed 7 matcher
-3.69780; both `best_iter=1999`, LL still rising at 2000). Combined with the
issue's established fact that correlation *degrades* from ~300 iters (0.98-0.999)
to 2000 iters (0.87-0.97), this says the extended Newton phase moves torch *away*
from Fortran's basin into a nearby higher-LL one, on the weak components only.
Torch has more run-to-run spread on these components than Fortran (within-torch
0.963 vs within-Fortran 0.9997).

## Why this matters for the fix
There is no conditioning/Hessian-damping lever to pull (zero marginal pairs), so a
posdef-threshold or step-cap "fix" would be a pure no-op on this data. The real
question is now: is the torch-vs-Fortran divergence (a) a compounding
implementation difference in the well-conditioned steps (per-step ~1e-9
differences accumulating along the flat weak-component directions over ~1700
Newton steps), or (b) genuine multi-basin structure that Fortran's dynamics is
more consistently attracted out of than torch's? These are distinguished only by a
**same-init torch-vs-Fortran trajectory comparison** (init torch from Fortran's
exact starting A/sphere and log both), which no prior run has done - every
comparison so far used independent inits.
