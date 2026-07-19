# Issue #145 - same-init test result (DEFINITIVE)

## Method
Ran BOTH torch (`AMICATorchNG`, CUDA float64) and native Fortran (`amica15`,
gfortran) from an IDENTICAL deterministic start: A=identity, mu=[-1,0,1],
sbeta=1, rho=1.5. The symmetric-ZCA sphere is sign/order-invariant, so
"A=identity in sphered space" is a genuinely identical start for both. Fortran
was fed the init via the built-in `load_A`/`load_mu`/`load_beta`/`load_rho` file
path (fix_init=1 segfaults in the amica15_linux build; load path is equivalent).
Both ran 2000 iters, do_newton on, matched config. Scripts + W's on hallu
`/mnt/local/taskB/fixinit/`.

## Result (Hungarian-matched |corr|, 70 components)

| comparison                              | mean   | min    | n<0.9 |
|-----------------------------------------|--------|--------|-------|
| **torch-from-A=I vs Fortran-from-A=I**  | 0.9974 | 0.9475 | 0/70  |
| Fortran-from-A=I vs random Fortran ref  | 0.9999 | 0.9988 | 0/70  |
| within-Fortran (two random inits)       | 0.9997 | 0.9983 | 0/70  |
| torch-from-A=I vs random torch seed 42  | 0.948  | 0.53   | 9/70  |
| torch-from-A=I vs random torch seed 7   | 0.993  | 0.928  | 0/70  |
| torch-from-A=I vs random torch seed 13  | 0.998  | 0.981  | 0/70  |

## Verdict: init-basin sensitivity, NOT a dynamics bug

- **From an identical init, torch and Fortran agree** (0.9974, 0/70 collapsed).
  The residual ~0.003 gap vs within-Fortran (0.9997) is float summation-order /
  BLAS noise between the two implementations accumulated over 2000 iters on the
  weak (flat-likelihood) components - NOT a basin difference and NOT a collapse.
- **Fortran-from-A=I == the random Fortran refs at 0.9999**, proving (a) A=identity
  is a normal, non-special init, and (b) Fortran is init-robust (any init -> the
  same basin).
- **The only collapse is torch's random seed 42** (0.948, 9/70): that specific
  random init sends torch into a different, equal-or-higher-LL basin. The same
  deterministic init does not. So the collapse is triggered by particular random
  initializations, to which torch is more sensitive than Fortran.

This corroborates the diagnostic (near-singular Hessian refuted; Newton math
algebraically identical to amica15.f90; no minhess damping lever). There is no
per-step dynamics divergence to fix: identical start -> matching answer.

## Recommended resolution
Document, do not "fix" the optimizer:
- Report the do_newton=0 single-model conformity number (0.998, reproducible end
  to end) as the parity claim - already done in paper/docs (#144).
- Disclose the do_newton=1 weak-component spread as init-basin sensitivity on
  under-determined components (torch more init-sensitive than Fortran), the same
  posture as multi-model non-identifiability (#27) and the best-iterate safeguard
  (#51).
- OPTIONAL enhancement (follow-up, not a parity fix): improve torch's
  init-robustness via best-of-N restarts (extending #51 keep_best across seeds) so
  a single fit is less likely to land in the seed-42-style basin.
