# Issue #145 -- diagnostic setup and pre-diagnostic facts

Goal: root-cause why `AMICATorchNG` (do_newton=1) diverges from Fortran on a
handful of weak components by 2000 iters on data-adequate real EEG (seed 42
collapses: 10-11/70 comps < 0.9 vs Fortran; seeds 7/13 match at ~0.99). Fortran
is self-consistent (within-Fortran 0.9997), so it is a valid oracle here.

## Reproduction is confirmed (matches team-lead numbers)
Full ds002718 sub-002, 70ch x 747,750 frames, k=152.6. CUDA float64, 2000 iters.
- within-Fortran (2 runs): mean 0.9997, min 0.9983
- seed 7 vs Fortran:  mean 0.9956, min 0.955, 0/70 < 0.9   (MATCH)
- seed 13 vs Fortran: mean 0.9931, min 0.866, 1/70 < 0.9   (MATCH)
- seed 42 vs Fortran: mean 0.9429, min 0.485, 10-11/70 < 0.9 (COLLAPSE)
- All seeds: n_newton_fallbacks = 0 (collapse happens inside ACCEPTED posdef
  Newton steps, never via the natural-gradient fallback).
- seed 42 final LL -3.6975 is marginally HIGHER than 7/13 (-3.6978): basin
  selection, not worse convergence. keep_best faithfully preserves the wrong
  (but higher-LL) basin.

## Clue from the reproduction log (torch.log)
- seed 42 ended `end_newtrate=0.25`; seeds 7/13 ended `end_newtrate=1`.
  newtrate is halved by the LL-decrease ratchet (core.py:1856-1861) after
  `maxdecs` decreases. So seed 42 suffered repeated Newton OVERSHOOTS (LL
  non-monotonicity) that 7/13 did not -- consistent with accepted-huge-step.

## Config is NOT the mismatch (verified)
Reference Fortran run input.param (/tmp/ds002718_full_parity_d03rs2rm/run_0)
uses EXACTLY the torch fit config: newtrate=1.0, newt_start=50, newt_ramp=10,
lratefact=0.5, max_decs=3, lrate=0.05, rholrate=0.05, rholratefact=0.5,
rho0=1.5, do_newton=1, do_reject=0, num_mix=3, block_size=512, doscaling=1
scalestep=1, do_mean/do_sphere/doPCA=1 pcakeep=70. Both take full (newtrate=1.0)
Newton steps on an identical schedule. So the divergence is purely numerical.

## Newton math is algebraically identical to Fortran (verified by source read)
- Accumulation (core.py:1072-1076 vs amica15.f90:1423-1490): dsigma2=sum v*b^2,
  dkappa=sum(ufp*fp)*sbeta^2, dlambda=sum u*(fp*y-1)^2. Identical.
- Finalization (core.py:_finalize_newton_stats vs amica15.f90:1650-1662): the
  baralpha/denominator masses cancel, leaving sigma2=dsigma2/dgm,
  kappa=sum_j dkappa/dgm, lambda=sum_j(dlambda + dkappa*mu^2)/dgm. Identical.
- 2x2 solve + guard (core.py:_newton_direction vs amica15.f90:1703-1718):
  H[i,i]=dA[i,i]/lambda[i]; sk1=sigma2[i]*kappa[k], sk2=sigma2[k]*kappa[i];
  H[i,k]=(sk1*dA[i,k]-dA[k,i])/(sk1*sk2-1) if sk1*sk2>1 else posdef=False.
  BYTE-identical guard (prod>1.0). NumPy backend (numpy_impl/core.py:1206-1225)
  is identical too.
- Any torch-vs-Fortran difference is float64 summation-ORDER noise (~1e-9),
  which is exactly what the near-singular-amplification hypothesis needs.

## `minhess` is dead code
amica15_header.f90:73 declares `minhess = 1.0e-5` but it is NEVER referenced in
amica15.f90 or amica17.f90 (like Spinv2 / do_choose_pdfs moment buffers). So
there is NO Fortran Hessian floor to port -- Fortran uses the raw prod>1 guard.

## Key conditioning identity (for attribution)
prod[i,k] = sk1*sk2 = (sigma2[i]*kappa[i]) * (sigma2[k]*kappa[k]). Define
p_i = sigma2[i]*kappa[i]. A pair (i,k) is near-singular (prod -> 1+) exactly when
p_i*p_k -> 1. So the components with the SMALLEST p_i drive the boundary; the
diagnostic reconstructs per-component p_i from saved sigma2,kappa to attribute
near-singular pairs to specific (collapsing?) components.

## Tests that must stay green (bit-exact #24 parity)
- tests/torch_tests/test_ng_backend.py: test_newton_stats_match_numpy_reference,
  test_newton_mstep_matches_numpy_reference, test_newton_direction_matches_formula
  (asserts posdef True on a well-conditioned dA, posdef False on tiny
  sigma2/kappa). Any guard/margin change must keep the well-conditioned case
  posdef=True and the NumPy<->NG paths bit-identical.

## Sample-data (32ch #24 fixture) Newton conditioning -- the margin CEILING
Ran the exact test_end_to_end_correlation_vs_fortran config (32ch, block_size=512,
do_newton, newt_start=50, newtrate=1.0, lrate=0.05, 100 iters, CPU float64):
- n_newton_fallbacks=0, final_ll=-3.41125 (matches #24)
- Newton fires iters 50-99; min ACCEPTED (prod-1) over ALL of them = **2.09**
  (prod >= 3.1), min p_i=sigma2*kappa = 1.73, n_marginal(<1.05)=0 every iter.
=> The sample data is VERY well-conditioned. ANY posdef-guard margin up to ~2.0
is a complete no-op there (every pair has prod-1 >= 2.09 >> margin), so
n_newton_fallbacks stays 0 and the #24 correlation stays bit-exact. Generous
headroom for a margin; the binding constraint is the full-70ch FLOOR (seed 42's
danger-zone denom vs seed 7's min denom), which the diagnostic supplies.

## Diagnostic OUTCOME: near-singular hypothesis REFUTED (see diagnostic.md)
Ran the instrumented DiagNG subclass (per-iteration min(prod-1) over off-diag
pairs, min accepted denom, n_marginal, max|H_off|, per-column A step, and
per-component sigma2/kappa/lambda/mu/beta/alpha/rho) for seed 42 (collapser) and
seed 7 (matcher) to 2000 iters, CUDA float64, from the a6b967f baseline extracted
read-only to /mnt/local/diag145. Result: the Hessian NEVER approaches singular
(min(p_i*p_k)-1 stays 2.24 -> 0.68; zero marginal pairs at every iteration; the
collapsed components are well-conditioned, p_i*p_k = 1.6-2.7). The collapse is
BASIN SELECTION on the ~7 weakest components via well-conditioned Newton steps,
not a conditioning pathology. Full evidence in `diagnostic.md`.

### Consequence for the notes above
The margin/posdef-threshold fix reasoned about here (and the sample-data CEILING
of 2.09) is MOOT: with zero marginal pairs on the full data there is nothing for
a conditioning guard to catch, so any threshold/step-cap change is a pure no-op.
The source-read facts above (config identical to Fortran, Newton math
algebraically identical, minhess dead code, sample data well-conditioned) stand
and CORROBORATE the refutation -- there is no Fortran damping lever to port and no
numerical near-singular event to stabilize. Next step (team-lead directed): a
same-init torch-vs-Fortran trajectory comparison to localize the divergence.
Held pending that instruction; no further long hallu runs launched.
