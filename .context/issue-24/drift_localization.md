# Issue #24 drift localization: which M-step term drifts from Fortran

**Date:** 2026-07-02. Follow-up to `findings.md` (which proved the -3.46 gap is a residual
per-iteration M-step bug, not init-basin). This doc localizes the drifting term(s).

## Method

Teacher-forced per-iteration diff (`localize_drift.py`). Fortran's true NG trajectory
(`do_newton=0`, `lrate=0.05`) is obtained by running the binary with `max_iter=1..K` from the
Python init (`do_history` is broken -- it fails to `mkdir out/history` and aborts, so we re-run
incrementally). For each k we seed `CorrectedNG` from Fortran's state at iter k, run ONE M-step,
and compare each parameter to Fortran's state at iter k+1. Because every comparison restarts
from a known-good Fortran state, errors do not compound; the per-param one-step residual
localizes the drift.

Two payoff harnesses confirm each fix end-to-end with lrate pinned at 0.05 (the only regime the
fit-loop ratchet does not mask): a rho-only fix and a full rho fix.

## What drifts, and what does not

| param | one-step residual (k=0, from identical init) | verdict |
| ----- | -------------------------------------------- | ------- |
| W / A (unmixing) | 1.6e-6 defect | FAITHFUL (natural-gradient A update is correct) |
| alpha | ~0 | faithful |
| c     | ~3.6e-16 (Fortran c is machine zero) | faithful; the omitted c update is NOT the drift |
| **rho** | 0.149 (37.5% of the step) | **BUG 1+2 (below)** |
| **mu**  | 4.5e-3 (~1.3% of the step), grows with k | **BUG 3 (dominant, open)** |
| **beta**| grows with k, exceeds the step by ~k=8 | **BUG 3 (open)** |

## CONFIRMED BUGS

### Bug 1 -- rho numerator is missing a factor of `rho` (proven)

Fortran (`amica17.f90:1560-1578`, non-MKL branch) builds the rho numerator as:
`tmpy=|y|` -> `logab=ln|y|` -> `tmpy=|y|^rho` -> **`logab=ln(|y|^rho)=rho*ln|y|`** ->
`drho_numer = sum(u * tmpy * logab) = rho * sum(u*|y|^rho*ln|y|)`.

The prototype computes `sum(u*|y|^rho*ln|y|)` -- **missing the leading `rho`**.

Proof (`probe_rho.py`): seed the init, run one Fortran step. Python's iter-1 rho is off by
37.5% of the step without the factor, and matches Fortran to **1.7e-5** with it.

### Bug 2 -- rho per-component mask zeros the numerator at the clamp boundary (confirmed)

The prototype masks `mask = (rho != 1.0) & (rho != 2.0)` and zeros the ENTIRE numerator term for
any component at rho=1.0 or 2.0. Fortran has no such per-component skip; it only guards
per-SAMPLE underflow (`where(|y|^rho < eps) logab = 0`, `:1570`).

Fingerprint: with Bug 1 fixed, the teacher-forced `rho_abs` is at float precision (~7e-6) for
k=0-3, then jumps to **exactly 5.00e-2 = rholrate** at k>=4 -- precisely when a component's rho
first hits the `minrho=1.0` clamp. Fixing the mask (compute the term for all components, use only
the per-sample underflow guard) removes that jump.

### But Bugs 1+2 are NOT the dominant descent cause

End-to-end with lrate pinned at 0.05 (`verify_rhofix.py`, `verify_fullrho.py`):

```
rho fix          | pinned-NG endpoint (Fortran = -3.4269, ascending)
none             | -3.4974   (descends)
factor only      | -3.4972
factor + no mask | -3.4974
```

Fixing rho barely moves the endpoint. The descent is driven by BUG 3.

## OPEN: Bug 3 -- mu/beta exact-EM denominator drift (dominant)

The natural-gradient A/W update is faithful (defect 6e-5) and the mixture-summed score is right,
but the per-mixture **mu** and **beta** updates carry a small systematic residual that compounds
into the descent. Characterization (`probe_mu.py`):

- Present with AND without `doscaling` (4.47e-3 vs 4.57e-3) -> it is the exact-EM update, not the
  column rescale.
- Concentrated in the OUTER mixtures (mu ~ +/-1: residual 4.5e-3; the center mixture mu~0:
  residual 2e-5). Scales with |mu|.
- Too large (4.5e-3) to be float64 summation-order noise (~1e-8), so it is a real
  formula/definitional difference, not numerics.
- The mu update is `mu += dmu_numer/dmu_denom`, `dmu_denom = sbeta*sum(ufp/y)` (`amica17.f90:1537,
  1978`). The numerator `sum(ufp)` is per-mixture and NOT independently verified (W parity only
  confirms the mixture-SUMMED `g`), so the residual could live in the per-mixture numerator or in
  the singular `sum(ufp/y)` denominator (dominated by small-y `~|y|^(rho-2)` terms).

Next step to close it: seed from a Fortran state where rho has spread, dump Python's per-mixture
`dmu_numer` and `dmu_denom`, and compare against a hand-computed Fortran reference for a single
component (the same tactic that nailed Bug 1). Check the `rho<=2` vs `>2` denominator branch and
any small-y handling.

## FLAGGED: reference-source oddity (not our bug, note for anyone recompiling)

`amica17.f90:1465` (non-MKL `fp`): `vrda_exp((rho - dble(0.0))*tmpvec, ...)` -> `|y|^rho`, whereas
the MKL branch `:1462` uses `(rho - dble(1.0))` -> `|y|^(rho-1)` (the mathematically correct GG
score `fp = rho*sign(y)*|y|^(rho-1)`). The `amica15mac` binary behaves per the correct exponent
(our `_score = rho*sign(y)*|y|^(rho-1)` keeps W parity at 6e-5), so the shipped binary is fine,
but a non-MKL recompile from this source would compute the wrong score. Flag before rebuilding.

## BUGS TO FIX (when porting the corrected M-step into production)

- [ ] **Bug 1 (rho factor)** -- `corrected_mstep_prototype.py:102-110`: `drho_n` term must be
      `rho_h * |y|^rho * ln|y|` (add the leading `rho`). Mirror in the shipped
      `amica_torch_ng.py` rho path and `pyAMICA.py`.
- [ ] **Bug 2 (rho mask)** -- `corrected_mstep_prototype.py:104-108`: drop the per-component
      `(rho!=1)&(rho!=2)` mask; keep only a per-sample underflow guard (Fortran `:1570`).
- [ ] **Bug 3 (mu/beta denominator, dominant)** -- OPEN, needs root-cause. `mu`/`beta` exact-EM
      update in `_get_block_updates`/`_update_parameters`; residual is |mu|-dependent, in the
      `sbeta*sum(ufp/y)` denominator region.
- [ ] **Flag (reference)** -- `amica17.f90:1465` `(rho - dble(0.0))` should be `(rho - dble(1.0))`
      to match the MKL branch; only matters for a non-MKL rebuild.

## Reproduce

```
# teacher-forced per-param residual (add --fixrho to apply Bug 1):
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python .context/issue-24/localize_drift.py [run_dir] [--fixrho]
```
