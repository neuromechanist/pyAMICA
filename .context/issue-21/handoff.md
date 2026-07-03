# Issue #21 implementation handoff

Pick-up doc for implementing the NG-backend M-step fix (chosen path "b": implement the proven
fixes in-code). Diagnosis is complete; this is the engineering plan.

## Status in one paragraph

The NG backend's first-order M-step diverges from iteration 1 because it uses the density
derivative `dpdf` where Fortran uses the score `fp`; the fix (plus exact-EM mu/beta, source-space
natural gradient, symmetric ZCA sphere, rho psi-factor, and a W-transpose in reporting) is fully
diagnosed and validated against Fortran's converged solution. What remains is (1) porting the
validated M-step into the production files and re-baselining the tests, and (2) the one open
gate-blocker: Newton curvature bit-parity at lrate=1.0 (tracked as #24). See
`root_cause.md` for the full evidence; this doc is the how-to.

## Coordinates

- **Branch:** `feature/issue-21-ng-mstep-parity` (off epic `feature/issue-9-epic-torch-parity`).
- **PR:** #23 (draft, docs-only so far) -> epic branch.
- **Issues:** #21 (this work), #24 (Newton bit-parity / init-basin verification via Fortran load_*).
- **Env:** UV. Run parity work on CPU/float64: `PYTORCH_ENABLE_MPS_FALLBACK=1 uv run pytest ...`.
  NG needs `device="cpu"` for float64 (MPS lacks it).
- **Fortran reference solution:** `pyAMICA/sample_data/amicaout/` (200-iter run, LL -3.402).
  Raw files (`W`, `S`, `mu`, `sbeta`, `rho`, `alpha`, `gm`, `c`, `mean`) are float64,
  column-major (`np.fromfile(...).reshape(shape, order="F")`), in Fortran's internal order (NOT
  re-sorted like loadmodout15.m).

## The validated fix (port these into `amica_torch_ng.py`)

The reference implementation is `corrected_mstep_prototype.py` in this folder -- it is a
`AMICATorchNG` subclass that PASSES the fixed-point test. Port its method bodies into the real
class. Each change with its Fortran line ref:

1. **Score, not density derivative.** In `_get_block_updates`, use `fp = _score(y, rho)` for the
   first-order terms, NOT `dpdf`. `p'(y) = -fp(y)*p(y)`, so `dpdf` inverts the update sign.
   (`amica17.f90:1467`.) `_score` already exists in the module.

2. **Exact-EM mu/beta (new accumulator contract).** Replace single `dmu`/`dbeta` with numer+denom:
   - `dmu_numer = sum(u*fp)`, `dmu_denom = sbeta*sum(u*fp/y)`  ->  `mu += dmu_numer/dmu_denom`
     (no lrate). (`:1532,1537,1978`)
   - `dbeta_numer = sum(u)`, `dbeta_denom = sum(u*fp*y)`  ->  `sbeta *= sqrt(dbeta_numer/dbeta_denom)`.
     (`:1550,1556,1993`)
   - `alpha = dalpha_numer / sum_j dalpha_numer` (already correct in the shipped code). (`:1891`)
   Here `u = v_h * z` (model-weighted mixture responsibility), `y = sbeta*(b-mu)`.

3. **Source-space natural gradient + sign.** Replace data-space `dA = X @ (v*g)` with
   `dWtmp[h] = g^T @ b` (g outer b, both source-indexed; `:1592`) and update
   `A -= lrate * A @ (I - <g b^T>/dgm)` (SUBTRACT; `:1800-1917`). `g_i = sum_j sbeta_j*u*fp`
   (`:1493`).

4. **Symmetric ZCA sphere.** In `_preprocess`, `do_approx_sphere=True` must give `V D^-1/2 V^T`
   (symmetric; Fortran's default, matches the amicaout `S` to 5e-6), NOT the current PCA
   `D^-1/2 V^T`. The Python `do_approx_sphere` semantics are currently inverted vs Fortran
   (`:480-481`). Note `sample_params.json` sets `do_approx_sphere: false` -- that is wrong for
   parity with amicaout (which used Fortran's default = symmetric ZCA); set it true / omit it,
   and make the gate test use the default.

5. **rho psi-factor.** `rho += rholrate*(1 - (rho/psi(1+1/rho))*drho_numer/drho_denom)` with
   `drho_numer = sum(u*|y|^rho*log|y|)`, `drho_denom = sum(u)`. The shipped code omits the
   `1/psi(1+1/rho)` (digamma) factor. (`:2013-2014`) Floor the denom and nan_to_num for stability.

6. **Transpose fix (reporting).** NG's internal `self.W` is the TRANSPOSE of the true unmixing
   (`_forward` uses `X.T @ W`, which is correct; but `transform`, `get_unmixing_matrix`, and the
   validation comparison are transposed). Return/compare `self.W.T`. At Fortran's solution this
   takes correlation from ~0.5 to 0.9999. Fix `transform` to `self.W.T @ x` and
   `get_unmixing_matrix` to `self.W.T`; `get_mixing_matrix` adjusts accordingly.

## Ordered plan

1. **Port the M-step** (changes 1-3,5) + **sphere** (4) + **transpose** (6) into
   `amica_torch_ng.py`. Validate with a NEW in-code test that mirrors `fixed_point_test()` in the
   prototype: seed amicaout, assert seeded LL == -3.40186, natural-grad holds -3.402 for 10 steps,
   `corr(get_unmixing_matrix().T? -- already transposed, so compare directly)` >= 0.99. This is
   the green milestone; commit here.
2. **Re-baseline `tests/torch_tests/test_ng_backend.py`.** The current
   `test_sufficient_stats_match_numpy_reference` / `test_newton_*_match_numpy_reference` assert
   NG == NumPy on the OLD (buggy) accumulator keys (`dmu`, `dbeta`, `dA`). They must become
   NG-vs-Fortran parity tests (the fixed-point test above is the right shape), because the NumPy
   reference is equally buggy. Do not "fix" the tests to keep passing against NumPy.
3. **Mirror the fix in `pyAMICA.py`** (same score/exact-EM/source-space/sphere/rho changes;
   `_get_block_updates` ~611-755, `_update_parameters` ~757+). Keep NG and NumPy in sync.
4. **Fix `validate_implementations.py::compare_results`** to compare the basis-invariant total
   spatial filter `W@sphere` (with the transpose), not raw sphered-space `W` rows.
5. **Newton bit-parity (#24, the gate-blocker).** With 1-4 done, the natural-gradient phase is
   faithful (fixed-point stable) but plateaus ~-3.47; Fortran's Newton breaks through to -3.40 at
   lrate=1.0 while ours overshoots (the 2x2 solve `H=(sk1*dA[i,k]-dA[k,i])/(sk1*sk2-1)` amplifies
   the residual for near-marginal-posdef pairs). Ruled out: c-center (Fortran's converged c is
   ~1e-16), sphere, mean. Next: build a per-iteration parity harness (Fortran `do_history` writes
   only history/16 reliably -- prefer the deterministic-seed + `load_*` route from #24 to start
   Fortran from the Python init) and diff the Newton curvature (`sigma2`/`kappa`/`lambda`) and the
   lrate schedule (Fortran holds 0.05 through the NG phase, no ratchet decay; ramps to newtrate
   under Newton; ratchets only after `max_decs` consecutive decreases -- the shipped fit() halves
   on EVERY decrease, which starves Newton). Only when Newton reaches -3.40 does the >0.95 gate
   flip (`test_end_to_end_correlation_vs_fortran`, currently strict xfail -- do NOT flip it early).

## Gotchas

- **Do not flip the xfail gate** until a real fit reaches >0.95; it is a strict xfail on purpose.
- The accumulator-contract change breaks the current unit tests by design -- re-baseline against
  Fortran, not NumPy.
- `_forward` (`X.T @ W`) is already correct; do not "fix" it. The transpose is only in
  transform/get_unmixing/compare.
- Keep `device="cpu"`, `dtype=float64` for all parity runs and tests.
- The reference prototype is a DIAGNOSTIC, not shippable as-is (it subclasses and duplicates); port
  the logic, don't import it.

## Validation commands

```
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python .context/issue-21/reproduce_root_cause.py      # score bug
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python .context/issue-21/corrected_mstep_prototype.py # fixed-point + fit
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run pytest pyAMICA/tests/torch_tests/test_ng_backend.py
```
