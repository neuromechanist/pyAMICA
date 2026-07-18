# Issue #159 -- MATLAB interop verification (W / A / sbeta / rho)

Direction (2) of the interop contract: EEGLAB's real `loadmodout15.m` reads
pamica's (and Fortran's) bytes correctly. A pamica write->read round trip
cannot pin this -- it cancels the byte-order error -- so this is a manual MATLAB
run, recorded here (MATLAB is not on PATH and not in CI).

## Setup

- MATLAB `R2025b` at `/Volumes/S1/Applications/MATLAB_R2025b.app/bin/matlab`.
- Reader: pamica's own working `pamica/sample_data/loadmodout15.m`, added to the
  path LAST so it shadows any broken copy (postAmicaUtility's has a syntax error
  on R2025b). `loadmodout15` concatenates paths directly, so the outdir argument
  is passed with a trailing `filesep`.
- Scripts (reproducible): `.context/issue-159/gate_prep.py` (writes a real
  20-iter 2-model NG output, seed=4, and dumps Python `loadmodout` W/A/sbeta/rho),
  `gate_compare.m` (MATLAB `loadmodout15` on the same dirs), `gate_check.py`
  (element-wise compare).
- Inputs: (a) the genuine single-model Fortran fixture `sample_data/amicaout`;
  (b) the freshly written pamica 2-model output `.context/issue-159/gate_2model`.

## Result -- GATE PASS

Python `loadmodout` (a port of `loadmodout15.m`) now agrees with MATLAB
`loadmodout15` element-wise, on both directories:

| dir | array | max abs diff |
|---|---|---|
| fixture (1 model)   | W | 8.2e-14 |
| fixture (1 model)   | A | 2.2e-15 |
| fixture (1 model)   | sbeta | 3.0e-16 |
| fixture (1 model)   | rho | 0 |
| two_model (2 models)| W | 1.3e-13 |
| two_model (2 models)| A | 1.9e-15 |
| two_model (2 models)| sbeta | 3.2e-16 |
| two_model (2 models)| rho | 0 |

The multi-model `W` agreeing to 1e-13 is the load-bearing new result: it proves
the writer's genuine-Fortran layout (model axis slowest) is EEGLAB-readable, and
that `loadmodout` reads it back the same way MATLAB does. Before the fix these
disagreed by the internal transpose (`W` off, and `A`/`svar`/`origord` with it),
and `sbeta`/`rho` disagreed for `num_mix > 1`; the issue #159 comment records the
pre-fix numbers (LL recomputation off by 0.11, `rho` pairing max diff 0.969).

## Cross-check: the LL oracle (direction 1)

`test_loadmodout_reproduces_fortran_reported_ll` recomputes the fixture's own
reported converged LL (-3.4018730) from what `loadmodout` reads: -3.4018468 (5
sig figs; residual is the Fortran's block truncation). The shipped C-order read
gave -3.5167 (off by 0.11). Together the LL oracle and this MATLAB gate cover both
contract directions.
