# LLt interop verification: pyAMICA <-> EEGLAB `loadmodout15.m` (issue #155, PR #156)

## The contract

The acceptance bar for the `LLt` work is bidirectional interop, not internal
self-consistency:

1. **pyAMICA's reader must understand Fortran's output.**
2. **EEGLAB's reader (`loadmodout15.m`) must understand pyAMICA's output.**

PR #156's automated tests only pin direction 1 (via the real bundled
`sample_data/amicaout/LLt` fixture). They do **not** pin direction 2: the
round-trip tests prove pyAMICA's writer agrees with pyAMICA's *own* reader,
which a self-consistently-wrong writer/reader pair would also satisfy. Only
the genuine MATLAB reader can close that.

This matters because issue #92 caught a **real** column-major bug exactly this
way: `loadmodout15.m` read scrambled mixture params (per-component `alpha` did
not sum to 1) that code review had missed. `LLt` is a new file in the writer
and had never been through that gate. See `.context/scratch_history.md`.

MATLAB cannot run in CI, so this is a manual verification, recorded here the
same way #92's was.

## Method

MATLAB R2025b, real `loadmodout15.m` from `pyAMICA/sample_data/`, run against
three directories:

- `pyamica_m1` -- pyAMICA `AMICATorchNG` single-model fit, real bundled sample
  EEG (`eeglab_data.fdt`, 32ch x 4096 samples), seed 1, 20 iters.
- `pyamica_m2` -- same data, 2 models, seed 4, 8 iters (non-trivial `mod_prob`,
  so the reader's gm-based model reordering is actually exercised).
- `amicaout` -- the genuine Fortran-produced fixture bundled in the repo.

Each result was compared against `pyAMICA.numpy_impl.load.loadmodout`'s view of
the same directory with `np.array_equal` (not `allclose`).

## Result: bit-exact in both directions

```
pyamica_m1  Lht (1, 4096)  equal=True | Lt (4096,)  equal=True | max|dH|=0.0 max|dT|=0.0
pyamica_m2  Lht (2, 4096)  equal=True | Lt (4096,)  equal=True | max|dH|=0.0 max|dT|=0.0
fortran     Lht (1, 30504) equal=True | Lt (30504,) equal=True | max|dH|=0.0 max|dT|=0.0

EEGLAB loadmodout15 == pyAMICA loadmodout, bit-exact, all cases: True
pyamica_m1  single-model Lht[0]==Lt via MATLAB: True
fortran     single-model Lht[0]==Lt via MATLAB: True
```

Corroborating details:

- 2-model `mod_prob` agrees: MATLAB `[0.554689 0.445311]` vs pyAMICA
  `[0.5547 0.4453]`. Both readers apply the same gm-descending model
  reordering to `Lht`, so the model axis stays aligned with `W`/`mod_prob`
  across the two implementations.
- Genuine Fortran fixture reads to `Lt mean = -108.859935652062` under BOTH
  readers, matching `nw * final_ll = 32 * -3.40187` independently.
- MATLAB reads `Lt` as `(1, N)` and pyAMICA as `(N,)`; that is MATLAB's
  everything-is-2D convention, not a layout difference (values compare equal
  after `ravel`).

## Why it agrees

`loadmodout15.m:119-124` does `fread(...,'double')` then
`reshape(LLt, num_models+1, N)`. MATLAB's `reshape` is column-major, so this is
`order="F"`. pyAMICA's writer (`numpy_impl/load.py:write_amicaout`) does
`vstack([Lht, Lt]).ravel(order="F")`, and its reader does
`reshape(num_models+1, -1, order="F")`. All three agree with Fortran's
`write_output` (`amica15.f90:2308-2333`), which writes per timepoint each
model's `modloglik` then the total.

`loadmodout15.m:284` then derives `v(h,:) = 0.4343 * (Lht(h,:) - Lt)`, the
log10 model odds. The `0.4343` scaling is the reader's job; the writer emits
natural-log values, which is what both readers expect.

## Reproducing

MATLAB is not on CI, so re-run manually if the `LLt` layout, the writer, or
`loadmodout` ever change:

1. Fit a real single-model and multi-model model, `write_amica_output` each.
2. `matlab -batch` -> `loadmodout15('<outdir>/')` (note the trailing slash; the
   function concatenates paths directly).
3. Compare its `Lht`/`Lt` against `loadmodout`'s with `np.array_equal`.
4. Also run it against `sample_data/amicaout/` so both readers are checked
   against the same genuine Fortran bytes.

Do not weaken step 3 to `allclose`: both readers are parsing identical bytes,
so anything short of exact equality means a real layout or convention bug.
