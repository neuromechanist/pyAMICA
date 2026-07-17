# Phase 4 MATLAB gate: MIR/PMI visualizations (issue #136, epic #133)

## The contract

Same bidirectional standard as [issue #155](../issue-155/matlab_interop_verification.md),
adapted to plots. Plots have no byte-level contract, so the gate has two levels:

1. **Data gate**: every quantity we plot must match MATLAB's where an oracle exists.
2. **Visual gate**: our figure and MATLAB's, rendered on the same real data, must be
   a meaningful representation of the same thing (a human judgement call, made by
   the user; accepted 2026-07-16).

MATLAB cannot run in CI, so this is recorded rather than automated.

## Licensing posture (why this gate exists at all)

`pop_topohistplot.m` and `pop_modPMI.m` carry explicit **GPL-2.0-or-later** headers
(Copyright Ozgur Baklan, SCCN, INC, UCSD). `modprobplot.m`, `minfojp.m`, `LLt2v.m`,
`smooth_amica_prob.m` carry no header but sit inside that GPL plugin, so they are
conservatively GPL too. pyAMICA is BSD-3-Clause, which is why Phase 2's PMI was a
clean-room reimplementation.

**Posture: run-and-observe only. No `.m` implementation source was read at any point.**
Everything below came from MATLAB `help` text (public API documentation), rendered
figures, and black-box input/output behaviour. That is what makes the MATLAB gate not
merely a check but the *mechanism* that keeps the reimplementation clean: it lets us
match observable behaviour without deriving from protected expression.

`pre_ICA_cleaning/getMIR.m` is **Apache-2.0** and was legitimately read (Phase 1
ported it with attribution; see `THIRD_PARTY_NOTICES.md`).

## Data gate: results

All on real bundled sample EEG (`eeglab_data.fdt`, 32ch x 30504, srate 128) and a
real 2-model `AMICATorchNG` fit. Compared with `np.array_equal`/`corrcoef`, never
`allclose` where an exact match was expected.

| Quantity | MATLAB oracle | Result |
|---|---|---|
| `mir()` | `getMIR.m` (Apache-2.0) | **1.7e-15 relative** (33.110492656163046 vs ...103 at N=30504; 37.541867541278549 vs ...663 at N=4096) |
| P(model \| data) | `LLt2v` | **1.4e-14** vs our `softmax(Lht)` |
| `v` (log10 model odds) | `loadmodout15.m:284` | **bit-exact** (max diff 0.0) |
| Hanning-smoothed probability | `smooth_amica_prob` | **r = 0.988577** (1 s), **r = 0.959376** (5 s); mean abs diff 0.014 / 0.045 |
| `pairwise_mi` | `minfojp` via `pop_modPMI` (GPL, black-box) | **r = 0.9887** off-diagonal, on identical signals |

Notes on the two correlation-rather-than-equality rows:

- **Smoothing**: `max|diff|` is ~1.0 and that is expected, not alarming. Near a model
  switch the probability is near-binary, so a sub-sample timing difference flips 0<->1.
  Mean abs diff (0.014) and correlation are the meaningful measures. We deliberately do
  not replicate MATLAB's endpoint convention (see divergences below).
- **PMI**: our estimator is clean-room, so it is a *different* estimator. Correlation is
  the right bar; equality would be suspicious. **Do not tune ours toward theirs** — that
  would convert a clean-room reimplementation into a derivation.

## Visual gate: accepted

Side-by-side renders on the same data are committed here:

- `cmp_modprob.png` — MATLAB `modprobplot` vs `pyAMICA.viz.plot_model_probability`
  (1 s smoothing). Same two stacked panels, same switching times (~1.4, 3.4, 6.5, 9.3,
  12.5, 15.7 s), same log-likelihood trace and range (-105 to -125), seconds axis.
- `cmp_pmi.png` — MATLAB `pop_modPMI` vs `pyAMICA.viz.plot_pmi_heatmap`, rendered on
  **identical signals** (MATLAB's own `EEG.icaact`) so the comparison isolates the
  estimator and the ordering. Both show the same dependent-subspace cluster near the
  centre with the same radiating cross pattern, at the same MI scale.

Accepted deliberate differences: viridis vs jet, we add a colorbar (MATLAB has none),
0-based component labels (pythonic; MATLAB is 1-based), and a different ordering
algorithm (ours greedy nearest-neighbour chain vs MATLAB's iterative cost minimisation,
`cost = 20.0533 -> 19.5856 -> 18.9268`).

## Deliberate divergences (documented so nobody "fixes" them toward MATLAB)

1. **Smoothing endpoints.** MATLAB pins the exact first/last sample to the raw
   *unsmoothed* input value (a MATLAB `smooth()` idiom). We return a locally averaged
   value there. Ours is arguably more correct (the first sample of a smoothed signal
   should not be unsmoothed), and matching exactly would require reading GPL source.
2. **MI diagonal.** MATLAB zeroes it. Our `pairwise_mi` diagonal is self-entropy
   (~2.83 vs off-diagonal ~0.06), so `plot_pmi_heatmap` masks it (`mask_diagonal=True`).
   Plotting it unmasked destroys the colour scale and hides all structure.
3. **Negative MI.** MATLAB's bias correction yields small negatives (-0.006); ours is
   strictly positive. Expected estimator difference, not a bug.

## Traps found the hard way (re-read before touching this)

1. **`mInfoMatrix` is stored ALREADY REORDERED** by `mInforOrders`. Comparing it raw
   against a natural-order matrix gives **r = -0.13 and 0/8 overlap in top dependent
   pairs**, which reads exactly like "our PMI is broken". Un-permuted it is **r = 0.9887**.
   Always un-permute (`inv = np.argsort(order-1); mi[np.ix_(inv, inv)]`) first.
2. **A naive Hanning smooth silently corrupts both plot edges.**
   `np.convolve(row, w/w.sum(), mode="same")` zero-pads, and since `Lht ~= -108` (nowhere
   near 0) the padding drags the first/last half-window toward zero: input[0] = -123.4701
   became **-60.39**. After the softmax that yields confidently WRONG model probabilities
   at the start and end of every plot — plausible-looking and completely wrong. Fix, used
   in `viz.py`: divide by the window overlap (`np.convolve(np.ones_like(row), w, "same")`).
3. **Getting sources from an `AmicaOutput` is error-prone.** The maintainer got this wrong
   while building this very gate: `W @ S @ (X - mean)` is NOT the sources (it produced a
   plausible heatmap with a 3x-wrong MI range, 0.195 vs 0.062). The canonical formula is
   `W[:, :, m].T @ (sphere @ (X - mean) - c[:, m])` (`torch_impl/core.py: transform`).
   Note `loadmodout` additionally normalises and variance-reorders. If it caught the
   maintainer with full context, it will catch users — this is the argument for Phase 5
   (#137) exposing `model.pmi()` / `model.mir()` so nobody hand-rolls it.
4. **postAmicaUtility's own `loadmodout15.m` is broken on R2025b** (`Unmatched ']'`,
   line 120). pyAMICA's bundled copy works. `addpath` `pyAMICA/sample_data` LAST so ours
   shadows theirs, or the model never loads and every plot fails with a misleading
   "No AMICA solution found".
5. **`pop_topohistplot` is broken upstream** (`Unrecognized function or variable
   'showhist'` on current EEGLAB), so there is **no visual reference** for the topo/PDF
   plot. `plot_topo_pdf` is therefore a fresh design, which #136 already mandated.
6. **Do not read the figures to settle numeric questions.** The maintainer twice
   misread which model was active at t=0 from a low-res render and briefly believed the
   models were swapped. Three numeric checks (correlation +0.9886, direct values at t=0,
   and pixel-colour sampling showing orange's plateau starting at x=74 vs blue's at
   x=120) all confirmed there was no swap. Trust the data gate over the visual one.

## Reproducing

MATLAB R2025b at `/Volumes/S1/Applications/MATLAB_R2025b.app/bin/matlab -batch`.
References are NOT vendored (GPL); clone to a scratch dir:

```
git clone --depth 1 https://github.com/sccn/eeglab.git
git clone --depth 1 https://github.com/sccn/postAmicaUtility.git     # GPL: run, never read
git clone --depth 1 https://github.com/bigdelys/pre_ICA_cleaning.git # Apache-2.0: getMIR.m
```

Then: `addpath(genpath(eeglab)); addpath(postAmicaUtility); addpath(pyAMICA/sample_data)`
(last, per trap 4); `eeglab nogui`; `pop_loadset` + `pop_loadmodout(EEG, <amicaout dir>)`;
`modprobplot(EEG, 1:num_models, smooth_sec, [])` returns `[v2plot, llt2plot]` — the exact
series it draws — and `pop_modPMI(EEG, 'models2plot', 1, 'order', true)` writes
`mInfoMatrix`/`mInfoVar`/`mInforOrders` into `EEG.etc.amica`. Figures are uifigures; export
with `exportapp`, not `print`.
