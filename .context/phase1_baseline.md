# Phase 1 Baseline: Torch backend vs Fortran parity

Generated 2026-07-02 as the Phase 1 (issue #10) deliverable. Produced by
`validate_implementations.py`, which runs the real Fortran reference binary
(`pyAMICA/sample_data/amica15mac`, via Rosetta 2 on this arm64 Mac) and the
default PyTorch backend (`AMICATorch`, Newton disabled, no adaptive PDF) on
the real sample EEG data (`pyAMICA/sample_data/eeglab_data.fdt`, 32 channels,
30504 samples), then Hungarian-matches components and reports correlation and
log-likelihood metrics. No synthetic data was used.

The script required no code changes to run end-to-end; it already worked
once the `load_eeglab_data` data-loading bug (see below) was fixed.

## Commands used

```bash
cd /Users/yahya/Documents/git/phase1-harness
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python validate_implementations.py --max-iter 20 --seed 42
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python validate_implementations.py --max-iter 100 --seed 42
```

Device: MPS (Apple Silicon). Both runs use the same seed (42) and the same
sample data. `--max-iter` controls both implementations' iteration count
identically (the script passes the same value to each).

## Results

| Metric | 20 iterations | 100 iterations |
|---|---|---|
| Fortran final LL | -3.4456 | -3.4108 |
| PyTorch final LL | -44.7738 | -43.8014 |
| LL ratio (PyTorch / Fortran) | 12.99x | 12.84x |
| Mean component correlation (Hungarian-matched) | 0.8351 | 0.4313 |
| Min component correlation | 0.6980 | 0.2126 |
| Max component correlation | 0.9445 | 0.6896 |
| Std component correlation | 0.0582 | 0.0994 |
| Mixing matrix (A) relative Frobenius error | 0.7024 | 1.6312 |

Fortran converges effectively immediately (LL is already near its converged
value at iteration 20); the two iteration counts mainly show how the PyTorch
trajectory evolves.

### Per-channel-normalized LL

Fortran reports `LL = LLtmp2 / (num_samples * nw)` (nw = 32 channels here);
the torch backend normalizes by `num_samples` only (`amica_torch.py:204`,
tracked as AGENTS.md Known Issue #2). Dividing the torch final LL by nw=32
for a rough apples-to-apples comparison: `-43.8014 / 32 = -1.369` vs
Fortran's `-3.4108`. This is closer but still off by roughly 2.5x, confirming
the LL gap is not fully explained by the missing `nw` normalization factor
alone; the design review in `.context/research.md` ("2026-07-02 design
review") identifies additional contributing bugs (swapped mixture
factorization, `alpha` collapsed to scalar, extra `logdet` term).

## Key finding: correlation gets worse with more iterations, not better

Mean correlation drops from 0.84 (20 iters) to 0.43 (100 iters) as the torch
backend runs longer. This is a concrete, measured illustration of the root
cause documented in ADR 0001 and the design review: the torch backend
optimizes NLL via Adam on a reparameterized surface, which is a materially
different trajectory from Fortran's natural-gradient EM fixed-point
iteration. Running the torch optimizer longer moves it further from, not
closer to, the Fortran solution. This is consistent with `.context/plan.md`'s
existing note ("Correctness is defined by parity with the Fortran binary, not
by convergence alone") and with the previously recorded correlation range
(~0.46-0.9, run-dependent); this run's 100-iteration mean (0.43) sits at the
low end of that range.

## Interpretation

These numbers reproduce the two parity blockers already tracked in
`AGENTS.md`/`.context/research.md`:

1. **LL scale/sign gap (~13x)**: reproduced almost exactly (12.84-12.99x vs
   the previously documented ~13x for the basic backend).
2. **Component correlation well below the 0.95 target**: reproduced (0.43
   mean at 100 iterations), and shown here to *degrade* with more
   optimization steps rather than improve, strengthening the case for the
   natural-gradient EM rewrite (ADR 0001) over incremental fixes to the
   current Adam-based optimizer.

Neither issue is a data-loading or plumbing bug; both are algorithm-level
(optimization objective and update rule) issues explicitly out of scope for
Phase 1, tracked for the Phase 2/3 rewrite (epic #9).

## Fortran binary note

The Fortran binary `amica15mac` is an x86_64 Mach-O executable; it runs
successfully on this arm64 Mac via Rosetta 2 with no special setup. No
degraded/skip path was needed — `run_fortran_amica()` in
`validate_implementations.py` already handles the "binary not found" case
gracefully (prints a warning and returns `None`, and `compare_results()`
falls back to reporting PyTorch-only metrics), but that path was not
exercised here since the binary was available and ran normally.
