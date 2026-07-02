@AGENTS.md

# Claude Code Notes

The project's instructions, architecture, environment, and conventions live in `AGENTS.md`
(imported above). Only Claude Code-specific guidance belongs here.

## Environment reminder
Canonical env is UV (see AGENTS.md). The legacy conda env still exists as `torch-312`; migration
to UV is tracked in `.context/plan.md`. Prefer `PYTORCH_ENABLE_MPS_FALLBACK=1` when running on MPS.

## Skills & commands
- `/review-pr` (pr-review-toolkit) before finalizing any PR; run reviewers with the model set to Sonnet.
- `/plan` for detailed implementation planning on non-trivial features.
- ADRs live in `.context/decisions/`; copy `0000-template.md` to start a new one.

## Validation
Correctness = parity with the Fortran binary. Use `validate_implementations.py` (real sample data,
never synthetic) as the source of truth when comparing backends.
