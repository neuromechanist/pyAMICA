# Contributing to pamica

Thanks for your interest in contributing. pamica is a Python implementation of
Adaptive Mixture Independent Component Analysis (AMICA) that reproduces the
reference Fortran implementation. Because **numerical parity with the Fortran
reference is the definition of correctness**, contributions are held to that
standard rather than to "it converges."

## Getting help and reporting issues

- **Questions, bugs, and feature requests:** please open an issue on the
  [GitHub issue tracker](https://github.com/sccn/pAMICA/issues).
- When reporting a bug, include the pamica version, platform, device
  (CPU/CUDA/MPS/MLX), precision (float32/float64), and a minimal example.

## Development setup

pamica uses [UV](https://docs.astral.sh/uv/) for environment and dependency
management.

```bash
git clone https://github.com/sccn/pAMICA.git
cd pAMICA
uv sync                 # install the project and dependencies
uv run pytest           # run the test suite
```

On Apple MPS, run with `PYTORCH_ENABLE_MPS_FALLBACK=1` for ops MPS does not yet
support.

## Testing

- **Real data only.** Correctness tests use the real sample EEG and the Fortran
  binary shipped in `pamica/sample_data/`. Do not use mocks, stubs, or synthetic
  data as the basis for a correctness test: no test is better than a fake passing
  test.
- Run with coverage: `uv run pytest --cov`.
- The natural-gradient backend computes in float64 for Fortran parity; use
  float64 for parity-sensitive tests.

## Code style

- **Lint and format** with Ruff before committing:
  ```bash
  uv run ruff check --fix . && uv run ruff format .
  ```
- Follow the conventions in the surrounding code and in `AGENTS.md`.
- No em-dashes in prose; define abbreviations on first use.

## Pull requests

1. Open an issue first (except for minor fixes).
2. Create a branch (for example `gh issue develop <n>`).
3. Make atomic commits with concise messages (no emojis, no AI attribution).
4. Add or update tests, and run the suite before pushing.
5. Open a PR describing what changed and how it was tested.
6. Ensure CI is green before requesting a merge.

## License

By contributing, you agree that your contributions will be licensed under the
project's [BSD 3-Clause License](LICENSE).
