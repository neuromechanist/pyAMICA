# Contributing

Contributions are welcome. This page summarizes the development workflow; see the
full [CONTRIBUTING](https://github.com/neuromechanist/pyAMICA/blob/main/CONTRIBUTING.md)
and [Code of Conduct](https://github.com/neuromechanist/pyAMICA/blob/main/CODE_OF_CONDUCT.md)
in the repository root.

## Development setup

```bash
git clone https://github.com/neuromechanist/pyAMICA.git
cd pyAMICA
uv sync                 # install the project and dependencies
uv run pytest           # run the test suite
```

## Conventions

- **Environment:** UV only (no pip/conda/virtualenv for project management).
- **Lint/format:** `uv run ruff check --fix . && uv run ruff format .`
- **Tests:** real sample data and the Fortran binary only, never synthetic data
  (see [Testing](testing.md)).
- **Correctness:** numerical parity with the Fortran reference is the
  specification (see [Validation & Parity](../guides/validation.md)).
- **Commits:** atomic, concise messages, no emojis.

## Reporting issues and getting help

Please open an issue on the
[GitHub issue tracker](https://github.com/neuromechanist/pyAMICA/issues) for bug
reports, feature requests, and questions.
