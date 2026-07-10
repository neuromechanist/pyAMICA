# Testing

## Real data only

Correctness tests use real sample EEG and the reference Fortran binary shipped in
`pyAMICA/sample_data/`. Mocks, stubs, and synthetic data are not used as the
basis for correctness tests: no test is better than a fake passing test.

## Running the suite

```bash
uv run pytest                    # full suite
uv run pytest --cov              # with coverage
uv run pytest pyAMICA/tests/torch_tests/   # PyTorch-vs-Fortran parity tests
```

## Layout

- `pyAMICA/tests/` — end-to-end and interface tests.
- `pyAMICA/tests/torch_tests/` — natural-gradient backend parity, PDF families,
  component sharing, float32 stability, and edge cases.
- `pyAMICA/tests/mlx_tests/` — MLX backend tests (Apple Silicon).
- `validate_implementations.py` — cross-implementation validation harness
  (Hungarian component matching against Fortran).
