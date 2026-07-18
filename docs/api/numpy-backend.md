# NumPy backend (AMICA_NumPy)

The legacy NumPy reference implementation, retained as an oracle and for its
command-line interface. It carries the same parity fixes as the PyTorch backend,
plus baralpha and outlier rejection (`do_reject`), which mirrors the PyTorch
backend's `good_idx` sample-dropping mechanism (issue #123).

::: pamica.AMICA_NumPy
