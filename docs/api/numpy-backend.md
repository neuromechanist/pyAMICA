# NumPy backend (AMICA_NumPy)

The legacy NumPy reference implementation, retained as an oracle and for its
command-line interface. It carries the same parity fixes as the PyTorch backend,
plus baralpha. Outlier rejection (`do_reject`) is not functional on this backend
and `fit()` refuses it; use the PyTorch backend for outlier rejection (see issue
#123).

::: pyAMICA.AMICA_NumPy
