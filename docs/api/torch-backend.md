# PyTorch backend (AMICATorchNG)

The natural-gradient EM backend that reaches Fortran parity (Newton, exact-EM
mixture updates, symmetric-ZCA sphere, Jacobian log-likelihood). The
[`AMICA`](amica.md) interface delegates to this class; use it directly for
lower-level control.

::: pyAMICA.AMICATorchNG
