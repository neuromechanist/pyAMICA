# pyAMICA Design Ideas

High-level PyTorch design decisions and library options for the AMICA port.

## Vision
Build a GPU-accelerated AMICA (CUDA / Apple MPS / CPU) that matches the Fortran reference, using
PyTorch autograd instead of hand-coded derivatives, and leaning on battle-tested libraries for
standard components (Newton, mixture models, natural gradient) so effort concentrates on
AMICA-specific logic.

## Architecture
- **`nn.Module` framework:** parameters (A, c, alpha, mu, beta, rho) as `nn.Parameter`; free
  serialization and optimizer compatibility.
- **Log-space throughout:** stable `logsumexp` for mixtures; epsilon on denominators.
- **Pluggable optimizers:** natural gradient for warm-up, Newton for final convergence.
- **Configurable device placement:** auto-select MPS/CUDA/CPU.
- **Hybrid EM-gradient (candidate):** EM for mixture weights (stable), gradients for mixing matrices.

## Library options considered
- **Newton:** `pytorch-minimize` (rfeinman) - exact Hessian + Newton-CG, GPU. Built-in
  `torch.optim.LBFGS` as fallback; `hjmshi/PyTorch-LBFGS` for mini-batch/line-search stability.
- **Mixture models:** `gmm-torch` (ldeecke) - GPU, sklearn-like; `GMMPytorch` (kylesayrs) -
  gradient-based with singularity mitigation.
- **Natural gradient / Fisher:** `NNGeometry` (tfjgeorge) - KFAC/EKFAC/diagonal Fisher, GPU;
  `gpytorch.optim.NGD`.
  (These are optional; see the commented entries in `../requirements-torch.txt`.)

## Advantages of the PyTorch port
- GPU/MPS acceleration; autograd removes manual gradient code.
- Built-in stability tools (clipping, mixed precision, stable log-sum-exp).
- Faster iteration and easier extension than the Fortran/NumPy paths.

## Key risks -> mitigations
- **Memory:** gradient checkpointing, batch/block processing, low-rank Fisher.
- **Numerical stability:** log-probabilities, gradient clipping, epsilon denominators, bounded values.
- **Matching Fortran exactly:** identical initialization, log all intermediates for comparison,
  replicate Fortran's specific numerical tricks, validate against the binary.

## Open questions
1. Why `do_approx_sphere=.true.` in Fortran while Python uses exact sphering?
2. Relationship between `sbeta` and `beta`.
3. Exact mixture-model update in Fortran.
4. What is `baralpha` and why 3D?
