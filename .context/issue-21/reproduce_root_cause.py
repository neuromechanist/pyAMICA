"""Reproduce issue #21's root cause: the NG first-order M-step uses the density
derivative `dpdf = p'(y)` where Fortran uses the score `fp = rho*sign(y)*|y|^(rho-1)`.
Because `p'(y) = -fp(y)*p(y)`, this flips the update sign, so the log-likelihood
DESCENDS from iteration 1 (the divergence issue #21 mislabels as a "0.68 plateau").

Run from the repo root:  uv run python .context/issue-21/reproduce_root_cause.py

Expected output:
  BEFORE (dpdf): LL goes DOWN every iteration (-3.51, -3.54, -3.58, ...), Newton
                 never fires (posdef guard fails every iteration).
  AFTER  (score fp): LL goes UP for the first iterations, tracking Fortran
                 (-3.512 -> -3.501 -> -3.484), and Newton becomes positive-definite.

This is a DIAGNOSTIC, not the fix. The full fix (score fp + exact-EM mu/beta +
source-space natural gradient + symmetric ZCA sphere + rho psi-factor) is tracked in
.context/issue-21/root_cause.md.
"""

from pathlib import Path

import numpy as np
import torch

import pyAMICA.torch_impl.amica_torch_ng as ng_mod
from pyAMICA.torch_impl import AMICATorchNG
from pyAMICA.torch_impl.utils import load_eeglab_data

SAMPLE = Path(__file__).resolve().parents[2] / "pyAMICA" / "sample_data"
NW, FIELD, SEED = 32, 30504, 42

_orig = ng_mod._log_pdf_and_deriv


def _patched(y, rho):
    """Return the score fp in the slot the M-step reads as `dpdf`."""
    log_pdf, _ = _orig(y, rho)
    return log_pdf, ng_mod._score(y, rho)


def _traj(ll, idxs):
    return "  ".join(f"i{i}={ll[i]:+.4f}" if i < len(ll) else f"i{i}=--" for i in idxs)


def _run(patch, **kw):
    ng_mod._log_pdf_and_deriv = _patched if patch else _orig
    try:
        m = AMICATorchNG(
            n_channels=NW,
            n_models=1,
            n_mix=3,
            seed=SEED,
            device="cpu",
            dtype=torch.float64,
            block_size=512,
            **kw,
        )
        m.fit(data, max_iter=60, verbose=False)
    finally:
        ng_mod._log_pdf_and_deriv = _orig
    return m


if __name__ == "__main__":
    data = load_eeglab_data(
        str(SAMPLE / "eeglab_data.fdt"), data_dim=NW, field_dim=FIELD
    ).astype(np.float64)
    idxs = [0, 1, 2, 5, 10, 20, 40, 59]
    print(
        "Fortran reference (natural-gradient phase): i1=-3.513 i10=-3.450 i40=-3.442\n"
    )

    m = _run(patch=False, do_newton=False)
    print(f"BEFORE (dpdf, current code): {_traj(m.ll_history, idxs)}")
    print("  -> log-likelihood DESCENDS: the M-step update direction is inverted.\n")

    m = _run(patch=True, do_newton=True, newt_start=20, newtrate=1.0, lrate=0.05)
    print(f"AFTER  (score fp): {_traj(m.ll_history, idxs)}")
    print(
        f"  -> ascends early (tracks Fortran); Newton fallbacks = {m.n_newton_fallbacks} "
        "(was 68/68 before the fix)."
    )
