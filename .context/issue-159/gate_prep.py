"""MATLAB gate prep for issue #159 (run first).

Writes a real 2-model pamica output, then dumps what Python ``loadmodout``
reads for W/A/sbeta/rho on (a) the genuine single-model Fortran fixture and
(b) that 2-model output. ``gate_compare.m`` then loads the same two directories
with EEGLAB's ``loadmodout15.m`` and this script's companion assertion (run last)
checks they agree element-wise. Python ``loadmodout`` is a port of
``loadmodout15.m``, so on correct byte order the two must match to ~1e-12; the
pre-#159 C-order read made W (and A/svar) disagree by the internal transpose.
"""

import os
import numpy as np

from pamica import AMICA
from pamica.numpy_impl.load import loadmodout
from pamica.torch_impl.utils import load_eeglab_data

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
SAMPLE = os.path.join(ROOT, "pamica", "sample_data")
FIXTURE = os.path.join(SAMPLE, "amicaout")
TWO_MODEL = os.path.join(HERE, "gate_2model")

X = load_eeglab_data(os.path.join(SAMPLE, "eeglab_data.fdt"), 32, 30504).astype(
    np.float64
)

# Deterministic real 2-model fit -> genuine-Fortran multi-model output.
model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
model.fit(X[:, :8192], max_iter=20, block_size=1024, seed=4)
assert model.is_fitted_, "2-model gate fit ended degenerate; adjust seed"
model.write_amica_output(TWO_MODEL)

for tag, d in (("fixture", FIXTURE), ("two_model", TWO_MODEL)):
    out = loadmodout(d)
    np.savez(
        os.path.join(HERE, f"py_{tag}.npz"),
        W=out.W,
        A=out.A,
        sbeta=out.sbeta,
        rho=out.rho,
        num_models=out.num_models,
    )
    print(f"{tag}: num_models={out.num_models} W{out.W.shape} A{out.A.shape}")

print(f"\nTwo-model output written to: {TWO_MODEL}")
print("Next: run gate_compare.m in MATLAB, then gate_check.py.")
