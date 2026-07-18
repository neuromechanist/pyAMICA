"""MATLAB gate check for issue #159 (run last).

Compares Python ``loadmodout`` against MATLAB ``loadmodout15.m`` element-wise for
W/A/sbeta/rho, on the genuine single-model fixture and the pamica 2-model output.
Both are the same algorithm (Python is a port), so a correct byte order makes them
agree to floating-point noise; the pre-#159 C-order read made W disagree by the
internal transpose (and A/svar with it), and sbeta/rho disagree for num_mix > 1.
"""

import os
import numpy as np
from scipy.io import loadmat

HERE = os.path.dirname(os.path.abspath(__file__))
TOL = 1e-9

ok = True
for tag in ("fixture", "two_model"):
    py = np.load(os.path.join(HERE, f"py_{tag}.npz"))
    mat = loadmat(os.path.join(HERE, f"mat_{tag}.mat"))
    print(f"\n== {tag} (num_models={int(py['num_models'])}) ==")
    for name in ("W", "A", "sbeta", "rho"):
        p = np.asarray(py[name])
        m = np.asarray(mat[name])
        # MATLAB drops trailing singleton dims (num_models==1 -> 2-D); align.
        if m.ndim < p.ndim:
            m = m.reshape(p.shape)
        close = p.shape == m.shape and np.allclose(p, m, atol=TOL, rtol=1e-7)
        diff = np.max(np.abs(p - m)) if p.shape == m.shape else float("nan")
        ok &= close
        print(
            f"  {name:6s} py{p.shape} mat{m.shape}  max|diff|={diff:.2e}  "
            f"{'OK' if close else 'MISMATCH'}"
        )

print("\nGATE", "PASS" if ok else "FAIL")
