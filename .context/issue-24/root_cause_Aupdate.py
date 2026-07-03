"""Issue #24 ROOT CAUSE: the natural-gradient A-update is transposed / multiplied
on the wrong side. This closes #24 (and supersedes the earlier "Bug 3 mu/beta
denominator" theory, which was a sphere-comparison artifact).

Four decisive checks, all against the Fortran reference binary on real sample data:

  1. The Python vs Fortran sphere differ by a PURE SCALAR sqrt(N/(N-1)) -- Python
     spheres with torch.cov (cov/(N-1)); Fortran uses cov/N. Benign in a real run.
  2. With Fortran's sphere fed to Python, the mu/beta exact-EM update is BIT-EXACT
     (~1e-13). So the mu/beta "drift" localized earlier was the sphere artifact,
     amplified by the singular sum(u*rho*|y|^(rho-2)) denominator -- NOT a formula bug.
  3. The A-update is the real bug. At k=0 (matched sphere), of the four candidate
     forms only  A <- A - lr*(I - G^T/dgm) @ A  (G = g^T b, LEFT-multiply, TRANSPOSED)
     matches Fortran to machine precision (~1e-15). The shipped prototype does
     A <- A - lr*A @ (I - G/dgm) (right-multiply, untransposed): wrong on BOTH counts.
     Invisible at the fixed point (G ~ G^T there) so fixed_point_test never caught it.
  4. With the A-update fixed, the real pinned-lrate fit ASCENDS to Fortran's
     pinned-NG endpoint (-3.4265 vs Fortran -3.4269; corr 0.648 vs 0.645) instead of
     descending to -3.4974 / 0.506.

Run:  PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python .context/issue-24/root_cause_Aupdate.py [--fit]
(--fit adds check 4, two 200-iter fits; slower.)
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "pyAMICA" / "sample_data"
AO = SAMPLE / "amicaout"
BIN = SAMPLE / "amica15mac"
PROTO = REPO / ".context" / "issue-21" / "corrected_mstep_prototype.py"
NW, NMIX, FIELD, SEED = 32, 3, 30504, 42
RUN = Path(__file__).resolve().parent / "_rootcause"
RUN.mkdir(parents=True, exist_ok=True)

spec = importlib.util.spec_from_file_location("corrected_mstep_prototype", PROTO)
assert spec is not None and spec.loader is not None
pr = importlib.util.module_from_spec(spec)
sys.path.insert(0, str(REPO))
spec.loader.exec_module(pr)
CorrectedNG, load_eeglab_data = pr.CorrectedNG, pr.load_eeglab_data

HY = dict(
    n_channels=NW,
    n_models=1,
    n_mix=NMIX,
    seed=SEED,
    device="cpu",
    dtype=torch.float64,
    block_size=512,
    lrate=0.05,
    rholrate=0.05,
    minrho=1.0,
    maxrho=2.0,
    rho0=1.5,
    do_newton=False,
    do_mean=True,
    do_sphere=True,
    invsigmin=0.0,
    invsigmax=100.0,
    scalestep=1,
    doscaling=False,
)


class FullRhoNG(CorrectedNG):
    """rho fix (Bugs 1+2): drho numerator = rho * sum(u |y|^rho ln|y|), no per-comp mask."""

    def _get_block_updates(self, X):
        out = super()._get_block_updates(X)
        logV, _, z_list, y_list, _ = self._forward(X)
        v = torch.softmax(logV, dim=1)
        drho_n = torch.zeros(
            self.n_mix, self.n_comps, dtype=self.dtype, device=self.device
        )
        tiny = torch.finfo(self.dtype).tiny
        for h in range(self.n_models):
            idx = self.comp_list[:, h]
            y = y_list[h]
            rho_h = self.rho[:, idx].T
            u = v[:, h].unsqueeze(-1).unsqueeze(-1) * z_list[h]
            ay = y.abs()
            ayrho = ay.pow(rho_h.unsqueeze(0))
            logab = rho_h.unsqueeze(0) * torch.log(ay.clamp_min(tiny))
            logab = torch.where(ayrho < tiny, torch.zeros_like(logab), logab)
            drho_n.index_add_(1, idx, (u * (ayrho * logab)).sum(0).T)
        out["drho_n"] = drho_n
        return out


def _write_load(indir, m):
    indir.mkdir(exist_ok=True)

    def w(name, arr):
        np.asarray(arr, "<f8").flatten(order="F").tofile(indir / name)

    w("A", m.A.cpu().numpy().T)
    w("mean", m.mean.cpu().numpy().reshape(-1))
    w("mu", m.mu.cpu().numpy())
    w("sbeta", m.beta.cpu().numpy())
    w("rho", m.rho.cpu().numpy())
    w("alpha", m.alpha.cpu().numpy())
    w("gm", m.gm.cpu().numpy())
    w("c", m.c.cpu().numpy())


_PARAM = """files ./eeglab_data.fdt
outdir ./out/
indir {indir}
num_models 1
num_mix_comps 3
pdftype 0
max_iter 1
num_samples 1
data_dim 32
field_dim 30504
lrate 0.05
rholrate 0.05
rho0 1.5
minrho 1.0
maxrho 2.0
do_newton 0
do_mean 1
do_sphere 1
doPCA 1
pcakeep 32
byte_size 4
max_threads 8
do_rho 1
doscaling 0
scalestep 1
block_size 512
load_mean 1
load_sphere 0
load_A 1
load_mu 1
load_beta 1
load_rho 1
load_alpha 1
load_gm 1
load_c 1
"""


def fortran_1step(m):
    """Seed Fortran from m's state, run one NG step, return its post-step state."""
    indir = RUN / "init"
    _write_load(indir, m)
    wd = RUN / "F1"
    wd.mkdir(exist_ok=True)
    if not (wd / "eeglab_data.fdt").exists():
        (wd / "eeglab_data.fdt").symlink_to(SAMPLE / "eeglab_data.fdt")
    if (wd / "out").exists():
        shutil.rmtree(wd / "out")
    (wd / "run.param").write_text(_PARAM.format(indir=str(indir)))
    subprocess.run([str(BIN), "run.param"], cwd=wd, capture_output=True, timeout=300)
    o = wd / "out"

    def rd(name, shp):
        return np.fromfile(o / name, np.float64).reshape(shp, order="F")

    return dict(
        W=rd("W", (NW, NW)),
        mu=rd("mu", (NMIX, NW)),
        sbeta=rd("sbeta", (NMIX, NW)),
        S=rd("S", (NW, NW)),
    )


raw = load_eeglab_data(
    str(SAMPLE / "eeglab_data.fdt"), data_dim=NW, field_dim=FIELD
).astype(np.float64)
base = FullRhoNG(**HY)
base._preprocess(raw)
base._initialize_parameters()
S_py = base.sphere.cpu().numpy()
snap = {
    k: getattr(base, k).clone()
    for k in ("A", "mu", "beta", "rho", "alpha", "gm", "c", "mean")
}
F = fortran_1step(base)
S_F = F["S"]

# --- check 1: sphere is a pure scalar sqrt(N/(N-1)) off ----------------------
ratio = S_F / S_py
fin = np.isfinite(ratio) & (np.abs(S_py) > 1e-6)
print("=" * 78)
print("[1] SPHERE  Python cov/(N-1) vs Fortran cov/N")
print(f"    max|S_F - S_py|      = {np.abs(S_F - S_py).max():.3e}")
print(f"    S_F/S_py             = {ratio[fin].mean():.8f} +/- {ratio[fin].std():.1e}")
print(
    f"    sqrt(N/(N-1))        = {np.sqrt(FIELD / (FIELD - 1)):.8f}"
    "   (pure scalar => benign global rescale)"
)


def one_step(sphere_np):
    """Fresh FullRhoNG seeded from the init snapshot with the given sphere; one M-step."""
    m = FullRhoNG(**HY)
    for k in ("A", "mu", "beta", "rho", "alpha", "gm", "c"):
        setattr(m, k, snap[k].clone())
    m.mean = snap["mean"].clone()
    m.sphere = torch.tensor(sphere_np)
    m.sldet = base.sldet
    m.comp_list = torch.arange(NW).unsqueeze(1)
    m.iteration = 0
    m.lrate = m.lrate_cap = 0.05
    m.n_newton_fallbacks = 0
    m._update_unmixing_matrices()
    Xs = m.sphere @ (torch.tensor(raw) - m.mean)
    acc = m._accumulate_blocks(Xs)
    m._update_parameters(acc, Xs.shape[1])
    return m.mu.cpu().numpy(), m.beta.cpu().numpy(), acc


# --- check 2: mu/beta bit-exact once the sphere matches ----------------------
mu_py, be_py, _ = one_step(S_py)
mu_F, be_F, accF = one_step(S_F)
print("[2] mu/beta ONE-STEP residual vs Fortran")
print(
    f"    Python sphere : max|mu-muF|={np.abs(mu_py - F['mu']).max():.2e}"
    f"  beta={np.abs(be_py - F['sbeta']).max():.2e}"
)
print(
    f"    Fortran sphere: max|mu-muF|={np.abs(mu_F - F['mu']).max():.2e}"
    f"  beta={np.abs(be_F - F['sbeta']).max():.2e}  <= bit-exact"
)

# --- check 3: the four A-update forms at k=0 (Fortran sphere) -----------------
G = accF["dWtmp"][:, :, 0].cpu().numpy()
dgm = float(accF["dgm"][0])
A_init = snap["A"].cpu().numpy()
A_F1 = np.linalg.inv(F["W"].T)  # Fortran's A_new = A_fort_new^T = inv(W_F.T)
lr = 0.05
eye = np.eye(NW)
forms = {
    "1 current  A - lr*A@(I - G/n)    ": A_init - lr * (A_init @ (eye - G / dgm)),
    "2 right,T  A - lr*A@(I - G^T/n)  ": A_init - lr * (A_init @ (eye - G.T / dgm)),
    "3 left,T   A - lr*(I - G^T/n)@A  ": A_init - lr * ((eye - G.T / dgm) @ A_init),
    "4 left     A - lr*(I - G/n)@A    ": A_init - lr * ((eye - G / dgm) @ A_init),
}
print("[3] A-UPDATE forms vs Fortran A_new (G = g^T b):")
for k, val in forms.items():
    tag = "  <== MACHINE-EXACT (correct)" if np.abs(val - A_F1).max() < 1e-12 else ""
    print(f"    {k} max|.-A_F1| = {np.abs(val - A_F1).max():.2e}{tag}")
print("=" * 78)


# --- check 4 (optional): pinned-lrate fit parity -----------------------------
if "--fit" in sys.argv:
    from scipy.optimize import linear_sum_assignment

    class AFixNG(FullRhoNG):
        """rho fix + corrected NG A-update: A <- A - lr*(I - G^T/dgm) @ A."""

        def _update_parameters(self, acc, n):
            self.gm = acc["dgm"] / n
            self.alpha = acc["dalpha_n"] / acc["dalpha_n"].sum(0, keepdim=True)
            self.mu = self.mu + acc["dmu_n"] / acc["dmu_d"]
            self.beta = torch.clamp(
                self.beta * torch.sqrt(acc["dbeta_n"] / acc["dbeta_d"]),
                self.invsigmin,
                self.invsigmax,
            )
            rho = self.rho
            if self.do_rho and not torch.all(rho == 1.0) and not torch.all(rho == 2.0):
                drho = acc["drho_n"] / acc["dalpha_n"].clamp_min(1e-8)
                psi = torch.special.digamma(1.0 + 1.0 / rho)
                self.rho = torch.clamp(
                    torch.nan_to_num(
                        rho + self.rholrate * (1.0 - (rho / psi) * drho), nan=1.5
                    ),
                    self.minrho,
                    self.maxrho,
                )
            ident = torch.eye(NW, dtype=self.dtype, device=self.device)
            self.lrate = min(
                self.lrate_cap, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
            )
            for h in range(self.n_models):
                idx = self.comp_list[:, h]
                A_cols = self.A[:, idx]
                Gh = acc["dWtmp"][:, :, h]
                self.A[:, idx] = A_cols - self.lrate * (
                    (ident - Gh.T / acc["dgm"][h]) @ A_cols
                )
            if self.doscaling and (self.iteration % self.scalestep == 0):
                s = torch.sqrt((self.A**2).sum(0))
                nz = s > 0
                self.A[:, nz] = self.A[:, nz] / s[nz]
                self.mu[:, nz] = self.mu[:, nz] * s[nz]
                self.beta[:, nz] = self.beta[:, nz] / s[nz]
            self._update_unmixing_matrices()

    W_ref = np.fromfile(AO / "W", np.float64).reshape(NW, NW, order="F")
    S_ref = np.fromfile(AO / "S", np.float64).reshape(NW, NW, order="F")

    def corr(W, S):
        Fa, Fb = W @ S, W_ref @ S_ref
        a = Fa / np.linalg.norm(Fa, axis=1, keepdims=True)
        b = Fb / np.linalg.norm(Fb, axis=1, keepdims=True)
        C = np.abs(a @ b.T)
        r, c = linear_sum_assignment(1 - C)
        return float(C[r, c].mean())

    FIT = dict(
        HY,
        doscaling=True,
        lratefact=1.0,
        maxdecs=10**9,
        minlrate=1e-8,
        newt_ramp=10,
        newt_start=50,
        newtrate=1.0,
        rholratefact=0.5,
    )
    print("[4] pinned-lrate fit (Fortran pinned NG: end=-3.4269 corr=0.645)")
    for cls, name in [(FullRhoNG, "current A-update"), (AFixNG, "A-UPDATE FIX")]:
        m = cls(**FIT)
        m.fit(raw, max_iter=200, verbose=False)
        cg = corr(m.W[:, :, 0].T.cpu().numpy(), m.sphere.cpu().numpy())
        print(f"    {name:>16}:  end={m.ll_history[-1]:.4f}  corr={cg:.3f}")
