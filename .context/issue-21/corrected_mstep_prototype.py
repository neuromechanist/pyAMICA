"""VALIDATED reference prototype for the issue #21 fix (do not ship as-is; port
these methods into amica_torch_ng.py and pamica.py -- see handoff.md).

`CorrectedNG` subclasses the shipped `AMICATorchNG` and overrides the M-step with the
Fortran-faithful version that this session validated:

  * score `fp` (not the density derivative `dpdf`) in g/dA, dmu, dbeta
  * exact-EM updates:  mu += dmu_numer/dmu_denom ; sbeta *= sqrt(dbeta_numer/dbeta_denom)
  * source-space natural gradient:  dWtmp = g^T b ;  A -= lrate*A*(I - <g b^T>/dgm)
  * symmetric ZCA sphere  V diag(1/sqrt(eval)) V^T  (Fortran do_approx_sphere=True)
  * rho update with the 1/psi(1+1/rho) digamma factor
  * transpose fix: the TRUE unmixing is self.W.T (get_unmixing_matrix / transform / compare)

Run `uv run python .context/issue-21/corrected_mstep_prototype.py` to reproduce the two
decisive validations (fixed-point test + short fit). Fortran line refs are in the comments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pamica.torch_impl import AMICATorchNG
from pamica.torch_impl.amica_torch_ng import _score
from pamica.torch_impl.utils import load_eeglab_data

NW, FIELD, SEED = 32, 30504, 42
SAMPLE = Path(__file__).resolve().parents[2] / "pamica" / "sample_data"
AO = SAMPLE / "amicaout"


class CorrectedNG(AMICATorchNG):
    do_rho = True

    # --- symmetric ZCA sphere (Fortran amica17.f90:480-481, do_approx_sphere=True path) ---
    def _preprocess(self, X):
        Xc = torch.from_numpy(np.ascontiguousarray(X)).to(torch.float64)
        if self.do_mean:
            mean = Xc.mean(1, keepdim=True)
        else:
            mean = torch.zeros(Xc.shape[0], 1, dtype=torch.float64)
        Xc = Xc - mean
        if self.do_sphere:
            ev, V = torch.linalg.eigh(torch.cov(Xc))
            sphere = V @ torch.diag(1.0 / torch.sqrt(ev)) @ V.T  # V D^-1/2 V^T
            self.sldet = float(-0.5 * torch.log(ev).sum().item())
        else:
            sphere = torch.eye(Xc.shape[0], dtype=torch.float64)
            self.sldet = 0.0
        Xc = sphere @ Xc
        self.mean = mean.to(self.device, self.dtype)
        self.sphere = sphere.to(self.device, self.dtype)
        return Xc.to(self.device, self.dtype)

    # --- Fortran-faithful sufficient statistics (amica17.f90:1437-1592) ---
    def _get_block_updates(self, X):
        nmix, nm = self.n_mix, self.n_models
        logV, b_list, z_list, y_list, _ = self._forward(X)
        v = torch.softmax(logV, dim=1)  # (batch, nm)
        block_ll = torch.logsumexp(logV, dim=1).sum()
        dev, dt = self.device, self.dtype

        def z(*s):
            return torch.zeros(*s, dtype=dt, device=dev)

        dgm = z(nm)
        dalpha_n = z(nmix, self.n_comps)  # dalpha_numer = sum(u)
        dmu_n = z(nmix, self.n_comps)
        dmu_d = z(nmix, self.n_comps)
        dbeta_n = z(nmix, self.n_comps)
        dbeta_d = z(nmix, self.n_comps)
        drho_n = z(nmix, self.n_comps)
        dWtmp = z(NW, NW, nm)
        dc = z(NW, nm)
        newton = self.do_newton
        if newton:
            dsig = z(NW, nm)
            dkap = z(nmix, NW, nm)
            dlam = z(nmix, NW, nm)
        tiny = torch.finfo(dt).tiny

        for h in range(nm):
            idx = self.comp_list[:, h]
            b, zr, y = b_list[h], z_list[h], y_list[h]  # zr = mixture responsibility
            v_h = v[:, h]
            beta_h = self.beta[:, idx].T.unsqueeze(0)  # sbeta, (1,nw,nmix)
            rho_h = self.rho[:, idx].T  # (nw,nmix)
            fp = _score(y, rho_h.unsqueeze(0))  # score, NOT dpdf (:1467)
            u = v_h.unsqueeze(-1).unsqueeze(-1) * zr  # u = v*z (:1439)
            ufp = u * fp  # (:1485)

            dgm[h] = v_h.sum()
            dalpha_n.index_add_(1, idx, u.sum(0).T)
            dmu_n.index_add_(1, idx, ufp.sum(0).T)  # sum(ufp) (:1532)
            dmu_d.index_add_(
                1, idx, (beta_h.squeeze(0) * (ufp / y).sum(0)).T
            )  # sbeta*sum(ufp/y) (:1537)
            dbeta_n.index_add_(1, idx, u.sum(0).T)  # sum(u) (:1550)
            dbeta_d.index_add_(1, idx, (ufp * y).sum(0).T)  # sum(ufp*y) (:1556)
            # drho_numer = sum(u * |y|^rho * log|y|)  (non-MKL branch, :1578)
            ay = y.abs()
            mask = (rho_h != 1.0) & (rho_h != 2.0)
            term = torch.where(
                mask.unsqueeze(0),
                ay.pow(rho_h.unsqueeze(0)) * torch.log(ay.clamp_min(tiny)),
                torch.zeros_like(ay),
            )
            drho_n.index_add_(1, idx, (u * term).sum(0).T)

            g = (beta_h * ufp).sum(-1)  # g_i = sum_j sbeta*ufp (:1493)
            dWtmp[:, :, h] = g.T @ b  # source-space sum g_t b_t^T (:1592)
            dc[:, h] = g.sum(0)
            if newton:
                dsig[:, h] = (v_h.unsqueeze(-1) * b.pow(2)).sum(0)  # (:1419)
                dkap[:, :, h] = (
                    (u * fp.pow(2)).sum(0) * beta_h.squeeze(0).pow(2)
                ).T  # (:1500)
                dlam[:, :, h] = (u * (fp * y - 1.0).pow(2)).sum(0).T  # (:1511)

        out = {
            "dgm": dgm,
            "dalpha_n": dalpha_n,
            "dmu_n": dmu_n,
            "dmu_d": dmu_d,
            "dbeta_n": dbeta_n,
            "dbeta_d": dbeta_d,
            "drho_n": drho_n,
            "dWtmp": dWtmp,
            "dc": dc,
            "ll": block_ll,
        }
        if newton:
            out.update(dsigma2_numer=dsig, dkappa_numer=dkap, dlambda_numer=dlam)
        return out

    def _finalize_newton_stats(self, acc):
        dgm = acc["dgm"].unsqueeze(0)
        sigma2 = acc["dsigma2_numer"] / dgm
        kappa = acc["dkappa_numer"].sum(0) / dgm
        mu_at = self.mu[:, self.comp_list]
        lam = (acc["dlambda_numer"] + acc["dkappa_numer"] * mu_at.pow(2)).sum(0) / dgm
        return sigma2, lam, kappa

    # --- Fortran-faithful M-step (amica17.f90:1890-1993) ---
    def _update_parameters(self, acc, n_samples):
        self.gm = acc["dgm"] / n_samples
        self.alpha = acc["dalpha_n"] / acc["dalpha_n"].sum(0, keepdim=True)  # (:1891)
        self.mu = self.mu + acc["dmu_n"] / acc["dmu_d"]  # (:1978)
        self.beta = torch.clamp(
            self.beta * torch.sqrt(acc["dbeta_n"] / acc["dbeta_d"]),
            self.invsigmin,
            self.invsigmax,
        )  # (:1993)
        rho = self.rho
        if self.do_rho and not torch.all(rho == 1.0) and not torch.all(rho == 2.0):
            drho = acc["drho_n"] / acc["dalpha_n"].clamp_min(1e-8)
            psi = torch.special.digamma(1.0 + 1.0 / rho)  # (:2013-2014)
            nr = rho + self.rholrate * (1.0 - (rho / psi) * drho)
            self.rho = torch.clamp(
                torch.nan_to_num(nr, nan=1.5), self.minrho, self.maxrho
            )

        newton_active = self.do_newton and self.iteration >= self.newt_start
        if newton_active:
            sigma2, lam, kappa = self._finalize_newton_stats(acc)
        eye = torch.eye(NW, dtype=self.dtype, device=self.device)
        dirs, no_newt = [], False
        for h in range(self.n_models):
            dA_h = (
                -acc["dWtmp"][:, :, h] / acc["dgm"][h] + eye
            )  # I - <g b^T>/dgm (:1800-1807)
            if newton_active:
                H, posdef = self._newton_direction(
                    dA_h, sigma2[:, h], lam[:, h], kappa[:, h]
                )
                dirs.append(H if posdef else dA_h)
                no_newt |= not posdef
            else:
                dirs.append(dA_h)
        if newton_active and no_newt:
            self.n_newton_fallbacks += 1
        if newton_active and not no_newt:
            self.lrate = min(
                self.newtrate, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
            )
        else:
            self.lrate = min(
                self.lrate_cap, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
            )
        for h in range(self.n_models):
            idx = self.comp_list[:, h]
            A_cols = self.A[:, idx]
            self.A[:, idx] = A_cols - self.lrate * (
                A_cols @ dirs[h]
            )  # A += -lrate*A*dir (:1909/1916)
        if self.doscaling and (self.iteration % self.scalestep == 0):
            s = torch.sqrt((self.A**2).sum(0))
            nz = s > 0
            self.A[:, nz] = self.A[:, nz] / s[nz]
            self.mu[:, nz] = self.mu[:, nz] * s[nz]
            self.beta[:, nz] = self.beta[:, nz] / s[nz]
        self._update_unmixing_matrices()

    # --- transpose fix: the TRUE unmixing is self.W.T ---
    def get_unmixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        return self.W[:, :, model_idx].T.cpu().numpy()


# ---------------------------------------------------------------------------
# Validations (the guard the in-code implementation must reproduce).
# ---------------------------------------------------------------------------
def _load(name, shape):
    return np.fromfile(AO / name, dtype=np.float64).reshape(shape, order="F")


def _corr(A, B):
    from scipy.optimize import linear_sum_assignment

    a = A / np.linalg.norm(A, axis=1, keepdims=True)
    b = B / np.linalg.norm(B, axis=1, keepdims=True)
    C = np.abs(a @ b.T)
    r, c = linear_sum_assignment(1 - C)
    return C[r, c].mean()


def fixed_point_test():
    """Seed Fortran's converged solution; assert it is a stable fixed point and
    that the transpose-fixed unmixing matches Fortran to >0.99."""
    raw = load_eeglab_data(
        str(SAMPLE / "eeglab_data.fdt"), data_dim=NW, field_dim=FIELD
    ).astype(np.float64)
    W = _load("W", (NW, NW))
    S = _load("S", (NW, NW))
    mean = np.fromfile(AO / "mean", dtype=np.float64)
    mu = _load("mu", (3, NW))
    sbeta = _load("sbeta", (3, NW))
    rho = _load("rho", (3, NW))
    alpha = _load("alpha", (3, NW))
    gm = np.fromfile(AO / "gm", dtype=np.float64)

    m = CorrectedNG(
        n_channels=NW,
        n_models=1,
        n_mix=3,
        seed=0,
        device="cpu",
        dtype=torch.float64,
        block_size=512,
        do_newton=False,
        lrate=0.05,
    )
    Xs = torch.tensor(S @ (raw - mean[:, None]))
    m.sphere = torch.tensor(S)
    m.mean = torch.tensor(mean[:, None])
    ev = np.linalg.eigvalsh(np.cov(raw - mean[:, None]))
    m.sldet = float(-0.5 * np.log(ev).sum())
    m.comp_list = torch.arange(NW).unsqueeze(1)
    m.A = torch.tensor(
        np.linalg.inv(W.T)
    )  # NOTE the transpose: Fortran W = inv(A_internal).T
    m.mu = torch.tensor(mu)
    m.beta = torch.tensor(sbeta)
    m.rho = torch.tensor(rho)
    m.alpha = torch.tensor(alpha)
    m.gm = torch.tensor(gm)
    m.c = torch.zeros(NW, 1, dtype=torch.float64)
    m.iteration = 100
    m.lrate = 0.05
    m.lrate_cap = 0.05
    m._update_unmixing_matrices()

    n = Xs.shape[1]
    ll0 = float(m._accumulate_blocks(Xs)["ll"] / (n * NW))
    for _ in range(10):
        m._update_parameters(m._accumulate_blocks(Xs), n)
    ll1 = float(m._accumulate_blocks(Xs)["ll"] / (n * NW))
    Wt = m.get_unmixing_matrix(0)
    print(
        f"[fixed-point] seeded LL={ll0:.5f} (expect -3.40186); after 10 steps={ll1:.5f} (expect ~ -3.402)"
    )
    print(f"[fixed-point] corr(W.T, Fortran W)={_corr(Wt, W):.4f} (expect ~0.9997)")


def short_fit():
    raw = load_eeglab_data(
        str(SAMPLE / "eeglab_data.fdt"), data_dim=NW, field_dim=FIELD
    ).astype(np.float64)
    m = CorrectedNG(
        n_channels=NW,
        n_models=1,
        n_mix=3,
        seed=SEED,
        device="cpu",
        dtype=torch.float64,
        block_size=512,
        do_newton=False,
        lrate=0.05,
        lratefact=1.0,
        maxdecs=10**9,
    )
    m.fit(raw, max_iter=30, verbose=False)
    print(
        f"[short fit] LL i1={m.ll_history[1]:.4f} i10={m.ll_history[10]:.4f} "
        f"i29={m.ll_history[-1]:.4f} (ascends from ~ -3.51; do NOT diverge)"
    )


if __name__ == "__main__":
    fixed_point_test()
    short_fit()
