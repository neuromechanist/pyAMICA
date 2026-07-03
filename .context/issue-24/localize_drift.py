"""Issue #24 follow-up: localize WHICH M-step term drifts from Fortran.

Teacher-forced per-iteration diff. Fortran's true NG trajectory (do_newton=0,
lrate=0.05) is obtained by running the binary with max_iter=1..K from the Python
init (do_history is broken -- it fails to mkdir out/history and aborts -- so we
re-run incrementally). For each k we seed CorrectedNG from Fortran's state at iter
k, run ONE M-step, and compare each parameter to Fortran's state at iter k+1.
Because every comparison restarts from a known-good Fortran state, errors do not
compound; the per-param one-step residual localizes the drift.

Pass --fixrho to apply Bug 1 (drho_numer = rho*sum(u|y|^rho ln|y|)); see
drift_localization.md. Result: W/A/alpha/c are faithful, rho is Bug 1+2, and the
dominant residual is the mu/beta exact-EM denominator (Bug 3, open).
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "pyAMICA" / "sample_data"
BIN = SAMPLE / "amica15mac"
PROTO = REPO / ".context" / "issue-21" / "corrected_mstep_prototype.py"
NW, NMIX, FIELD, SEED = 32, 3, 30504, 42
K = 12  # number of Fortran trajectory steps to probe
FIX_RHO = "--fixrho" in sys.argv


def _import_proto():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("corrected_mstep_prototype", PROTO)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prototype at {PROTO}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CorrectedNG, mod.load_eeglab_data


_CorrectedNG, load_eeglab_data = _import_proto()


class CorrectedNG(_CorrectedNG):
    def _update_parameters(self, acc, n_samples):
        if FIX_RHO:
            acc = dict(acc)
            acc["drho_n"] = (
                acc["drho_n"] * self.rho
            )  # Bug 1: amica17.f90:1568 leading rho
        return super()._update_parameters(acc, n_samples)


HYPER = dict(
    n_channels=NW,
    n_models=1,
    n_mix=NMIX,
    seed=SEED,
    device="cpu",
    dtype=torch.float64,
    block_size=512,
    lrate=0.05,
    minlrate=1e-8,
    lratefact=0.5,
    maxdecs=3,
    newt_ramp=10,
    newt_start=50,
    newtrate=1.0,
    rho0=1.5,
    minrho=1.0,
    maxrho=2.0,
    rholrate=0.05,
    rholratefact=0.5,
    invsigmin=0.0,
    invsigmax=100.0,
    doscaling=True,
    scalestep=1,
    do_mean=True,
    do_sphere=True,
    do_newton=False,
)

PARAM = """\
files ./eeglab_data.fdt
outdir ./out/
indir {indir}
block_size 512
num_models 1
num_mix_comps 3
pdftype 0
max_iter {k}
num_samples 1
data_dim 32
field_dim 30504
do_history 0
lrate 0.050000
lratefact 0.500000
rholrate 0.050000
rho0 1.500000
minrho 1.000000
maxrho 2.000000
rholratefact 0.500000
do_newton 0
newt_start 50
do_reject 0
writestep 1
write_nd 0
write_LLt 0
decwindow 1
max_decs 1000000
fix_init 0
do_mean 1
do_sphere 1
doPCA 1
pcakeep 32
byte_size 4
max_threads 8
invsigmax 100.0
invsigmin 0.0
do_rho 1
doscaling 1
scalestep 1
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


def snapshot_init(raw):
    m = CorrectedNG(**HYPER)
    m._preprocess(raw)
    m._initialize_parameters()
    return m


def write_load_files(indir, m):
    indir.mkdir(parents=True, exist_ok=True)

    def w(name, arr):
        np.asarray(arr, dtype="<f8").flatten(order="F").tofile(indir / name)

    w("A", m.A.cpu().numpy().T)
    w("mean", m.mean.cpu().numpy().reshape(-1))
    w("mu", m.mu.cpu().numpy())
    w("sbeta", m.beta.cpu().numpy())
    w("rho", m.rho.cpu().numpy())
    w("alpha", m.alpha.cpu().numpy())
    w("gm", m.gm.cpu().numpy())
    w("c", m.c.cpu().numpy())


def _mat(f, shape):
    return np.fromfile(f, dtype=np.float64).reshape(shape, order="F")


def fortran_state(run_root, indir, k):
    """Run Fortran max_iter=k; return its state after k updates + LL list."""
    wd = run_root / f"F_{k}"
    wd.mkdir(parents=True, exist_ok=True)
    if not (wd / "eeglab_data.fdt").exists():
        (wd / "eeglab_data.fdt").symlink_to(SAMPLE / "eeglab_data.fdt")
    if (wd / "out").exists():
        shutil.rmtree(wd / "out")
    (wd / "run.param").write_text(PARAM.format(indir=str(indir), k=k))
    subprocess.run(
        [str(BIN), "run.param"], cwd=wd, capture_output=True, text=True, timeout=600
    )
    o = wd / "out"
    log = (o / "out.txt").read_text()
    ll = [
        float(m.group(1))
        for m in re.finditer(r"iter\s+\d+\s+lrate\s*=\s*\S+\s+LL\s*=\s*(\S+)", log)
    ]
    return dict(
        W=_mat(o / "W", (NW, NW)),
        mu=_mat(o / "mu", (NMIX, NW)),
        sbeta=_mat(o / "sbeta", (NMIX, NW)),
        rho=_mat(o / "rho", (NMIX, NW)),
        alpha=_mat(o / "alpha", (NMIX, NW)),
        gm=np.fromfile(o / "gm", dtype=np.float64),
        c=_mat(o / "c", (NW, 1)),
        ll=ll,
    )


def seed_model(base, st):
    """Fresh CorrectedNG seeded from a Fortran state dict `st`."""
    m = CorrectedNG(**HYPER)
    m.sphere = base.sphere.clone()
    m.mean = base.mean.clone()
    m.sldet = base.sldet
    m.comp_list = torch.arange(NW).unsqueeze(1)
    m.A = torch.tensor(np.linalg.inv(st["W"].T))  # true unmixing = inv(A).T
    m.mu = torch.tensor(st["mu"].copy())
    m.beta = torch.tensor(st["sbeta"].copy())
    m.rho = torch.tensor(st["rho"].copy())
    m.alpha = torch.tensor(st["alpha"].copy())
    m.gm = torch.tensor(st["gm"].copy())
    m.c = torch.tensor(st["c"].copy())
    m.iteration = 10
    m.lrate = m.lrate_cap = 0.05
    m.n_newton_fallbacks = 0
    m._update_unmixing_matrices()
    return m


def rowcorr_defect(Wa, Wb):
    a = Wa / np.linalg.norm(Wa, axis=1, keepdims=True)
    b = Wb / np.linalg.norm(Wb, axis=1, keepdims=True)
    return 1.0 - float(np.abs((a * b).sum(1)).mean())  # matched rows, no permutation


def main():
    run_root = (
        Path(sys.argv[1])
        if len(sys.argv) > 1 and not sys.argv[1].startswith("--")
        else Path(__file__).resolve().parent / "_drift"
    )
    run_root.mkdir(parents=True, exist_ok=True)
    raw = load_eeglab_data(
        str(SAMPLE / "eeglab_data.fdt"), data_dim=NW, field_dim=FIELD
    ).astype(np.float64)
    base = snapshot_init(raw)
    indir = run_root / "init"
    write_load_files(indir, base)
    Xs = base.sphere @ (torch.tensor(raw) - base.mean)
    n = Xs.shape[1]

    F = {
        0: dict(
            W=np.linalg.inv(base.A.cpu().numpy()).T,
            mu=base.mu.cpu().numpy(),
            sbeta=base.beta.cpu().numpy(),
            rho=base.rho.cpu().numpy(),
            alpha=base.alpha.cpu().numpy(),
            gm=base.gm.cpu().numpy(),
            c=base.c.cpu().numpy(),
        )
    }
    print(f"running Fortran incrementally (FIX_RHO={FIX_RHO}) ...")
    for k in range(1, K + 1):
        F[k] = fortran_state(run_root, indir, k)

    print(f"\n{'=' * 96}")
    print(
        "Teacher-forced one-step M-step residual   "
        "abs=max|P_{k+1}-F_{k+1}| ; step=max|F_{k+1}-F_k|"
    )
    print(f"{'=' * 96}")
    print(
        f"{'k':>3} {'F_LL_k1':>9} {'Wdefect':>9} | "
        f"{'mu_abs':>9} {'mu_step':>9} | {'be_abs':>9} {'be_step':>9} | "
        f"{'rho_abs':>9} {'rho_step':>9} | {'rho_rng':>13}"
    )

    def ab(pk1, fk1, fk):
        return np.abs(pk1 - fk1).max(), np.abs(fk1 - fk).max()

    for k in range(0, K):
        m = seed_model(base, F[k])
        m._update_parameters(m._accumulate_blocks(Xs), n)
        P = dict(
            W=m.W[:, :, 0].T.cpu().numpy(),
            mu=m.mu.cpu().numpy(),
            beta=m.beta.cpu().numpy(),
            rho=m.rho.cpu().numpy(),
        )
        fk, fk1 = F[k], F[k + 1]
        mu_a, mu_s = ab(P["mu"], fk1["mu"], fk["mu"])
        be_a, be_s = ab(P["beta"], fk1["sbeta"], fk["sbeta"])
        rh_a, rh_s = ab(P["rho"], fk1["rho"], fk["rho"])
        print(
            f"{k:>3} {fk1['ll'][-1]:>9.5f} {rowcorr_defect(P['W'], fk1['W']):>9.2e} | "
            f"{mu_a:>9.2e} {mu_s:>9.2e} | {be_a:>9.2e} {be_s:>9.2e} | "
            f"{rh_a:>9.2e} {rh_s:>9.2e} | "
            f"{fk1['rho'].min():.3f}-{fk1['rho'].max():.3f}"
        )

    print(
        "\nWdefect=1-|matched row corr| of unmixing. If *_abs >> *_step, that update drifts."
    )


if __name__ == "__main__":
    main()
