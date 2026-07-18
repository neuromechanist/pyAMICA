"""Issue #24: separate "different random-init basin" from a residual per-iteration
M-step/Newton bit-parity gap for the NG backend.

Method: seed BOTH the Fortran reference binary and the corrected NG model
(``.context/issue-21/corrected_mstep_prototype.py``) from the *same* initial
parameters (the Python NG init, seed=42) and compare their trajectories/endpoints.

Fortran is seeded through its ``load_*`` mechanism: float64 column-major files
``mean, A, mu, sbeta, rho, alpha, gm, c`` under ``indir`` (the same format the
binary writes). ``load_sphere`` is intentionally NOT used -- the reference binary
segfaults in that path (it never allocates the ``Stmp2`` temp it later uses at
amica17.f90:553). Instead Fortran computes its own sphere, which is the same
symmetric ZCA (``do_approx_sphere=.true.`` by default) as ``CorrectedNG._preprocess``;
the harness asserts ``max|S_fortran - S_python|`` is tiny so the two truly share a
sphered space.

Transpose mapping: the Python true unmixing is ``inv(self.A).T`` (Fortran ``W``),
so Fortran mixing ``A = inv(W) = self.A.T``.

Run (CPU/float64; MPS lacks float64):
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python .context/issue-24/verify_init_basin.py [run_dir]

Result (see findings.md): from the identical init Fortran reaches the gold -3.402
(corr 0.998) while Python plateaus at -3.460 (corr 0.51). The -3.46 basin is a
residual M-step bit-parity bug, NOT init. It is already present in the first-order
natural-gradient phase: with lrate pinned at 0.05 (Fortran's NG value) Python tracks
Fortran for ~5 iters, then reverses and descends. The fit-loop lrate ratchet masks
this as a benign "-3.46 plateau".
"""

from __future__ import annotations

import importlib.util
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

REPO = Path(__file__).resolve().parents[2]
SAMPLE = REPO / "pamica" / "sample_data"
AO = SAMPLE / "amicaout"
BIN = SAMPLE / "amica15mac"
PROTO = REPO / ".context" / "issue-21" / "corrected_mstep_prototype.py"
NW, NMIX, FIELD, SEED = 32, 3, 30504, 42


def _import_corrected_ng():
    sys.path.insert(0, str(REPO))
    spec = importlib.util.spec_from_file_location("corrected_mstep_prototype", PROTO)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load prototype at {PROTO}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.CorrectedNG, mod.load_eeglab_data


CorrectedNG, load_eeglab_data = _import_corrected_ng()


# Fortran-matched hyperparameters (from pamica/sample_data/input.param).
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
)


def load_raw() -> np.ndarray:
    return load_eeglab_data(
        str(SAMPLE / "eeglab_data.fdt"), data_dim=NW, field_dim=FIELD
    ).astype(np.float64)


def snapshot_init(raw: np.ndarray, do_newton: bool) -> dict:
    """Build the model, run preprocess+init, return the pristine init state.

    A fresh ``fit(raw, ...)`` with the same seed reproduces this exact init, so the
    Fortran run (seeded from this snapshot) and the Python fit start identically.
    """
    m = CorrectedNG(do_newton=do_newton, **HYPER)
    m._preprocess(raw)  # sets m.sphere, m.mean, m.sldet
    m._initialize_parameters()  # sets A, mu, beta, rho, alpha, gm, c, W
    return dict(
        A=m.A.cpu().numpy().copy(),  # internal mixing (nw, ncomp)
        mu=m.mu.cpu().numpy().copy(),  # (nmix, ncomp)
        beta=m.beta.cpu().numpy().copy(),  # sbeta (nmix, ncomp)
        rho=m.rho.cpu().numpy().copy(),  # (nmix, ncomp)
        alpha=m.alpha.cpu().numpy().copy(),  # (nmix, ncomp)
        gm=m.gm.cpu().numpy().copy(),  # (nmodels,)
        c=m.c.cpu().numpy().copy(),  # (nw, nmodels)
        sphere=m.sphere.cpu().numpy().copy(),  # (nw, nw)
        mean=m.mean.cpu().numpy().copy().reshape(-1),  # (nw,)
    )


def write_load_files(indir: Path, snap: dict) -> None:
    indir.mkdir(parents=True, exist_ok=True)

    def w(name: str, arr: np.ndarray) -> None:
        np.asarray(arr, dtype="<f8").flatten(order="F").tofile(indir / name)

    w("A", snap["A"].T)  # Fortran mixing = python_internal_A.T
    w("mean", snap["mean"])
    w("mu", snap["mu"])
    w("sbeta", snap["beta"])
    w("rho", snap["rho"])
    w("alpha", snap["alpha"])
    w("gm", snap["gm"])
    w("c", snap["c"])


PARAM_TEMPLATE = """\
files ./eeglab_data.fdt
outdir {outdir}
indir {indir}
block_size 512
do_opt_block 0
num_models 1
max_threads 8
use_min_dll 1
min_dll 1.000000e-09
use_grad_norm 1
min_grad_norm 1.000000e-07
num_mix_comps 3
pdftype 0
max_iter {max_iter}
num_samples 1
data_dim 32
field_dim 30504
field_blocksize 1
do_history 0
lrate 0.050000
minlrate 1.000000e-08
mineig 1.000000e-12
lratefact 0.500000
rholrate 0.050000
rho0 1.500000
minrho 1.000000
maxrho 2.000000
rholratefact 0.500000
do_newton {do_newton}
newt_start 50
newt_ramp 10
newtrate 1.000000
do_reject 0
writestep 200
write_nd 0
write_LLt 0
decwindow 1
max_decs 3
fix_init 0
update_A 1
update_c 1
update_gm 1
update_alpha 1
update_mu 1
update_beta 1
invsigmax 100.000000
invsigmin 0.000000
do_rho 1
do_mean 1
do_sphere 1
doPCA 1
pcakeep 32
pcadb 30.000000
byte_size 4
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


def run_fortran(workdir: Path, indir: Path, do_newton: bool, max_iter: int) -> dict:
    workdir.mkdir(parents=True, exist_ok=True)
    fdt = workdir / "eeglab_data.fdt"
    if not fdt.exists():
        fdt.symlink_to(SAMPLE / "eeglab_data.fdt")
    outdir = workdir / "out"
    if outdir.exists():
        shutil.rmtree(outdir)
    param = workdir / "run.param"
    param.write_text(
        PARAM_TEMPLATE.format(
            outdir="./out/",
            indir=str(indir),
            max_iter=max_iter,
            do_newton=1 if do_newton else 0,
        )
    )
    proc = subprocess.run(
        [str(BIN), "run.param"],
        cwd=workdir,
        capture_output=True,
        text=True,
        timeout=1200,
    )
    out_txt = outdir / "out.txt"
    log = out_txt.read_text() if out_txt.exists() else proc.stdout

    def read_mat(name):
        f = outdir / name
        return (
            np.fromfile(f, dtype=np.float64).reshape(NW, NW, order="F")
            if f.exists()
            else None
        )

    return dict(
        ll=parse_fortran_ll(log),
        W=read_mat("W"),
        S=read_mat("S"),
        returncode=proc.returncode,
        stderr=proc.stderr,
    )


def parse_fortran_ll(log: str) -> list[float]:
    lls = []
    for line in log.splitlines():
        m = re.search(r"iter\s+\d+\s+lrate\s*=\s*\S+\s+LL\s*=\s*(\S+)", line)
        if m:
            lls.append(float(m.group(1)))
    return lls


def run_python(raw: np.ndarray, do_newton: bool, max_iter: int, **overrides) -> dict:
    m = CorrectedNG(do_newton=do_newton, **{**HYPER, **overrides})
    m.fit(raw, max_iter=max_iter, verbose=False)
    return dict(
        ll=list(m.ll_history),
        W=m.W[:, :, 0].T.cpu().numpy(),  # true unmixing (matches Fortran W)
        sphere=m.sphere.cpu().numpy(),
        stop_reason=getattr(m, "stop_reason", "?"),
        n_newton_fallbacks=getattr(m, "n_newton_fallbacks", -1),
    )


def total_filter_corr(Wa, Sa, Wb, Sb) -> float:
    """Mean abs correlation of Hungarian-matched rows of the total spatial filter
    ``W@S`` (basis-invariant; raw ``W`` rows live in different sphered bases)."""
    Fa, Fb = Wa @ Sa, Wb @ Sb
    a = Fa / np.linalg.norm(Fa, axis=1, keepdims=True)
    b = Fb / np.linalg.norm(Fb, axis=1, keepdims=True)
    C = np.abs(a @ b.T)
    r, c = linear_sum_assignment(1 - C)
    return float(C[r, c].mean())


def fmt(ll, idxs) -> str:
    return "  ".join(f"i{i}={ll[i]:.5f}" if i < len(ll) else f"i{i}=--" for i in idxs)


def main() -> None:
    if len(sys.argv) > 1:
        run_root = Path(sys.argv[1])
    else:
        run_root = Path(tempfile.mkdtemp(prefix="issue24_"))
    run_root.mkdir(parents=True, exist_ok=True)
    print(f"run dir: {run_root}")
    max_iter = 200
    raw = load_raw()
    idxs = [0, 1, 5, 10, 20, 49, 50, 60, 100, 150, 199]

    W_ref = np.fromfile(AO / "W", dtype=np.float64).reshape(NW, NW, order="F")
    S_ref = np.fromfile(AO / "S", dtype=np.float64).reshape(NW, NW, order="F")

    summary = []
    for tag, do_newton in [("ng_only", False), ("full_newton", True)]:
        print(f"\n{'=' * 72}\nCONFIG: {tag} (do_newton={do_newton})\n{'=' * 72}")
        snap = snapshot_init(raw, do_newton)
        indir = run_root / f"init_{tag}"
        write_load_files(indir, snap)

        f = run_fortran(run_root / f"fortran_{tag}", indir, do_newton, max_iter)
        if f["returncode"] != 0 or not f["ll"]:
            print(f"  Fortran rc={f['returncode']} stderr:\n{f['stderr'][-400:]}")
            continue
        p = run_python(raw, do_newton, max_iter)

        if f["S"] is not None:
            print(
                f"\n  seeding check: max|S_fortran - S_python| = "
                f"{np.abs(f['S'] - snap['sphere']).max():.2e}   "
                f"init LL F={f['ll'][0]:.5f} P={p['ll'][0]:.5f} (equal => same start)"
            )
        print(f"  Fortran : {fmt(f['ll'], idxs)}")
        print(
            f"  Python  : {fmt(p['ll'], idxs)}   "
            f"[stop={p['stop_reason']} newton_fallbacks={p['n_newton_fallbacks']}]"
        )

        cf = total_filter_corr(f["W"], snap["sphere"], W_ref, S_ref)
        cp = total_filter_corr(p["W"], p["sphere"], W_ref, S_ref)
        print(
            f"  endpoint LL  Fortran={f['ll'][-1]:.5f}  Python={p['ll'][-1]:.5f}  "
            f"(gold ref = -3.40187)"
        )
        print(f"  corr vs gold total-filter:  Fortran={cf:.4f}  Python={cp:.4f}")
        summary.append((tag, f["ll"][-1], p["ll"][-1], cf, cp))

        if tag == "ng_only":
            # Isolate M-step direction from the fit-loop lrate ratchet: pin lrate
            # at 0.05 (Fortran's NG value) by disabling annealing.
            pf = run_python(raw, False, max_iter, lratefact=1.0, maxdecs=10**9)
            cpf = total_filter_corr(pf["W"], pf["sphere"], W_ref, S_ref)
            print("\n  [Python NG, lrate PINNED 0.05, ratchet OFF]")
            print(f"  Python  : {fmt(pf['ll'], idxs)}")
            print(
                f"  endpoint LL={pf['ll'][-1]:.5f} (Fortran NG={f['ll'][-1]:.5f})  "
                f"corr vs gold={cpf:.4f}"
            )
            summary.append(("ng_only_pinnedlr", f["ll"][-1], pf["ll"][-1], cf, cpf))

    print(f"\n{'#' * 72}\nSUMMARY\n{'#' * 72}")
    print(
        f"{'config':>18}  {'Fortran_end':>11}  {'Python_end':>10}  "
        f"{'corr(F,gold)':>12}  {'corr(P,gold)':>12}"
    )
    for tag, fe, pe, cf, cp in summary:
        print(f"{tag:>18}  {fe:>11.4f}  {pe:>10.4f}  {cf:>12.3f}  {cp:>12.3f}")


if __name__ == "__main__":
    main()
