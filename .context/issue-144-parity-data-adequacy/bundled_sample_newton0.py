"""do_newton=0 parity + Fortran-vs-Fortran self-consistency on the bundled
32-channel sample EEG (the dataset paper.md's Table 1 actually describes:
"uses only the bundled real sample EEG ... with no external download").

Runs N seeds of: Fortran (amica15mac, do_newton overridden to 0) and
AMICATorchNG (do_newton=False), then reports:
  - mean/min Hungarian-matched correlation + Amari distance, NG vs Fortran
  - mean/min Hungarian-matched correlation + Amari distance, Fortran vs Fortran

    uv run python bundled_sample_newton0.py [n_seeds] [max_iter] [out_dir]
"""

import itertools
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO))
from pyAMICA import AMICA  # noqa: E402
from pyAMICA.torch_impl.utils import load_eeglab_data  # noqa: E402


def xcorr(Wa, Wb):
    na = Wa / (np.linalg.norm(Wa, axis=1, keepdims=True) + 1e-12)
    nb = Wb / (np.linalg.norm(Wb, axis=1, keepdims=True) + 1e-12)
    corr = np.abs(na @ nb.T)
    r, c = linear_sum_assignment(1 - corr)
    return corr[r, c]


def amari_index(gain):
    n = gain.shape[0]
    abs_gain = np.abs(gain)
    row_max = abs_gain.max(axis=1)
    col_max = abs_gain.max(axis=0)
    row_term = (abs_gain.sum(axis=1) / row_max - 1).sum()
    col_term = (abs_gain.sum(axis=0) / col_max - 1).sum()
    return (row_term + col_term) / (2 * n * (n - 1))


def amari_distance(Wa, Wb):
    forward = amari_index(Wa @ np.linalg.pinv(Wb))
    backward = amari_index(Wb @ np.linalg.pinv(Wa))
    return float((forward + backward) / 2)


def run_fortran(data_dim, max_iter, seed, work_dir):
    work_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        REPO / "pyAMICA/sample_data/eeglab_data.fdt", work_dir / "eeglab_data.fdt"
    )
    template = (REPO / "pyAMICA/sample_data/input.param").read_text().splitlines()
    lines = []
    for line in template:
        if line.startswith("files"):
            lines.append("files ./eeglab_data.fdt")
        elif line.startswith("outdir"):
            lines.append("outdir ./fortran_output/")
        elif line.startswith("max_iter"):
            lines.append(f"max_iter {max_iter}")
        elif line.startswith("do_newton"):
            lines.append("do_newton 0")
        else:
            lines.append(line)
    (work_dir / "input.param").write_text("\n".join(lines) + "\n")
    (work_dir / "fortran_output").mkdir(exist_ok=True)

    result = subprocess.run(
        [str(REPO / "pyAMICA/sample_data/amica15mac"), "input.param"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        timeout=600,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Fortran failed (seed {seed}): {result.stderr}")

    W = np.fromfile(work_dir / "fortran_output/W", dtype=np.float64).reshape(
        data_dim, data_dim, order="F"
    )
    ll = None
    out_txt = work_dir / "fortran_output/out.txt"
    if out_txt.exists():
        for line in out_txt.read_text().splitlines():
            if line.strip().startswith("iter"):
                parts = line.split()
                if "LL" in parts:
                    ll = float(parts[parts.index("LL") + 2])
    return W, ll


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    max_iter = int(sys.argv[2]) if len(sys.argv) > 2 else 2000
    out_dir = (
        Path(sys.argv[3])
        if len(sys.argv) > 3
        else Path(tempfile.gettempdir()) / "pyamica_newton0_bundled"
    )
    seeds = list(range(301, 301 + n_seeds))

    with open(REPO / "pyAMICA/sample_data/sample_params.json") as f:
        params = json.load(f)
    data_dim = params["data_dim"]
    field_dim = params["field_dim"][0]

    data = load_eeglab_data(
        str(REPO / "pyAMICA/sample_data/eeglab_data.fdt"),
        data_dim=data_dim,
        field_dim=field_dim,
        dtype=np.float32,
    ).astype(np.float64)

    tmp_root = out_dir
    tmp_root.mkdir(parents=True, exist_ok=True)

    fortran_Ws = {}
    ng_corrs = []
    ng_amaris = []
    for seed in seeds:
        work_dir = tmp_root / f"seed{seed}"
        W_f, ll_f = run_fortran(data_dim, max_iter, seed, work_dir)
        fortran_Ws[seed] = W_f
        print(f"seed {seed}: Fortran done, LL={ll_f}", flush=True)

        np.random.seed(seed)
        torch.manual_seed(seed)
        model = AMICA(n_models=1, n_mix=params.get("num_mix", 3), verbose=False)
        model.fit(
            data,
            max_iter=max_iter,
            lrate=params.get("lrate", 0.05),
            do_mean=params.get("do_mean", True),
            do_sphere=params.get("do_sphere", True),
            do_approx_sphere=params.get("do_approx_sphere", True),
            do_newton=False,
            seed=seed,
        )
        W_ng = model.get_unmixing_matrix(0)

        corrs = xcorr(W_f, W_ng)
        amari = amari_distance(W_f, W_ng)
        ng_corrs.append(corrs)
        ng_amaris.append(amari)
        print(
            f"seed {seed}: NG vs Fortran mean_corr={corrs.mean():.4f} "
            f"min={corrs.min():.4f} amari={amari:.4f} ng_ll={model.final_ll_:.4f}",
            flush=True,
        )

    all_means = np.array([c.mean() for c in ng_corrs])
    all_mins = np.array([c.min() for c in ng_corrs])
    print(
        f"\nNG-vs-Fortran ({n_seeds} seeds): mean_corr={all_means.mean():.4f} "
        f"(sd {all_means.std():.4f}), min_corr overall={all_mins.min():.4f}, "
        f"mean_amari={np.mean(ng_amaris):.4f}",
        flush=True,
    )

    ff_means = []
    ff_mins = []
    ff_amaris = []
    for a, b in itertools.combinations(seeds, 2):
        corrs = xcorr(fortran_Ws[a], fortran_Ws[b])
        amari = amari_distance(fortran_Ws[a], fortran_Ws[b])
        ff_means.append(corrs.mean())
        ff_mins.append(corrs.min())
        ff_amaris.append(amari)
        print(
            f"fortran seed {a} vs {b}: mean_corr={corrs.mean():.4f} "
            f"min={corrs.min():.4f} amari={amari:.4f}",
            flush=True,
        )

    if not ff_means:
        print(
            "\nFortran-vs-Fortran: skipped (need at least 2 seeds for a pair)",
            flush=True,
        )
        return

    ff_means = np.array(ff_means)
    ff_mins = np.array(ff_mins)
    print(
        f"\nFortran-vs-Fortran ({n_seeds} seeds, {len(ff_means)} pairs): "
        f"mean_corr={ff_means.mean():.4f} (sd {ff_means.std():.4f}), "
        f"min_corr overall={ff_mins.min():.4f}, mean_amari={np.mean(ff_amaris):.4f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
