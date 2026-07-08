"""Structural tests for the native-Fortran benchmark adapter (issue #85).

Real sample EEG only (NO MOCKS): the ``.fdt`` round-trip and the ``out.txt``
parser are checked against the committed EEGLAB data and a committed amica run.
The end-to-end run test is skipped unless a runnable native amica binary is
present (built on the CUDA host via ``benchmarks/fortran/build_amica.sh``, or
pointed at by ``AMICA_FORTRAN_BIN``), so CI and Apple-only checkouts stay green.
"""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest

from pyAMICA.numpy_impl.data import load_data_file

_REPO = Path(__file__).resolve().parents[2]
SAMPLE_DIR = _REPO / "pyAMICA" / "sample_data"
FDT = SAMPLE_DIR / "eeglab_data.fdt"
OUT_TXT = SAMPLE_DIR / "amicaout" / "out.txt"
DATA_DIM, FIELD_DIM = 32, 30504


def _load_bench():
    """Load benchmarks/benchmark_dimsweep.py (not an installed package)."""
    path = _REPO / "benchmarks" / "benchmark_dimsweep.py"
    spec = importlib.util.spec_from_file_location("benchmark_dimsweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()


def test_write_fdt_roundtrip(tmp_path):
    """_write_fdt is the exact inverse of the canonical load_data_file, so
    re-writing the committed .fdt reproduces it byte-for-byte -- locking the
    float32 column-major (channel-fastest) convention."""
    data = load_data_file(str(FDT), DATA_DIM, FIELD_DIM, dtype=np.float32)
    assert data.shape == (DATA_DIM, FIELD_DIM)
    out = tmp_path / "roundtrip.fdt"
    bench._write_fdt(data, out)
    assert out.read_bytes() == FDT.read_bytes()


def test_write_fdt_channel_subset(tmp_path):
    """A channel subset writes channel-fastest, so the canonical reader reads it
    back exactly (the sweep slices full[:nc, :ns])."""
    data = load_data_file(str(FDT), DATA_DIM, FIELD_DIM, dtype=np.float32)
    nc, ns = 16, 5000
    sub = np.ascontiguousarray(data[:nc, :ns])
    out = tmp_path / "subset.fdt"
    bench._write_fdt(sub, out)
    back = load_data_file(str(out), nc, ns, dtype=np.float32)
    np.testing.assert_array_equal(back, sub)


def test_parse_fortran_out():
    """Parse the committed amica out.txt -> per-iter seconds + final LL."""
    secs, ll = bench._parse_fortran_out(OUT_TXT)
    assert len(secs) == 200  # max_iter of the committed run
    assert all(s >= 0.0 for s in secs)
    assert secs[0] > 0.0
    assert ll == pytest.approx(-3.4018729891, abs=1e-6)


def test_write_fortran_param(tmp_path):
    """The rendered param carries the matched benchmark knobs for the subset,
    and forces the full fixed-length iteration budget."""
    bench._write_fortran_param(
        tmp_path, nc=16, ns=5000, iters=30, threads=4, n_models=1
    )
    text = (tmp_path / "input.param").read_text()
    for expected in (
        "files ./bench.fdt",
        "data_dim 16",
        "field_dim 5000",
        "max_iter 30",
        "max_threads 4",
        "num_mix_comps 3",
        "pdftype 0",
        "do_newton 0",
        "block_size 512",
        "use_min_dll 0",  # run the full budget, matched to torch/mlx
        "use_grad_norm 0",
    ):
        assert expected in text, f"missing {expected!r} in rendered param"


_FBIN = os.environ.get("AMICA_FORTRAN_BIN")


@pytest.mark.skipif(
    not bench._fortran_available(_FBIN),
    reason="no runnable native amica binary (build via "
    "benchmarks/fortran/build_amica.sh or set AMICA_FORTRAN_BIN); skipped on "
    "CI / Apple-only checkouts",
)
def test_run_fortran_smoke():
    """End-to-end: a short native fit on a real full-width subset yields finite
    per-iteration timing and a sane converged LL. Uses the full 32 channels /
    all samples so per-iteration time clears amica's ~10 ms stamp resolution
    (a 16-ch/8k config rounds to 0.00 s and is intentionally rejected)."""
    # the committed .fdt is float32 on disk (byte_size 4); _write_fdt recasts to
    # float32 anyway, so read it at its real dtype.
    data = load_data_file(str(FDT), DATA_DIM, FIELD_DIM, dtype=np.float32)
    ms, ll = bench._run_fortran(data, iters=10, repeats=1, binary=_FBIN, threads=4)
    assert ms > 0.0
    assert -5.0 < ll < -2.0
