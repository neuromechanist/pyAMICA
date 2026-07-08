"""Unit tests for the CPU core-count scaling sweep wiring (issue #86).

Pure classification/config checks (no data or GPU needed): they lock which
backends the --threads sweep iterates over (the CPU ones) vs which run once
(GPU, thread-independent).
"""

import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np

from pyAMICA.numpy_impl.data import load_data_file

_REPO = Path(__file__).resolve().parents[2]
_FDT = _REPO / "pyAMICA" / "sample_data" / "eeglab_data.fdt"


def _load_bench():
    path = _REPO / "benchmarks" / "benchmark_dimsweep.py"
    spec = importlib.util.spec_from_file_location("benchmark_dimsweep", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


bench = _load_bench()


def test_is_cpu_classifies_backends():
    """CPU backends (swept by --threads) vs GPU backends (run once)."""
    cpu = {"torch-cpu-f64", "torch-cpu-f32", "numpy-cpu-f64", "native-fortran-f64"}
    gpu = {"torch-cuda-f64", "torch-cuda-f32", "torch-mps-f32", "mlx-f32"}
    for b in cpu:
        assert bench._is_cpu(b) is True, f"{b} should be CPU-swept"
    for b in gpu:
        assert bench._is_cpu(b) is False, f"{b} should not be thread-swept"


def test_is_cpu_covers_every_backend():
    """No backend is left unclassified (guards against a new backend slipping
    through the sweep/skip decision)."""
    for b in bench._BACKENDS:
        assert isinstance(bench._is_cpu(b), bool)


def test_report_surfaces_all_failed_backend(tmp_path):
    """A backend that errors on every row must not vanish from the report; it is
    listed under 'failed', while a working backend's fastest thread count wins."""
    rows = [
        # torch-cpu-f64 fails on every attempted row
        {
            "config": "m1",
            "channels": 32,
            "backend": "torch-cpu-f64",
            "threads": 4,
            "error": "CUDA OOM",
        },
        {
            "config": "m1",
            "channels": 32,
            "backend": "torch-cpu-f64",
            "threads": 8,
            "error": "CUDA OOM",
        },
        # numpy succeeds at two thread counts (8t is faster -> the summary pick)
        {
            "config": "m1",
            "channels": 32,
            "backend": "numpy-cpu-f64",
            "threads": 4,
            "ms_per_iter": 90.0,
            "final_ll": -3.3,
        },
        {
            "config": "m1",
            "channels": 32,
            "backend": "numpy-cpu-f64",
            "threads": 8,
            "ms_per_iter": 70.0,
            "final_ll": -3.3,
        },
    ]
    path = tmp_path / "r.json"
    path.write_text(json.dumps({"platform": {}, "rows": rows}))
    buf = io.StringIO()
    with redirect_stdout(buf):
        bench._report([str(path)])
    out = buf.getvalue()
    assert "torch-cpu-f64" in out and "FAILED" in out  # not silently dropped
    assert "70.00" in out  # fastest (8t) numpy number is the summary pick
    assert "90.00" not in out.split("CPU scaling")[0]  # slower 4t not in summary


def test_threads_recorded_end_to_end(tmp_path, monkeypatch):
    """main() --threads sweep records the actual thread count per row. Real 32ch
    sample EEG, numpy-only, 1 iter -- no GPU/native binary, cheap enough for CI."""
    data = load_data_file(str(_FDT), 32, 30504, dtype=np.float32)[:, :2000]
    npy = tmp_path / "d.npy"
    np.save(npy, data)
    out = tmp_path / "o.json"
    argv = [
        "prog",
        "--data",
        str(npy),
        "--channels",
        "32",
        "--samples",
        "2000",
        "--iters",
        "1",
        "--repeats",
        "1",
        "--backends",
        "numpy-cpu-f64",
        "--threads",
        "1,2",
        "--out",
        str(out),
    ]
    monkeypatch.setattr("sys.argv", argv)
    with redirect_stdout(io.StringIO()):
        assert bench.main() == 0
    rows = json.loads(out.read_text())["rows"]
    got = sorted(r["threads"] for r in rows if r["backend"] == "numpy-cpu-f64")
    assert got == [1, 2]  # both swept thread counts recorded, one row each
    assert all("error" not in r for r in rows)  # 1-iter numpy converges (finite LL)


def test_run_torch_restores_global_thread_count():
    """_run_torch pins set_num_threads for the CPU run but must restore the prior
    process-global value, so it can't leak into a later backend/run."""
    import torch

    # the committed .fdt is float32 on disk; read it at its real dtype
    data = load_data_file(str(_FDT), 32, 30504, dtype=np.float32)[:, :2000]
    before = torch.get_num_threads()
    bench._run_torch(
        np.ascontiguousarray(data),
        device="cpu",
        dtype_str="f64",
        iters=2,
        repeats=1,
        threads=2,
    )
    assert torch.get_num_threads() == before
