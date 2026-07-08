"""Unit tests for the CPU core-count scaling sweep wiring (issue #86).

Pure classification/config checks (no data or GPU needed): they lock which
backends the --threads sweep iterates over (the CPU ones) vs which run once
(GPU, thread-independent).
"""

import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


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
