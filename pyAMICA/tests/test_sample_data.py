"""Tests for pyAMICA using sample data."""

import os.path as op
import numpy as np
import pytest
from pathlib import Path

import pyAMICA
from pyAMICA import AMICA_NumPy as AMICA
from pyAMICA.amica_data import load_data_file
from pyAMICA.amica_load import loadmodout

# Setup sample data paths
sample_data_path = op.join(pyAMICA.__path__[0], "sample_data")
sample_params_file = op.join(sample_data_path, "sample_params.json")
eeglab_data_file = op.join(sample_data_path, "eeglab_data.fdt")
amicaout_dir = op.join(sample_data_path, "amicaout")


@pytest.mark.slow
@pytest.mark.xfail(
    reason="full 2000-iter run does not match Fortran (LL scale/sign bug, "
    "AGENTS.md Known Issue #2; parity gated by epic #9)",
    strict=False,
)
def test_sample_data_scikit(tmp_path):
    """Test pyAMICA using scikit-learn style API."""
    # Load original results for comparison
    orig_results = loadmodout(amicaout_dir)

    # Initialize and fit AMICA model using scikit-learn style API.
    # Override outdir (the params file defaults to the relative './amicaout/')
    # so this test does not write stray output into the repo root.
    model = AMICA.from_json_file(sample_params_file, outdir=str(tmp_path / "amicaout"))
    model.fit()

    # Compare weights
    W_pyamica = model.W[:, :, 0]  # Get weights from first model

    correlations = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            correlations[i, j] = abs(
                np.corrcoef(W_pyamica[i], orig_results.W[j, :, 0])[0, 1]
            )

    # Verify results
    max_correlations = np.max(correlations, axis=1)
    assert np.all(max_correlations > 0.8), (
        "Some components don't match original results"
    )

    best_matches = np.argmax(correlations, axis=1)
    assert len(np.unique(best_matches)) == 32, "Some components are duplicated"


@pytest.mark.slow
@pytest.mark.xfail(
    reason="full 2000-iter run does not match Fortran (LL scale/sign bug, "
    "AGENTS.md Known Issue #2; parity gated by epic #9)",
    strict=False,
)
def test_sample_data_cli():
    """Test pyAMICA using CLI interface."""
    import subprocess
    import sys

    # amica_cli.py uses relative imports, so it must be run as a module
    # (see its module docstring), not as a direct script path.
    test_outdir = Path("test_output")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyAMICA.amica_cli",
                sample_params_file,
                "--outdir",
                str(test_outdir),
            ],
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Load and compare results
        orig_results = loadmodout(amicaout_dir)
        test_results = loadmodout(test_outdir)

        correlations = np.zeros((32, 32))
        for i in range(32):
            for j in range(32):
                correlations[i, j] = abs(
                    np.corrcoef(test_results.W[i, :, 0], orig_results.W[j, :, 0])[0, 1]
                )

        # Verify results
        max_correlations = np.max(correlations, axis=1)
        assert np.all(max_correlations > 0.8), (
            "Some components don't match original results"
        )

        best_matches = np.argmax(correlations, axis=1)
        assert len(np.unique(best_matches)) == 32, "Some components are duplicated"

    finally:
        # Cleanup
        if test_outdir.exists():
            import shutil

            shutil.rmtree(test_outdir)


@pytest.mark.xfail(
    reason="components not fully decorrelated after only 50 iterations "
    "(parity/convergence gated by epic #9)",
    strict=False,
)
def test_sample_data_light(tmp_path):
    """Light version of the test for CI that runs only 50 iterations."""
    # Load and preprocess the data (eeglab_data.fdt stores float32 samples)
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32)

    # Initialize AMICA model with reduced iterations. Override outdir (the
    # params file defaults to the relative './amicaout/') so this test does
    # not write stray output into the repo root.
    model = AMICA.from_json_file(sample_params_file, outdir=str(tmp_path / "amicaout"))
    model.max_iter = 50  # Override max_iter for quick testing

    # Fit the model
    model.fit(data)

    # Basic checks
    W = model.get_weights()
    assert W.shape == (32, 32), f"Expected weight matrix shape (32, 32), got {W.shape}"
    assert not np.any(np.isnan(W)), "Weight matrix contains NaN values"
    assert not np.any(np.isinf(W)), "Weight matrix contains infinite values"

    # Check if weights are reasonably scaled
    assert np.all(np.abs(W) < 100), "Weight values are unreasonably large"
    assert np.all(np.abs(W) > 1e-6), "Weight values are unreasonably small"

    # Check if components are decorrelated
    Y = model.transform(data)
    corr = np.corrcoef(Y[:, :, 0])
    np.fill_diagonal(corr, 0)  # Remove diagonal elements
    assert np.all(np.abs(corr) < 0.1), "Components are not sufficiently decorrelated"


if __name__ == "__main__":
    pytest.main([__file__])
