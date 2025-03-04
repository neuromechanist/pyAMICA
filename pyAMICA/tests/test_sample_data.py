"""Tests for pyAMICA using sample data."""

import os.path as op
import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path

import pyAMICA
from pyAMICA import AMICA
from pyAMICA.amica_data import load_data_file
from pyAMICA.amica_load import loadmodout

# Setup sample data paths
sample_data_path = op.join(pyAMICA.__path__[0], 'sample_data')
sample_params_file = op.join(sample_data_path, 'sample_params.json')
eeglab_data_file = op.join(sample_data_path, 'eeglab_data.fdt')
amicaout_dir = op.join(sample_data_path, 'amicaout')


def test_sample_data_scikit():
    """Test pyAMICA using scikit-learn style API."""
    # Load original results for comparison
    orig_results = loadmodout(amicaout_dir)

    # Initialize and fit AMICA model using scikit-learn style API
    model = AMICA.from_json_file(sample_params_file)
    model.fit()

    # Compare weights
    W_pyamica = model.W[:, :, 0]  # Get weights from first model

    correlations = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            correlations[i, j] = abs(np.corrcoef(W_pyamica[i], orig_results.W[j, :, 0])[0, 1])

    # Verify results
    max_correlations = np.max(correlations, axis=1)
    assert np.all(max_correlations > 0.8), "Some components don't match original results"

    best_matches = np.argmax(correlations, axis=1)
    assert len(np.unique(best_matches)) == 32, "Some components are duplicated"


def test_sample_data_cli():
    """Test pyAMICA using CLI interface."""
    import subprocess
    import sys
    from pathlib import Path

    # Run AMICA through CLI
    cli_path = Path(__file__).parent.parent / 'amica_cli.py'
    test_outdir = Path('test_output')

    try:
        subprocess.run([
            sys.executable,
            str(cli_path),
            sample_params_file,
            '--outdir', str(test_outdir)
        ], check=True)

        # Load and compare results
        orig_results = loadmodout(amicaout_dir)
        test_results = loadmodout(test_outdir)

        correlations = np.zeros((32, 32))
        for i in range(32):
            for j in range(32):
                correlations[i, j] = abs(np.corrcoef(
                    test_results.W[i, :, 0],
                    orig_results.W[j, :, 0])[0, 1])

        # Verify results
        max_correlations = np.max(correlations, axis=1)
        assert np.all(max_correlations > 0.8), "Some components don't match original results"

        best_matches = np.argmax(correlations, axis=1)
        assert len(np.unique(best_matches)) == 32, "Some components are duplicated"

    finally:
        # Cleanup
        if test_outdir.exists():
            import shutil
            shutil.rmtree(test_outdir)


def test_sample_data_light():
    """Light version of the test for CI that runs only 50 iterations."""
    # Load and preprocess the data
    data = load_data_file(eeglab_data_file, 32, 30504, 1)

    # Initialize AMICA model with reduced iterations
    model = AMICA.from_json_file(sample_params_file)
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


if __name__ == '__main__':
    pytest.main([__file__])
