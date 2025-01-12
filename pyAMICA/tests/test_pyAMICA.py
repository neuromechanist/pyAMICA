import os.path as op
import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path
import tempfile
import shutil

import pyAMICA
from pyAMICA import AMICA
from pyAMICA.amica_utils import create_output_dirs
from pyAMICA.amica_data import load_data_file, preprocess_data
from pyAMICA.amica_viz import plot_components
from pyAMICA.amica_pdf import compute_pdf

# Setup test data path
data_path = op.join(pyAMICA.__path__[0], 'data')


@pytest.fixture
def random_data():
    """Generate random test data."""
    rng = np.random.RandomState(42)
    n_channels = 64
    n_samples = 1000
    return rng.randn(n_channels, n_samples)


@pytest.fixture
def temp_dir():
    """Create and clean up a temporary directory."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_amica_initialization():
    """Test AMICA model initialization with various parameters."""
    # Test default initialization
    model = AMICA()
    assert model.num_models == 1
    assert model.max_iter == 1000

    # Test custom parameters
    model = AMICA(num_models=2, max_iter=500, do_newton=True)
    assert model.num_models == 2
    assert model.max_iter == 500
    assert model.do_newton is True

    # Test invalid parameters
    with pytest.raises(ValueError):
        AMICA(num_models=0)
    with pytest.raises(ValueError):
        AMICA(max_iter=-1)


def test_data_preprocessing(random_data):
    """Test data preprocessing functionality."""
    # Test mean removal
    data, mean, _ = preprocess_data(random_data.copy(), do_mean=True, do_sphere=False)
    npt.assert_allclose(data.mean(axis=1), np.zeros(data.shape[0]), atol=1e-10)
    npt.assert_allclose(mean.ravel(), random_data.mean(axis=1))

    # Test sphering
    data, _, sphere = preprocess_data(random_data.copy(), do_mean=False, do_sphere=True)
    cov = np.cov(data)
    npt.assert_allclose(cov, np.eye(data.shape[0]), atol=1e-10)


def test_pdf_computation():
    """Test PDF computation for different distributions."""
    y = np.linspace(-5, 5, 100)

    # Test Laplace distribution
    pdf, dpdf = compute_pdf(y, rho=1.0)
    npt.assert_allclose(pdf, np.exp(-np.abs(y)) / 2.0)
    npt.assert_allclose(dpdf, -np.sign(y) * pdf)

    # Test Gaussian distribution
    pdf, dpdf = compute_pdf(y, rho=2.0)
    npt.assert_allclose(pdf, np.exp(-y * y) / np.sqrt(np.pi))
    npt.assert_allclose(dpdf, -2 * y * pdf)


def test_output_directory_creation(temp_dir):
    """Test creation of output directories."""
    outdir = Path(temp_dir) / 'test_output'
    create_output_dirs(outdir)

    # Check that directories were created
    assert outdir.exists()
    assert (outdir / 'figures').exists()
    assert (outdir / 'results').exists()


def test_data_loading(temp_dir):
    """Test data loading functionality."""
    # Create test data
    data = np.random.randn(10, 100).astype(np.float32)
    data_file = Path(temp_dir) / 'test.bin'

    # Save in Fortran format
    with open(data_file, 'wb') as f:
        data.T.tofile(f)

    # Test loading
    loaded_data = load_data_file(data_file, 10, 100, 1, dtype=np.float32)
    npt.assert_allclose(loaded_data, data)

    # Test error handling
    with pytest.raises(FileNotFoundError):
        load_data_file(Path(temp_dir) / 'nonexistent.bin', 10, 100, 1)


def test_full_pipeline(random_data):
    """Test complete AMICA pipeline with simple data."""
    # Create simple mixing scenario
    n_sources = 4
    n_samples = 1000
    rng = np.random.RandomState(42)

    # Create independent sources
    S = rng.laplace(size=(n_sources, n_samples))
    A_true = rng.randn(n_sources, n_sources)
    X = np.dot(A_true, S)

    # Fit AMICA model
    model = AMICA(num_models=1, max_iter=100, do_newton=True, seed=42)
    model.fit(X)

    # Test transform
    S_est = model.transform(X)
    assert S_est.shape == (n_sources, n_samples, 1)

    # Test correlation with true sources
    corr = np.abs(np.corrcoef(S.ravel(), S_est[:, :, 0].ravel())[0, 1])
    assert corr > 0.8  # Should have high correlation with true sources


def test_visualization(random_data, temp_dir):
    """Test visualization functions."""
    outdir = Path(temp_dir)

    # Test component plotting
    fig = plot_components(random_data, outdir=outdir)
    assert fig is not None
    assert (outdir / 'figures').exists()


def test_error_handling():
    """Test error handling in various scenarios."""
    model = AMICA()

    # Test fitting with invalid data
    with pytest.raises(ValueError):
        model.fit(np.array([]))  # Empty array

    with pytest.raises(ValueError):
        model.fit(np.ones((10,)))  # 1D array

    # Test transform before fit
    with pytest.raises(RuntimeError):
        model.transform(np.random.randn(10, 100))
