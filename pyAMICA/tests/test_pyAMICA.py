import os.path as op
import numpy as np
import numpy.testing as npt
import pytest
from pathlib import Path
import tempfile
import shutil

import pyAMICA
from pyAMICA import AMICA_NumPy as AMICA
from pyAMICA.numpy_impl.data import load_data_file, preprocess_data
from pyAMICA.numpy_impl.pdf import compute_pdf

# Setup test data path
data_path = op.join(pyAMICA.__path__[0], "data")


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
    assert model.max_iter == 2000

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


def test_do_reject_refused_on_numpy_backend():
    """do_reject is non-functional on the NumPy backend (issue #123): fit()
    must refuse it loudly with NotImplementedError rather than crash mid-EM-loop
    with a bare AttributeError. The guard fires before any data handling, so no
    data is needed to exercise it."""
    model = AMICA(do_reject=True)
    with pytest.raises(NotImplementedError, match="do_reject"):
        model.fit()


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


def test_data_loading(temp_dir):
    """Test data loading functionality."""
    # Create test data
    data = np.random.randn(10, 100).astype(np.float32)
    data_file = Path(temp_dir) / "test.bin"

    # Save in Fortran format
    with open(data_file, "wb") as f:
        data.T.tofile(f)

    # Test loading
    loaded_data = load_data_file(data_file, 10, 100, dtype=np.float32)
    npt.assert_allclose(loaded_data, data)

    # Test error handling
    with pytest.raises(FileNotFoundError):
        load_data_file(Path(temp_dir) / "nonexistent.bin", 10, 100)


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
