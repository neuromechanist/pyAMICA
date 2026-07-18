"""Tests for AMICA implementation."""

import shutil
import tempfile
import numpy as np
from pathlib import Path
import unittest

from pamica.numpy_impl.data import load_data_file, preprocess_data
from pamica.numpy_impl.pdf import compute_pdf
from pamica.numpy_impl.newton import compute_newton_direction


class TestAMICA(unittest.TestCase):
    """Test AMICA implementation against Fortran code."""

    @classmethod
    def setUpClass(cls):
        """Create test data."""
        # Create random test data
        rng = np.random.RandomState(42)
        cls.data_dim = 64
        cls.num_samples = 1000
        cls.data = rng.randn(cls.data_dim, cls.num_samples)

        # Save test data in Fortran format. Use a unique temp dir (not a shared
        # relative "test_data/") so parallel workers (pytest-xdist -n auto) do
        # not race on creating/removing the same path.
        cls.test_dir = Path(tempfile.mkdtemp(prefix="pyamica_test_amica_"))

        data_file = cls.test_dir / "test.bin"
        with open(data_file, "wb") as f:
            cls.data.T.astype(np.float32).tofile(f)

    def test_data_loading(self):
        """Test data loading matches Fortran."""
        data = load_data_file(
            self.test_dir / "test.bin",
            self.data_dim,
            self.num_samples,
            dtype=np.float32,
        )
        np.testing.assert_allclose(data, self.data)

    def test_preprocessing(self):
        """Test preprocessing matches Fortran."""
        # Test mean removal
        data, mean, _ = preprocess_data(self.data.copy(), do_mean=True, do_sphere=False)
        np.testing.assert_allclose(mean.ravel(), self.data.mean(axis=1))
        np.testing.assert_allclose(
            data.mean(axis=1), np.zeros(self.data_dim), atol=1e-10
        )

        # Test sphering
        data, _, sphere = preprocess_data(
            self.data.copy(), do_mean=False, do_sphere=True
        )
        cov = np.cov(data)
        np.testing.assert_allclose(cov, np.eye(self.data_dim), atol=1e-10)

    def test_pdf_computation(self):
        """Test PDF computation matches Fortran."""
        # Test Laplace distribution
        y = np.linspace(-5, 5, 100)
        pdf, dpdf = compute_pdf(y, rho=1.0)
        np.testing.assert_allclose(pdf, np.exp(-np.abs(y)) / 2.0)
        np.testing.assert_allclose(dpdf, -np.sign(y) * pdf)

        # Test Gaussian distribution
        pdf, dpdf = compute_pdf(y, rho=2.0)
        np.testing.assert_allclose(pdf, np.exp(-y * y) / np.sqrt(np.pi))
        np.testing.assert_allclose(dpdf, -2 * y * pdf)

    def test_newton_direction(self):
        """Test Newton direction computation matches Fortran."""
        rng = np.random.RandomState(42)

        # Create test inputs
        data_dim = 10
        dA = rng.randn(data_dim, data_dim)
        sigma2 = np.abs(rng.randn(data_dim))
        lambda_ = np.abs(rng.randn(data_dim))
        kappa = np.abs(rng.randn(data_dim))

        # Compute Newton direction
        H = compute_newton_direction(
            dA, sigma2[:, None], lambda_[:, None], kappa[:, None], 0
        )

        # Test diagonal elements
        for i in range(data_dim):
            self.assertAlmostEqual(H[i, i], dA[i, i] / lambda_[i])

        # Test off-diagonal elements
        for i in range(data_dim):
            for j in range(data_dim):
                if i != j:
                    sk1 = sigma2[i] * kappa[j]
                    sk2 = sigma2[j] * kappa[i]
                    if sk1 * sk2 > 1.0:
                        self.assertAlmostEqual(
                            H[i, j], (sk1 * dA[i, j] - dA[j, i]) / (sk1 * sk2 - 1.0)
                        )
                    else:
                        self.assertAlmostEqual(H[i, j], 0.0)

    # NOTE: the former synthetic-data source-recovery test (test_full_amica) was
    # removed: it fabricated data (against the NO-MOCK policy) and used a broken
    # metric (corrcoef on raveled sources, no permutation/sign/scale matching, so
    # it could never pass). Real-data NumPy-vs-Fortran parity is covered by
    # tests/test_sample_data.py::test_sample_data_numpy_vs_fortran.

    @classmethod
    def tearDownClass(cls):
        """Clean up test files (ignore_errors: the temp dir may already be gone)."""
        shutil.rmtree(cls.test_dir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
