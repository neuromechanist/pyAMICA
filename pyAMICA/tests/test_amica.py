"""Tests for AMICA implementation."""

import numpy as np
from pathlib import Path
import unittest

from pyAMICA import AMICA
from pyAMICA.amica_data import load_data_file, preprocess_data
from pyAMICA.amica_pdf import compute_pdf
from pyAMICA.amica_newton import compute_newton_direction


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
        
        # Save test data in Fortran format
        cls.test_dir = Path('test_data')
        if not cls.test_dir.exists():
            cls.test_dir.mkdir()
            
        data_file = cls.test_dir / 'test.bin'
        with open(data_file, 'wb') as f:
            cls.data.T.astype(np.float32).tofile(f)
            
    def test_data_loading(self):
        """Test data loading matches Fortran."""
        data = load_data_file(
            self.test_dir / 'test.bin',
            self.data_dim,
            self.num_samples,
            1,
            dtype=np.float32
        )
        np.testing.assert_allclose(data, self.data)
        
    def test_preprocessing(self):
        """Test preprocessing matches Fortran."""
        # Test mean removal
        data, mean, _ = preprocess_data(
            self.data.copy(),
            do_mean=True,
            do_sphere=False
        )
        np.testing.assert_allclose(
            mean.ravel(),
            self.data.mean(axis=1)
        )
        np.testing.assert_allclose(
            data.mean(axis=1),
            np.zeros(self.data_dim),
            atol=1e-10
        )
        
        # Test sphering
        data, _, sphere = preprocess_data(
            self.data.copy(),
            do_mean=False,
            do_sphere=True
        )
        cov = np.cov(data)
        np.testing.assert_allclose(
            cov,
            np.eye(self.data_dim),
            atol=1e-10
        )
        
    def test_pdf_computation(self):
        """Test PDF computation matches Fortran."""
        # Test Laplace distribution
        y = np.linspace(-5, 5, 100)
        pdf, dpdf = compute_pdf(y, rho=1.0)
        np.testing.assert_allclose(
            pdf,
            np.exp(-np.abs(y)) / 2.0
        )
        np.testing.assert_allclose(
            dpdf,
            -np.sign(y) * pdf
        )
        
        # Test Gaussian distribution
        pdf, dpdf = compute_pdf(y, rho=2.0)
        np.testing.assert_allclose(
            pdf,
            np.exp(-y*y) / np.sqrt(np.pi)
        )
        np.testing.assert_allclose(
            dpdf,
            -2 * y * pdf
        )
        
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
            dA, sigma2[:, None], lambda_[:, None],
            kappa[:, None], 0
        )
        
        # Test diagonal elements
        for i in range(data_dim):
            self.assertAlmostEqual(
                H[i, i],
                dA[i, i] / lambda_[i]
            )
            
        # Test off-diagonal elements
        for i in range(data_dim):
            for j in range(data_dim):
                if i != j:
                    sk1 = sigma2[i] * kappa[j]
                    sk2 = sigma2[j] * kappa[i]
                    if sk1 * sk2 > 1.0:
                        self.assertAlmostEqual(
                            H[i, j],
                            (sk1 * dA[i, j] - dA[j, i]) / (sk1 * sk2 - 1.0)
                        )
                    else:
                        self.assertAlmostEqual(H[i, j], 0.0)
                        
    def test_full_amica(self):
        """Test full AMICA optimization."""
        # Create simple test case
        rng = np.random.RandomState(42)
        n_sources = 4
        n_samples = 1000
        
        # Create independent sources
        S = rng.laplace(size=(n_sources, n_samples))
        
        # Create random mixing matrix
        A_true = rng.randn(n_sources, n_sources)
        
        # Mix sources
        X = np.dot(A_true, S)
        
        # Run AMICA
        model = AMICA(
            num_models=1,
            num_mix=3,
            max_iter=100,
            do_newton=True,
            lrate=0.1,
            seed=42
        )
        model.fit(X)
        
        # Get unmixed sources
        S_est = model.transform(X)
        
        # Compute correlation with true sources
        corr = np.corrcoef(S.ravel(), S_est[:,:,0].ravel())[0,1]
        self.assertGreater(np.abs(corr), 0.9)
        
    @classmethod
    def tearDownClass(cls):
        """Clean up test files."""
        import shutil
        shutil.rmtree(cls.test_dir)


if __name__ == '__main__':
    unittest.main()
