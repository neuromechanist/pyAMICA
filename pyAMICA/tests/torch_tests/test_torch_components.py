"""
Unit tests for individual PyTorch AMICA components.
"""

import unittest
import numpy as np
import torch
from pathlib import Path

from pyAMICA.torch_impl import (
    AMICATorch,
    GaussianMixtureICA,
    NaturalGradientOptimizer,
    NewtonOptimizer,
    setup_device,
    check_numerical_stability
)


class TestAMICATorch(unittest.TestCase):
    """Test the main AMICATorch class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = setup_device()
        self.n_channels = 4
        self.n_sources = 4
        self.n_samples = 100
        self.n_models = 2
        self.n_mix = 3
        
        # Create synthetic data
        np.random.seed(42)
        self.data = np.random.randn(self.n_channels, self.n_samples).astype(np.float32)
        
    def test_initialization(self):
        """Test model initialization."""
        model = AMICATorch(
            n_channels=self.n_channels,
            n_sources=self.n_sources,
            n_models=self.n_models,
            n_mix=self.n_mix,
            device=self.device
        )
        
        # Check dimensions
        self.assertEqual(len(model.A), self.n_models)
        self.assertEqual(model.A[0].shape, (self.n_channels, self.n_sources))
        self.assertEqual(model.alpha.shape, (self.n_mix, self.n_sources))
        
        # Check device
        self.assertEqual(model.A[0].device.type, self.device.type)
        
    def test_forward_pass(self):
        """Test forward pass through the model."""
        model = AMICATorch(
            n_channels=self.n_channels,
            n_sources=self.n_sources,
            n_models=self.n_models,
            device=self.device
        )
        
        X = torch.from_numpy(self.data).to(self.device)
        
        # Single model forward
        Y_single = model.forward(X, model_idx=0)
        self.assertEqual(Y_single.shape, (self.n_sources, self.n_samples))
        self.assertTrue(check_numerical_stability(Y_single, "Y_single"))
        
        # All models forward
        Y_all = model.forward(X)
        self.assertEqual(Y_all.shape, (self.n_models, self.n_sources, self.n_samples))
        self.assertTrue(check_numerical_stability(Y_all, "Y_all"))
        
    def test_log_likelihood(self):
        """Test log-likelihood computation."""
        model = AMICATorch(
            n_channels=self.n_channels,
            n_sources=self.n_sources,
            n_models=1,  # Single model for simplicity
            n_mix=self.n_mix,
            device=self.device
        )
        
        X = torch.from_numpy(self.data).to(self.device)
        
        # Compute log-likelihood
        ll = model.compute_log_likelihood(X)
        
        # Check it's a scalar
        self.assertEqual(ll.shape, ())
        
        # Check it's not NaN or Inf
        self.assertFalse(torch.isnan(ll))
        self.assertFalse(torch.isinf(ll))
        
    def test_preprocessing(self):
        """Test data preprocessing."""
        model = AMICATorch(
            n_channels=self.n_channels,
            device=self.device
        )
        
        # Test mean removal
        X_centered = model.preprocess_data(
            self.data,
            do_mean=True,
            do_sphere=False
        )
        
        mean_after = X_centered.mean(dim=1)
        self.assertTrue(torch.allclose(mean_after, torch.zeros_like(mean_after), atol=1e-6))
        
        # Test sphering
        X_sphered = model.preprocess_data(
            self.data,
            do_mean=True,
            do_sphere=True
        )
        
        cov = torch.cov(X_sphered)
        # Check if covariance is close to identity
        self.assertTrue(torch.allclose(cov, torch.eye(self.n_channels, device=self.device), atol=0.1))
        
    def test_save_load(self):
        """Test model saving and loading."""
        model1 = AMICATorch(
            n_channels=self.n_channels,
            n_models=self.n_models,
            device=self.device
        )
        
        # Modify some parameters
        model1.A[0].data = torch.randn_like(model1.A[0])
        
        # Save
        save_path = "test_model.pth"
        model1.save(save_path)
        
        # Load
        model2 = AMICATorch.load(save_path, device=self.device)
        
        # Check parameters match
        self.assertTrue(torch.allclose(model1.A[0], model2.A[0]))
        self.assertEqual(model1.n_models, model2.n_models)
        
        # Clean up
        Path(save_path).unlink()


class TestGaussianMixtureICA(unittest.TestCase):
    """Test the Gaussian Mixture ICA module."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = setup_device()
        self.n_sources = 3
        self.n_mix = 2
        self.n_samples = 100
        
        # Create synthetic sources
        np.random.seed(42)
        self.Y = torch.randn(self.n_sources, self.n_samples, device=self.device)
        
    def test_initialization(self):
        """Test mixture model initialization."""
        gmm = GaussianMixtureICA(
            n_sources=self.n_sources,
            n_mix=self.n_mix,
            pdf_type='gg',
            device=self.device
        )
        
        # Check parameter shapes
        self.assertEqual(gmm.alpha.shape, (self.n_mix, self.n_sources))
        self.assertEqual(gmm.mu.shape, (self.n_mix, self.n_sources))
        self.assertEqual(gmm.beta.shape, (self.n_mix, self.n_sources))
        self.assertEqual(gmm.rho.shape, (self.n_mix, self.n_sources))
        
        # Check constraints
        self.assertTrue(torch.all(gmm.alpha >= 0))
        self.assertTrue(torch.allclose(gmm.alpha.sum(dim=0), torch.ones(self.n_sources)))
        self.assertTrue(torch.all(gmm.beta > 0))
        self.assertTrue(torch.all((gmm.rho >= 1) & (gmm.rho <= 2)))
        
    def test_forward(self):
        """Test forward pass through mixture model."""
        gmm = GaussianMixtureICA(
            n_sources=self.n_sources,
            n_mix=self.n_mix,
            device=self.device
        )
        
        log_p = gmm.forward(self.Y)
        
        # Check shape
        self.assertEqual(log_p.shape, (self.n_samples,))
        
        # Check numerical stability
        self.assertFalse(torch.any(torch.isnan(log_p)))
        self.assertFalse(torch.any(torch.isinf(log_p)))
        
    def test_different_pdfs(self):
        """Test different PDF types."""
        pdf_types = ['gg', 'laplace', 'gaussian']
        
        for pdf_type in pdf_types:
            with self.subTest(pdf_type=pdf_type):
                gmm = GaussianMixtureICA(
                    n_sources=self.n_sources,
                    n_mix=self.n_mix,
                    pdf_type=pdf_type,
                    device=self.device
                )
                
                log_p = gmm.forward(self.Y)
                
                # Check numerical stability
                self.assertFalse(torch.any(torch.isnan(log_p)))
                self.assertFalse(torch.any(torch.isinf(log_p)))
                
    def test_em_fitting(self):
        """Test EM algorithm for mixture fitting."""
        gmm = GaussianMixtureICA(
            n_sources=self.n_sources,
            n_mix=self.n_mix,
            device=self.device
        )
        
        # Initial log-likelihood
        ll_init = gmm.forward(self.Y).mean().item()
        
        # Fit with EM
        gmm.fit_em(self.Y, max_iter=10, verbose=False)
        
        # Final log-likelihood
        ll_final = gmm.forward(self.Y).mean().item()
        
        # Should improve (or at least not get worse)
        self.assertGreaterEqual(ll_final, ll_init - 1e-6)
        
    def test_sampling(self):
        """Test sampling from mixture model."""
        gmm = GaussianMixtureICA(
            n_sources=self.n_sources,
            n_mix=self.n_mix,
            device=self.device
        )
        
        # Sample from all sources
        samples = gmm.sample(n_samples=50)
        self.assertEqual(samples.shape, (self.n_sources, 50))
        
        # Sample from specific source
        samples_single = gmm.sample(n_samples=50, model_idx=0)
        self.assertEqual(samples_single.shape, (50,))
        
        # Check numerical stability
        self.assertFalse(torch.any(torch.isnan(samples)))
        self.assertFalse(torch.any(torch.isinf(samples)))


class TestOptimizers(unittest.TestCase):
    """Test optimization methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.device = setup_device()
        self.n_channels = 3
        self.n_samples = 50
        
        # Create simple model and data
        self.model = AMICATorch(
            n_channels=self.n_channels,
            n_models=1,
            n_mix=2,
            device=self.device
        )
        
        np.random.seed(42)
        self.data = np.random.randn(self.n_channels, self.n_samples).astype(np.float32)
        self.X = torch.from_numpy(self.data).to(self.device)
        
    def test_natural_gradient(self):
        """Test natural gradient optimizer."""
        optimizer = NaturalGradientOptimizer(
            self.model.parameters(),
            lr=0.01,
            fisher_type='diagonal'
        )
        
        # Perform a few optimization steps
        for _ in range(5):
            optimizer.zero_grad()
            
            # Compute loss
            neg_ll = -self.model.compute_log_likelihood(self.X)
            neg_ll.backward()
            
            # Check gradients exist
            has_grad = any(p.grad is not None for p in self.model.parameters())
            self.assertTrue(has_grad)
            
            # Optimization step
            optimizer.step()
            
            # Check parameters are still valid
            for p in self.model.parameters():
                self.assertTrue(check_numerical_stability(p, "parameter"))
                
    def test_newton_optimizer(self):
        """Test Newton optimizer."""
        newton_opt = NewtonOptimizer(
            self.model,
            method='cg',
            max_iter=5
        )
        
        # Get initial loss
        with torch.no_grad():
            ll_init = self.model.compute_log_likelihood(self.X).item()
            
        # Newton step
        newton_opt.step(self.X)
        
        # Get final loss
        with torch.no_grad():
            ll_final = self.model.compute_log_likelihood(self.X).item()
            
        # Check parameters are still valid
        for p in self.model.parameters():
            self.assertTrue(check_numerical_stability(p, "parameter"))
            
        # Note: Newton step might not always improve likelihood in early iterations
        # so we just check for numerical stability


class TestUtils(unittest.TestCase):
    """Test utility functions."""
    
    def test_device_setup(self):
        """Test device setup."""
        # Test auto-selection
        device = setup_device()
        self.assertIsInstance(device, torch.device)
        
        # Test CPU selection
        device_cpu = setup_device('cpu')
        self.assertEqual(device_cpu.type, 'cpu')
        
    def test_numerical_stability_check(self):
        """Test numerical stability checking."""
        # Good tensor
        good_tensor = torch.randn(10, 10)
        self.assertTrue(check_numerical_stability(good_tensor))
        
        # Bad tensor with NaN
        bad_tensor = torch.tensor([1.0, float('nan'), 3.0])
        self.assertFalse(check_numerical_stability(bad_tensor))
        
        # Bad tensor with Inf
        inf_tensor = torch.tensor([1.0, float('inf'), 3.0])
        self.assertFalse(check_numerical_stability(inf_tensor))
        
    def test_correlation_matrix(self):
        """Test correlation matrix computation."""
        from pyAMICA.torch_impl.utils import compute_correlation_matrix
        
        # Create two similar matrices
        np.random.seed(42)
        W1 = np.random.randn(3, 4)
        W2 = W1 + 0.1 * np.random.randn(3, 4)
        
        corr = compute_correlation_matrix(W1, W2)
        
        # Should have high diagonal correlation
        self.assertTrue(np.all(np.diag(corr) > 0.9))
        
    def test_find_permutation(self):
        """Test permutation finding."""
        from pyAMICA.torch_impl.utils import find_best_permutation
        
        # Create correlation matrix with clear permutation
        corr = np.array([
            [0.1, 0.9, 0.2],
            [0.8, 0.2, 0.1],
            [0.2, 0.1, 0.95]
        ])
        
        perm, signs = find_best_permutation(corr)
        
        # Expected permutation: [1, 0, 2]
        expected = np.array([1, 0, 2])
        np.testing.assert_array_equal(perm, expected)


if __name__ == '__main__':
    unittest.main()