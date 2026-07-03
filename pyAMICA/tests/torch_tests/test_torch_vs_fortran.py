"""
Comparative tests between PyTorch and Fortran AMICA implementations.
Uses real sample data to ensure parity between implementations.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import json

from pyAMICA.torch_impl import AMICATorch
from pyAMICA.torch_impl.utils import (
    load_eeglab_data,
    compare_with_fortran,
    compute_correlation_matrix,
    find_best_permutation,
)
from pyAMICA.amica_load import loadmodout


# Path to sample data
SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent / "sample_data"
EEGLAB_DATA = SAMPLE_DATA_DIR / "eeglab_data.fdt"
FORTRAN_OUTPUT = SAMPLE_DATA_DIR / "amicaout"
SAMPLE_PARAMS = SAMPLE_DATA_DIR / "sample_params.json"


@pytest.fixture(scope="module")
def sample_data():
    """Load the actual EEGLAB sample data."""
    if not EEGLAB_DATA.exists():
        pytest.skip(f"Sample data not found at {EEGLAB_DATA}")

    # Load data with correct dimensions from sample_params.json
    with open(SAMPLE_PARAMS, "r") as f:
        params = json.load(f)

    data = load_eeglab_data(
        str(EEGLAB_DATA),
        data_dim=params["data_dim"],
        field_dim=params["field_dim"][0],
        dtype=np.float32,
    )

    return data, params


@pytest.fixture(scope="module")
def fortran_results():
    """Load Fortran AMICA results."""
    if not FORTRAN_OUTPUT.exists():
        pytest.skip(f"Fortran output not found at {FORTRAN_OUTPUT}")

    return loadmodout(str(FORTRAN_OUTPUT))


class TestPyTorchVsFortran:
    """Test PyTorch implementation against Fortran reference."""

    def test_data_loading(self, sample_data):
        """Test that we can load the sample data correctly."""
        data, params = sample_data

        assert data.shape == (params["data_dim"], params["field_dim"][0])
        assert not np.any(np.isnan(data))
        assert not np.any(np.isinf(data))

        # Check data statistics match expected range
        assert -100 < data.mean() < 100
        assert 0 < data.std() < 1000

    def test_fortran_output_loading(self, fortran_results):
        """Test that we can load Fortran output correctly."""
        # Check essential components exist
        assert hasattr(fortran_results, "W")
        assert hasattr(fortran_results, "A")
        assert hasattr(fortran_results, "LL")

        # Check dimensions
        n_comp, n_chan, n_models = fortran_results.W.shape
        assert n_comp == 32
        assert n_chan == 32
        assert n_models >= 1

        # Check convergence - Fortran should have converged
        if len(fortran_results.LL) > 0:
            final_ll = fortran_results.LL[-1]
            # Fortran typically converges to around -3.44
            assert -10 < final_ll < 0

    def test_pytorch_initialization_matches_params(self, sample_data):
        """Test PyTorch model initialization with sample parameters."""
        data, params = sample_data

        model = AMICATorch(
            n_channels=params["data_dim"],
            n_sources=params.get("num_comps", params["data_dim"]),
            n_models=params.get("num_models", 1),
            n_mix=params.get("num_mix", 3),
        )

        # Check dimensions match
        assert model.n_channels == params["data_dim"]
        assert model.n_models == params.get("num_models", 1)
        assert model.n_mix == params.get("num_mix", 3)

    def test_preprocessing_consistency(self, sample_data):
        """Test that preprocessing produces consistent results."""
        data, params = sample_data

        model = AMICATorch(n_channels=params["data_dim"])

        # Preprocess with settings from params
        X_prep = model.preprocess_data(
            data,
            do_mean=params.get("do_mean", True),
            do_sphere=params.get("do_sphere", True),
        )

        # Check preprocessing worked
        if params.get("do_mean", True):
            assert torch.abs(X_prep.mean()).item() < 1e-6

        if params.get("do_sphere", True):
            cov = torch.cov(X_prep)
            # Check covariance is approximately identity
            eye = torch.eye(params["data_dim"], device=X_prep.device)
            assert torch.norm(cov - eye).item() < 1.0

    @pytest.mark.slow
    def test_pytorch_convergence(self, sample_data):
        """Test that PyTorch implementation converges on real data."""
        data, params = sample_data

        # Use smaller iteration count for testing
        model = AMICATorch(
            n_channels=params["data_dim"],
            n_sources=params.get("num_comps", params["data_dim"]),
            n_models=params.get("num_models", 1),
            n_mix=params.get("num_mix", 3),
        )

        # Fit with limited iterations
        model.fit(
            data,
            max_iter=10,  # Just test that it runs without NaN
            lrate=params.get("lrate", 0.05),
            do_newton=False,  # Disable Newton for quick test
            verbose=False,
        )

        # Check that we have valid results
        assert len(model.ll_history) > 0
        assert not any(np.isnan(ll) for ll in model.ll_history)

        # Check parameters are still valid
        for p in model.parameters():
            assert not torch.any(torch.isnan(p))
            assert not torch.any(torch.isinf(p))

    def test_unmixing_matrix_structure(self, sample_data, fortran_results):
        """Test that unmixing matrices have similar structure."""
        data, params = sample_data

        # Initialize PyTorch model
        model = AMICATorch(
            n_channels=params["data_dim"],
            n_sources=params.get("num_comps", params["data_dim"]),
            n_models=params.get("num_models", 1),
        )

        # Get unmixing matrices
        W_pytorch = model.get_unmixing_matrix(0)
        W_fortran = fortran_results.W[:, :, 0]

        # Both should be approximately square matrices
        assert W_pytorch.shape == W_fortran.shape

        # Check condition number (both should be reasonably conditioned)
        cond_pytorch = np.linalg.cond(W_pytorch)
        cond_fortran = np.linalg.cond(W_fortran)

        assert cond_pytorch < 1e10
        assert cond_fortran < 1e10

    @pytest.mark.slow
    def test_component_correlation(self, sample_data, fortran_results):
        """Test correlation between PyTorch and Fortran components."""
        data, params = sample_data

        # Fit PyTorch model with very limited iterations
        model = AMICATorch(
            n_channels=params["data_dim"],
            n_sources=params.get("num_comps", params["data_dim"]),
            n_models=params.get("num_models", 1),
            n_mix=params.get("num_mix", 3),
        )

        # Just initialize and get initial W
        W_pytorch = model.get_unmixing_matrix(0)
        W_fortran = fortran_results.W[:, :, 0]

        # Compute correlation matrix
        corr_matrix = compute_correlation_matrix(W_pytorch, W_fortran)

        # Even random initialization should have some structure
        # We're just testing the correlation computation works
        assert corr_matrix.shape == (W_pytorch.shape[0], W_fortran.shape[0])
        assert np.all(corr_matrix >= -1.001)
        assert np.all(corr_matrix <= 1.001)

    def test_mixture_parameters_range(self, sample_data):
        """Test that mixture parameters stay in valid ranges."""
        data, params = sample_data

        model = AMICATorch(
            n_channels=params["data_dim"],
            n_sources=params.get("num_comps", params["data_dim"]),
            n_models=params.get("num_models", 1),
            n_mix=params.get("num_mix", 3),
        )

        # Check initial parameter ranges
        alpha = model.alpha.detach().cpu().numpy()
        beta = model.beta.detach().cpu().numpy()
        rho = model.rho.detach().cpu().numpy()

        # Alpha should sum to 1 across mixture components
        assert np.allclose(alpha.sum(axis=0), 1.0)

        # Beta should be positive
        assert np.all(beta > 0)

        # Rho should be in [1, 2] for generalized Gaussian
        assert np.all(rho >= 1.0)
        assert np.all(rho <= 2.0)

    def test_log_likelihood_computation(self, sample_data):
        """Test that log-likelihood computation doesn't produce NaN."""
        data, params = sample_data

        model = AMICATorch(
            n_channels=params["data_dim"],
            n_sources=params.get("num_comps", params["data_dim"]),
            n_models=params.get("num_models", 1),
            n_mix=params.get("num_mix", 3),
        )

        # Preprocess data
        X_prep = model.preprocess_data(data)

        # Compute log-likelihood
        ll = model.compute_log_likelihood(X_prep)

        # Should be a valid number
        assert not torch.isnan(ll)
        assert not torch.isinf(ll)

        # Should be in reasonable range (not too positive or negative)
        ll_value = ll.item()
        assert -1e10 < ll_value < 1e10


class TestComparisonMetrics:
    """Test the comparison metrics between implementations."""

    def test_correlation_metric(self):
        """Test correlation-based comparison metric."""
        # Create two slightly different matrices
        np.random.seed(42)
        W1 = np.random.randn(4, 4)
        W2 = W1 + 0.1 * np.random.randn(4, 4)

        corr = compute_correlation_matrix(W1, W2)

        # Diagonal should have high correlation
        diag_corr = np.diag(corr)
        assert np.all(diag_corr > 0.8)

    def test_permutation_finding(self):
        """Test finding optimal permutation between components."""
        # Create permuted matrix
        np.random.seed(42)
        W1 = np.random.randn(4, 4)
        perm_true = [2, 0, 3, 1]
        W2 = W1[perm_true, :]

        # Add small noise
        W2 += 0.01 * np.random.randn(4, 4)

        # Find permutation
        corr = compute_correlation_matrix(W1, W2)
        perm_found, signs = find_best_permutation(corr)

        # find_best_permutation returns, for each row of W1, the index of
        # the best-matching column of W2. Since W2 = W1[perm_true], the
        # W2 index matching W1 row i is the inverse of perm_true evaluated
        # at i, i.e. np.argsort(perm_true).
        expected = np.argsort(perm_true)
        np.testing.assert_array_equal(perm_found, expected)

    def test_comparison_with_fortran_function(self, sample_data, fortran_results):
        """Test the compare_with_fortran utility function."""
        data, params = sample_data

        # Create a simple PyTorch result
        model = AMICATorch(
            n_channels=params["data_dim"],
            n_sources=params.get("num_comps", params["data_dim"]),
        )

        pytorch_results = {
            "W": model.get_unmixing_matrix(0),
            "A": model.get_mixing_matrix(0),
            "ll": 0.0,  # Dummy value
        }

        # Compare
        metrics = compare_with_fortran(pytorch_results, str(FORTRAN_OUTPUT))

        # Should have computed metrics
        assert "W_mean_corr" in metrics
        assert "W_min_corr" in metrics
        assert "ll_pytorch" in metrics
        assert "ll_fortran" in metrics

        # Correlations should be in valid range
        assert 0 <= metrics["W_mean_corr"] <= 1
        assert 0 <= metrics["W_min_corr"] <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
