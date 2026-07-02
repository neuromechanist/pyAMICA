"""
Utility functions for PyTorch AMICA.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


def setup_device(preferred: Optional[str] = None) -> torch.device:
    """
    Set up the best available device for computation.

    Parameters
    ----------
    preferred : str, optional
        Preferred device ('cpu', 'cuda', 'mps'). If None or unavailable,
        automatically selects the best available device.

    Returns
    -------
    device : torch.device
        Selected device for computation
    """
    if preferred:
        if preferred == "mps" and torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        elif preferred == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU (requested device unavailable)")
    else:
        # Auto-select best device
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name()}")
        else:
            device = torch.device("cpu")
            logger.info("Using CPU")

    return device


def check_numerical_stability(
    tensor: torch.Tensor, name: str = "tensor", raise_on_error: bool = False
) -> bool:
    """
    Check tensor for numerical issues (NaN, Inf).

    Parameters
    ----------
    tensor : torch.Tensor
        Tensor to check
    name : str, default="tensor"
        Name for logging
    raise_on_error : bool, default=False
        If True, raise exception on numerical issues

    Returns
    -------
    is_stable : bool
        True if tensor has no numerical issues
    """
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()

    if has_nan or has_inf:
        msg = f"Numerical instability in {name}: "
        if has_nan:
            msg += "contains NaN "
        if has_inf:
            msg += "contains Inf"

        if raise_on_error:
            raise ValueError(msg)
        else:
            logger.warning(msg)
            return False

    return True


def stabilize_log_probabilities(
    log_probs: torch.Tensor, min_log: float = -1500.0
) -> torch.Tensor:
    """
    Stabilize log probabilities to prevent underflow.

    Parameters
    ----------
    log_probs : torch.Tensor
        Log probabilities to stabilize
    min_log : float, default=-1500.0
        Minimum allowed log value

    Returns
    -------
    log_probs_stable : torch.Tensor
        Stabilized log probabilities
    """
    return torch.clamp(log_probs, min=min_log)


def compute_correlation_matrix(
    W1: np.ndarray, W2: np.ndarray, absolute: bool = True
) -> np.ndarray:
    """
    Compute correlation between two sets of components.

    Used for comparing unmixing matrices from different implementations.

    Parameters
    ----------
    W1 : np.ndarray
        First unmixing matrix of shape (n_components, n_channels)
    W2 : np.ndarray
        Second unmixing matrix of shape (n_components, n_channels)
    absolute : bool, default=True
        If True, use absolute correlation (ignores sign ambiguity)

    Returns
    -------
    corr_matrix : np.ndarray
        Correlation matrix of shape (n_comp_1, n_comp_2)
    """
    n1, _ = W1.shape
    n2, _ = W2.shape

    corr_matrix = np.zeros((n1, n2))

    for i in range(n1):
        for j in range(n2):
            corr = np.corrcoef(W1[i], W2[j])[0, 1]
            if absolute:
                corr = abs(corr)
            corr_matrix[i, j] = corr

    return corr_matrix


def find_best_permutation(corr_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find best permutation to match components.

    Parameters
    ----------
    corr_matrix : np.ndarray
        Correlation matrix between components

    Returns
    -------
    permutation : np.ndarray
        Best permutation indices
    signs : np.ndarray
        Sign corrections (+1 or -1)
    """
    from scipy.optimize import linear_sum_assignment

    # Use Hungarian algorithm to find best matching
    row_ind, col_ind = linear_sum_assignment(-np.abs(corr_matrix))

    # Determine signs
    signs = np.sign(corr_matrix[row_ind, col_ind])
    signs[signs == 0] = 1

    return col_ind, signs


def compare_with_fortran(
    pytorch_results: Dict[str, Any], fortran_dir: str
) -> Dict[str, float]:
    """
    Compare PyTorch results with Fortran AMICA output.

    Parameters
    ----------
    pytorch_results : dict
        Dictionary containing PyTorch results (W, A, ll, etc.)
    fortran_dir : str
        Directory containing Fortran AMICA output files

    Returns
    -------
    metrics : dict
        Comparison metrics (correlations, errors, etc.)
    """
    from ..amica_load import loadmodout

    # Load Fortran results
    fortran_results = loadmodout(fortran_dir)

    metrics = {}

    # Compare unmixing matrices
    if "W" in pytorch_results and hasattr(fortran_results, "W"):
        W_pytorch = pytorch_results["W"]
        W_fortran = fortran_results.W[:, :, 0]  # First model

        # Compute correlation matrix
        corr_matrix = compute_correlation_matrix(W_pytorch, W_fortran)

        # Find best permutation
        perm, signs = find_best_permutation(corr_matrix)

        # Metrics
        metrics["W_mean_corr"] = np.mean(np.max(corr_matrix, axis=1))
        metrics["W_min_corr"] = np.min(np.max(corr_matrix, axis=1))

    # Compare log-likelihood
    if "ll" in pytorch_results and hasattr(fortran_results, "LL"):
        ll_pytorch = pytorch_results["ll"]
        ll_fortran = fortran_results.LL[-1] if len(fortran_results.LL) > 0 else 0

        metrics["ll_pytorch"] = ll_pytorch
        metrics["ll_fortran"] = ll_fortran
        metrics["ll_diff"] = abs(ll_pytorch - ll_fortran)

    # Compare mixing matrices
    if "A" in pytorch_results and hasattr(fortran_results, "A"):
        A_pytorch = pytorch_results["A"]
        A_fortran = fortran_results.A[:, :, 0]  # First model

        # Frobenius norm of difference
        A_diff = np.linalg.norm(A_pytorch - A_fortran, "fro")
        metrics["A_frob_diff"] = A_diff

    return metrics


def load_eeglab_data(
    filepath: str, data_dim: int, field_dim: int, dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Load EEGLAB .fdt binary data file.

    Parameters
    ----------
    filepath : str
        Path to .fdt file
    data_dim : int
        Number of channels
    field_dim : int
        Number of time points
    dtype : np.dtype, default=np.float32
        Data type

    Returns
    -------
    data : np.ndarray
        Loaded data of shape (data_dim, field_dim)
    """
    with open(filepath, "rb") as f:
        # Read all data
        data = np.fromfile(f, dtype=dtype)

        # EEGLAB stores a (data_dim, field_dim) array in column-major
        # (Fortran) order, i.e. channel-fastest within each sample. Reshape
        # directly to (data_dim, field_dim) with order='F'; no transpose is
        # needed (transposing after a (field_dim, data_dim) reshape reads
        # the wrong elements for any non-square data_dim != field_dim).
        data = data.reshape((data_dim, field_dim), order="F")

    return data


def save_pytorch_results(model, output_dir: str, save_history: bool = True):
    """
    Save PyTorch model results in format compatible with comparison.

    Parameters
    ----------
    model : AMICATorch
        Trained model
    output_dir : str
        Output directory
    save_history : bool, default=True
        Save optimization history
    """
    from pathlib import Path

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get results
    results = {}

    # Unmixing and mixing matrices
    for h in range(model.n_models):
        W = model.get_unmixing_matrix(h)
        A = model.get_mixing_matrix(h)

        if h == 0:
            results["W"] = W
            results["A"] = A

        # Save individual model matrices
        np.save(output_dir / f"W_{h}.npy", W)
        np.save(output_dir / f"A_{h}.npy", A)

    # Mixture parameters
    results["alpha"] = model.alpha.detach().cpu().numpy()
    results["mu"] = model.mu.detach().cpu().numpy()
    results["beta"] = model.beta.detach().cpu().numpy()
    results["rho"] = model.rho.detach().cpu().numpy()

    np.save(output_dir / "alpha.npy", results["alpha"])
    np.save(output_dir / "mu.npy", results["mu"])
    np.save(output_dir / "beta.npy", results["beta"])
    np.save(output_dir / "rho.npy", results["rho"])

    # Model weights
    results["gm"] = model.gm.detach().cpu().numpy()
    np.save(output_dir / "gm.npy", results["gm"])

    # Preprocessing
    results["mean"] = model.mean.detach().cpu().numpy()
    results["sphere"] = model.sphere.detach().cpu().numpy()
    np.save(output_dir / "mean.npy", results["mean"])
    np.save(output_dir / "sphere.npy", results["sphere"])

    # History
    if save_history and hasattr(model, "ll_history"):
        results["ll"] = model.ll_history[-1] if model.ll_history else 0
        np.save(output_dir / "ll_history.npy", model.ll_history)

        # Write text summary
        with open(output_dir / "summary.txt", "w") as f:
            f.write("PyTorch AMICA Results\n")
            f.write("=" * 50 + "\n")
            f.write(f"Number of models: {model.n_models}\n")
            f.write(f"Number of sources: {model.n_sources}\n")
            f.write(f"Number of mixture components: {model.n_mix}\n")
            f.write(f"Final log-likelihood: {results['ll']:.6f}\n")
            f.write(f"Device: {model.device}\n")

    # Save full model
    torch.save(model.state_dict(), output_dir / "model_state.pth")
    model.save(str(output_dir / "model_full.pth"))

    logger.info(f"Results saved to {output_dir}")

    return results
