"""
Data loading and preprocessing for AMICA.

This module provides functionality for loading and preprocessing data for the AMICA
algorithm. It handles:

1. Loading binary data files in Fortran format
2. Data preprocessing operations:
   - Mean removal
   - Sphering (whitening) using PCA
   - Dimensionality reduction
3. Result saving and loading with optional compression

The preprocessing steps are crucial for AMICA's performance:
- Mean removal centers the data, removing DC offset
- Sphering decorrelates the data and normalizes variance, which can speed up
  convergence and improve separation quality
- PCA-based dimensionality reduction can help remove noise and reduce
  computational complexity
"""

import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional


def load_data_file(
    filepath: str,
    data_dim: int,
    field_dim: int,
    dtype: np.dtype = np.float64
) -> np.ndarray:
    """
    Load data from binary file in Fortran format.

    Parameters
    ----------
    filepath : str
        Path to binary data file
    data_dim : int
        Number of channels/dimensions
    field_dim : int
        Number of samples per field/channel
    dtype : np.dtype
        Data type (default: float64)

    Returns
    -------
    data : ndarray
        Loaded data array of shape (data_dim, field_dim)
    """
    # Read entire file at once
    with open(filepath, 'rb') as f:
        # Read all data and reshape considering Fortran order
        data = np.fromfile(f, dtype=dtype)
        # Reshape to (field_dim, data_dim) and transpose to get (data_dim, field_dim)
        # Using Fortran order 'F' since data is stored in column-major format
        data = data.reshape((field_dim, data_dim), order='F').T

    return data


def load_multiple_files(
    filepaths: List[str],
    data_dim: int,
    field_dims: List[int],
    dtype: np.dtype = np.float32
) -> np.ndarray:
    """
    Load and concatenate data from multiple binary files.

    Parameters
    ----------
    filepaths : list of str
        Paths to binary data files
    data_dim : int
        Number of channels/dimensions
    field_dims : list of int
        Number of samples per field for each file
    dtype : np.dtype
        Data type (default: float64)

    Returns
    -------
    data : ndarray
        Concatenated data array of shape (data_dim, total_samples)
    """
    # Calculate total samples
    total_samples = sum(field_dims)

    # Allocate output array
    data = np.zeros((data_dim, total_samples), dtype=dtype)

    # Load each file
    idx = 0
    for filepath, field_dim in zip(filepaths, field_dims):
        file_data = load_data_file(filepath, data_dim, field_dim, dtype)
        data[:, idx:idx + field_dim] = file_data
        idx += field_dim

    return data


def preprocess_data(
    data: np.ndarray,
    do_mean: bool = True,
    do_sphere: bool = True,
    do_approx_sphere: bool = True,
    pcakeep: Optional[int] = None,
    pcadb: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data by removing mean and applying sphering transformation.

    This function performs two main preprocessing steps:
    1. Mean removal: Subtracts the temporal mean from each channel
    2. Sphering: Applies a linear transformation to decorrelate the signals
       and normalize their variance. This can be done either exactly or
       approximately:
       - Exact: W = Λ^(-1/2) E^T where Λ, E are eigenvalues/vectors of covariance
       - Approximate: W = Λ^(-1/2) E^T (faster but less precise)

    Optionally reduces dimensionality by keeping only the top components based on
    either a fixed number (pcakeep) or an energy threshold (pcadb).

    Parameters
    ----------
    data : ndarray
        Input data array of shape (channels, samples)
    do_mean : bool
        Whether to remove mean
    do_sphere : bool
        Whether to perform sphering
    do_approx_sphere : bool
        Whether to use approximate sphering
    pcakeep : int, optional
        Number of components to keep
    pcadb : float, optional
        dB threshold for keeping components

    Returns
    -------
    data : ndarray
        Preprocessed data
    mean : ndarray
        Data mean (or zeros if do_mean=False)
    sphere : ndarray
        Sphering matrix (or identity if do_sphere=False)
    """
    data_dim = data.shape[0]

    # Remove mean if requested
    if do_mean:
        mean = np.mean(data, axis=1, keepdims=True)
        data = data - mean
    else:
        mean = np.zeros((data_dim, 1))

    # Compute sphering matrix if requested
    if do_sphere:
        # Compute covariance
        cov = np.cov(data)

        # Eigenvalue decomposition
        evals, evecs = np.linalg.eigh(cov)

        # Sort in descending order
        idx = np.argsort(evals)[::-1]
        evals = evals[idx]
        evecs = evecs[:, idx]

        # Determine number of components to keep
        if pcakeep is not None:
            n_comp = min(pcakeep, len(evals))
        elif pcadb is not None:
            db = 10 * np.log10(evals / evals[0])
            n_comp = np.sum(db > -pcadb)
        else:
            n_comp = len(evals)

        # Create sphering matrix
        if do_approx_sphere:
            # Approximate sphering (faster but less accurate)
            sphere = np.dot(
                np.diag(1.0 / np.sqrt(evals[:n_comp])),
                evecs[:, :n_comp].T)
        else:
            # Exact sphering
            sphere = np.linalg.inv(
                np.dot(np.diag(np.sqrt(evals[:n_comp])),
                       evecs[:, :n_comp].T))

        # Apply sphering
        data = np.dot(sphere, data)
    else:
        sphere = np.eye(data_dim)

    return data, mean, sphere


def save_results(
    outdir: str,
    A: np.ndarray,
    W: np.ndarray,
    c: np.ndarray,
    mu: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    rho: np.ndarray,
    gm: np.ndarray,
    mean: np.ndarray,
    sphere: np.ndarray,
    comp_list: np.ndarray,
    ll: List[float],
    nd: Optional[List[float]] = None,
    compress: bool = False
):
    """
    Save AMICA results to disk.

    Parameters
    ----------
    outdir : str
        Output directory
    A : ndarray
        Mixing matrix
    W : ndarray
        Unmixing matrices
    c : ndarray
        Bias terms
    mu : ndarray
        Component means
    alpha : ndarray
        Mixture weights
    beta : ndarray
        Scale parameters
    rho : ndarray
        Shape parameters
    gm : ndarray
        Model weights
    mean : ndarray
        Data mean
    sphere : ndarray
        Sphering matrix
    comp_list : ndarray
        Component assignments
    ll : list
        Log likelihood history
    nd : list, optional
        Gradient norm history
    compress : bool
        Whether to compress saved files
    """
    outdir = Path(outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    # Save parameters
    params = {
        'A': A,
        'W': W,
        'c': c,
        'mu': mu,
        'alpha': alpha,
        'beta': beta,
        'rho': rho,
        'gm': gm,
        'mean': mean,
        'sphere': sphere,
        'comp_list': comp_list
    }

    for name, param in params.items():
        if compress:
            np.savez_compressed(outdir / f"{name}.npz", param)
        else:
            np.save(outdir / f"{name}.npy", param)

    # Save optimization history
    history = {
        'll': np.array(ll)
    }
    if nd is not None:
        history['nd'] = np.array(nd)

    for name, hist in history.items():
        if compress:
            np.savez_compressed(outdir / f"{name}.npz", hist)
        else:
            np.save(outdir / f"{name}.npy", hist)


def load_results(
    indir: str,
    compressed: bool = False
) -> dict:
    """
    Load saved AMICA results from disk.

    Parameters
    ----------
    indir : str
        Input directory containing saved results
    compressed : bool
        Whether files are compressed

    Returns
    -------
    results : dict
        Dictionary of loaded parameters and history
    """
    indir = Path(indir)
    results = {}

    # Parameter names to load
    param_names = [
        'A', 'W', 'c', 'mu', 'alpha', 'beta', 'rho', 'gm',
        'mean', 'sphere', 'comp_list', 'll', 'nd'
    ]

    # Load each parameter
    for name in param_names:
        filepath = indir / f"{name}.{'npz' if compressed else 'npy'}"
        if filepath.exists():
            if compressed:
                results[name] = np.load(filepath)['arr_0']
            else:
                results[name] = np.load(filepath)

    return results
