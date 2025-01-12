"""Utility functions for AMICA implementation."""

import numpy as np
from scipy import special


def gammaln(x):
    """Compute the log of the gamma function."""
    return special.gammaln(x)


def psifun(x):
    """
    Compute the digamma function (derivative of log gamma).
    
    This is a Python implementation of the Fortran psifun function.
    """
    return special.digamma(x)


def determine_block_size(data, min_size, max_size, step_size, num_threads=1):
    """
    Determine optimal block size for data processing.
    
    Parameters
    ----------
    data : ndarray
        Input data array
    min_size : int
        Minimum block size to try
    max_size : int
        Maximum block size to try
    step_size : int
        Step size between block sizes to try
    num_threads : int
        Number of threads to use
        
    Returns
    -------
    optimal_size : int
        Optimal block size
    """
    import time
    
    block_times = []
    block_sizes = range(min_size, max_size + 1, step_size)
    
    # Test each block size
    for block_size in block_sizes:
        start_time = time.time()
        
        # Process one block
        for start in range(0, data.shape[1], block_size):
            end = min(start + block_size, data.shape[1])
            X = data[:, start:end]
            
            # Do some representative computation
            _ = np.dot(X.T, X)
            
        block_times.append(time.time() - start_time)
        
    # Return block size with minimum processing time
    return block_sizes[np.argmin(block_times)]


def identify_shared_components(A, W, comp_list, comp_thresh=0.99):
    """
    Identify components that are shared between models.
    
    Parameters
    ----------
    A : ndarray
        Mixing matrix
    W : ndarray
        Unmixing matrices
    comp_list : ndarray
        Component assignments
    comp_thresh : float
        Correlation threshold for identifying shared components
        
    Returns
    -------
    comp_list : ndarray
        Updated component assignments
    comp_used : ndarray
        Boolean mask of used components
    """
    num_models = W.shape[2]
    num_comps = A.shape[1]
    data_dim = A.shape[0]
    
    comp_used = np.ones(num_comps, dtype=bool)
    
    # Compare components between models
    for h1 in range(num_models):
        for h2 in range(h1+1, num_models):
            for i1 in range(data_dim):
                for i2 in range(data_dim):
                    k1 = comp_list[i1, h1]
                    k2 = comp_list[i2, h2]
                    
                    # Skip if components are already identified
                    if k1 == k2:
                        continue
                        
                    # Compute correlation
                    corr = np.abs(np.dot(A[:, k1], A[:, k2])) / (
                        np.sqrt(np.sum(A[:, k1]**2) * np.sum(A[:, k2]**2)))
                    
                    if corr >= comp_thresh:
                        # Check if components appear together in any model
                        shared = False
                        for h in range(num_models):
                            if (k1 in comp_list[:, h] and k2 in comp_list[:, h]):
                                shared = True
                                break
                                
                        if not shared:
                            # Merge components
                            comp_used[k2] = False
                            comp_list[comp_list == k2] = k1
                            
    return comp_list, comp_used


def get_unmixing_matrices(A, comp_list):
    """
    Compute unmixing matrices from mixing matrix and component assignments.
    
    Parameters
    ----------
    A : ndarray
        Mixing matrix
    comp_list : ndarray
        Component assignments
        
    Returns
    -------
    W : ndarray
        Unmixing matrices
    """
    data_dim = A.shape[0]
    num_models = comp_list.shape[1]
    
    W = np.zeros((data_dim, data_dim, num_models))
    
    for h in range(num_models):
        idx = comp_list[:, h]
        W[:, :, h] = np.linalg.inv(A[:, idx])
        
    return W


def reject_outliers(ll, rejsig):
    """
    Identify outlier samples based on log likelihood.
    
    Parameters
    ----------
    ll : ndarray
        Log likelihood values
    rejsig : float
        Number of standard deviations for rejection threshold
        
    Returns
    -------
    mask : ndarray
        Boolean mask of non-outlier samples
    """
    ll_mean = np.mean(ll)
    ll_std = np.std(ll)
    return ll >= (ll_mean - rejsig * ll_std)
