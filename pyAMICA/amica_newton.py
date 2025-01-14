"""
Newton optimization methods for AMICA.

This module implements Newton's method for optimizing the AMICA model parameters.
The Newton approach uses second-order information (the Hessian matrix) to improve
convergence compared to standard gradient descent.

Key components:
1. Newton direction computation using natural gradient and approximate Hessian
2. Parameter estimation for the Newton update
3. Unmixing matrix updates using the computed direction

The implementation follows the approach described in:
Palmer, J. A., Kreutz-Delgado, K., & Makeig, S. (2012).
AMICA: An adaptive mixture of independent component analyzers with shared components.
"""

import numpy as np
from scipy import linalg


def compute_newton_direction(
    dA: np.ndarray,
    sigma2: np.ndarray,
    lambda_: np.ndarray,
    kappa: np.ndarray,
    h: int
) -> np.ndarray:
    """
    Compute Newton direction for unmixing matrix update.

    This implements the Newton direction computation from the Fortran AMICA code,
    using the natural gradient preconditioned by the approximate Hessian.

    Parameters
    ----------
    dA : ndarray
        Natural gradient
    sigma2 : ndarray
        Second moment parameters
    lambda_ : ndarray
        Lambda parameters
    kappa : ndarray
        Kappa parameters
    h : int
        Model index

    Returns
    -------
    H : ndarray
        Newton direction
    """
    data_dim = dA.shape[0]
    H = np.zeros_like(dA)

    # Compute Newton direction
    for i in range(data_dim):
        for j in range(data_dim):
            if i == j:
                # Diagonal elements
                if lambda_[i, h] > 0:
                    H[i, i] = dA[i, i] / lambda_[i, h]
            else:
                # Off-diagonal elements
                sk1 = sigma2[i, h] * kappa[j, h]
                sk2 = sigma2[j, h] * kappa[i, h]
                if sk1 * sk2 > 1.0:
                    H[i, j] = (sk1 * dA[i, j] - dA[j, i]) / (sk1 * sk2 - 1.0)

    return H


def compute_newton_parameters(
    b: np.ndarray,
    z: np.ndarray,
    v: np.ndarray,
    y: np.ndarray,
    dpdf: np.ndarray,
    h: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute parameters needed for Newton optimization.

    This function estimates three key parameters:
    1. sigma2: Second moments of the activations
    2. lambda_: Diagonal elements of the approximate Hessian
    3. kappa: Off-diagonal elements of the approximate Hessian

    These parameters are used to construct the preconditioner for the
    natural gradient, which approximates the inverse Hessian of the
    log-likelihood with respect to the unmixing matrix.

    Parameters
    ----------
    b : ndarray
        Activations
    z : ndarray
        Responsibilities
    v : ndarray
        Model probabilities
    y : ndarray
        Scaled activations
    dpdf : ndarray
        PDF derivatives
    h : int
        Model index

    Returns
    -------
    sigma2 : ndarray
        Second moment parameters
    lambda_ : ndarray
        Lambda parameters
    kappa : ndarray
        Kappa parameters
    """
    batch_size = b.shape[0]
    data_dim = b.shape[1]

    # Initialize parameters
    sigma2 = np.zeros(data_dim)
    lambda_ = np.zeros(data_dim)
    kappa = np.zeros(data_dim)

    # Compute parameters
    for i in range(data_dim):
        # Second moments
        sigma2[i] = np.sum(v[:, h] * b[:, i]**2) / np.sum(v[:, h])

        # Lambda and kappa parameters
        for j in range(z.shape[2]):  # num_mix
            lambda_[i] += np.sum(
                v[:, h] * z[:, i, j, h] * (dpdf[:, i, j, h] * y[:, i, j, h] - 1)**2
            )
            kappa[i] += np.sum(
                v[:, h] * z[:, i, j, h] * dpdf[:, i, j, h]**2
            )

        lambda_[i] /= np.sum(v[:, h])
        kappa[i] /= np.sum(v[:, h])

    return sigma2, lambda_, kappa


def update_unmixing_matrix(
    A: np.ndarray,
    dA: np.ndarray,
    sigma2: np.ndarray,
    lambda_: np.ndarray,
    kappa: np.ndarray,
    comp_list: np.ndarray,
    h: int,
    lrate: float
) -> np.ndarray:
    """
    Update unmixing matrix using Newton direction.

    Parameters
    ----------
    A : ndarray
        Current mixing matrix
    dA : ndarray
        Natural gradient
    sigma2 : ndarray
        Second moment parameters
    lambda_ : ndarray
        Lambda parameters
    kappa : ndarray
        Kappa parameters
    comp_list : ndarray
        Component assignments
    h : int
        Model index
    lrate : float
        Learning rate

    Returns
    -------
    A : ndarray
        Updated mixing matrix
    """
    # Compute Newton direction
    H = compute_newton_direction(dA, sigma2, lambda_, kappa, h)

    # Update mixing matrix
    A[:, comp_list[:, h]] += lrate * np.dot(A[:, comp_list[:, h]], H)

    return A
