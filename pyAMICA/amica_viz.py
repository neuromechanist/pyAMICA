"""
Visualization tools for analyzing AMICA results.

This module provides a comprehensive set of visualization functions for analyzing
and interpreting AMICA results. The visualizations include:

1. Convergence Analysis:
   - Log likelihood progression
   - Gradient norm evolution
   - Helps assess optimization quality and convergence

2. Component Analysis:
   - Mixing vectors (spatial patterns)
   - Activation distributions
   - Helps interpret discovered sources

3. Model Comparison:
   - Data reconstructions
   - Component sharing patterns
   - Helps evaluate model quality and differences

4. PDF Analysis:
   - Fitted probability density functions
   - Mixture component contributions
   - Helps understand source distributions

These visualizations are essential for:
- Validating model convergence
- Interpreting discovered components
- Comparing different models
- Understanding source characteristics
- Diagnosing potential issues
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Union, List, Tuple

from amica_data import load_results


def plot_convergence(
    results_dir: Union[str, Path],
    figsize: Tuple[int, int] = (10, 5),
    compressed: bool = False
) -> None:
    """
    Plot convergence metrics (likelihood and gradient norm).

    Parameters
    ----------
    results_dir : str or Path
        Directory containing results
    figsize : tuple
        Figure size (width, height)
    compressed : bool
        Whether results are compressed
    """
    results = load_results(results_dir, compressed)

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Plot log likelihood
    ax = axes[0]
    ax.plot(results['ll'], 'b-')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log Likelihood')
    ax.set_title('Convergence: Log Likelihood')
    ax.grid(True)

    # Plot gradient norm if available
    if 'nd' in results:
        ax = axes[1]
        ax.plot(results['nd'], 'r-')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Gradient Norm')
        ax.set_title('Convergence: Gradient Norm')
        ax.grid(True)

    plt.tight_layout()


def plot_components(
    results_dir: Union[str, Path],
    data: Optional[np.ndarray] = None,
    max_comps: int = 20,
    figsize: Optional[Tuple[int, int]] = None,
    compressed: bool = False
) -> None:
    """
    Plot learned components and their activations.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing results
    data : ndarray, optional
        Data array to compute activations
    max_comps : int
        Maximum number of components to plot
    figsize : tuple
        Figure size (width, height)
    compressed : bool
        Whether results are compressed
    """
    results = load_results(results_dir, compressed)

    # Get number of components to plot
    n_comps = min(results['A'].shape[1], max_comps)

    if figsize is None:
        figsize = (12, 2 * n_comps)

    fig, axes = plt.subplots(n_comps, 2, figsize=figsize)

    # Plot mixing vectors and activations
    for i in range(n_comps):
        # Plot mixing vector
        ax = axes[i, 0]
        ax.plot(results['A'][:, i], 'b-')
        ax.set_title(f'Component {i+1}: Mixing Vector')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Weight')
        ax.grid(True)

        # Plot activation if data provided
        if data is not None:
            ax = axes[i, 1]
            activation = np.dot(results['W'][i, :, 0], data)
            ax.hist(activation, bins=50, density=True)
            ax.set_title(f'Component {i+1}: Activation Distribution')
            ax.set_xlabel('Activation')
            ax.set_ylabel('Density')
            ax.grid(True)

    plt.tight_layout()


def plot_model_comparison(
    results_dir: Union[str, Path],
    data: np.ndarray,
    n_examples: int = 5,
    figsize: Optional[Tuple[int, int]] = None,
    compressed: bool = False
) -> None:
    """
    Plot data reconstructions from different models.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing results
    data : ndarray
        Data array to reconstruct
    n_examples : int
        Number of example time points
    figsize : tuple
        Figure size (width, height)
    compressed : bool
        Whether results are compressed
    """
    results = load_results(results_dir, compressed)

    # Get reconstructions for each model
    n_models = results['W'].shape[2]
    n_channels = data.shape[0]

    if figsize is None:
        figsize = (12, 3 * n_examples)

    fig, axes = plt.subplots(n_examples, n_models + 1, figsize=figsize)

    # Randomly select time points
    rng = np.random.RandomState(42)
    times = rng.choice(data.shape[1], n_examples, replace=False)

    for i, t in enumerate(times):
        # Plot original data
        ax = axes[i, 0]
        ax.plot(data[:, t], 'k-')
        ax.set_title('Original' if i == 0 else '')
        ax.set_xlabel('Channel')
        ax.set_ylabel('Value')
        ax.grid(True)

        # Plot reconstructions
        for h in range(n_models):
            ax = axes[i, h + 1]

            # Get reconstruction
            S = np.dot(results['W'][:, :, h], data[:, t:t+1])
            X_hat = np.dot(results['A'][:, results['comp_list'][:, h]], S)

            ax.plot(X_hat, 'r-')
            ax.plot(data[:, t], 'k--', alpha=0.5)
            ax.set_title(f'Model {h+1}' if i == 0 else '')
            ax.set_xlabel('Channel')
            ax.grid(True)

    plt.tight_layout()


def plot_component_sharing(
    results_dir: Union[str, Path],
    compressed: bool = False
) -> None:
    """
    Plot component sharing matrix.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing results
    compressed : bool
        Whether results are compressed
    """
    results = load_results(results_dir, compressed)

    # Create sharing matrix
    n_models = results['comp_list'].shape[1]
    n_comps = results['A'].shape[1]
    sharing = np.zeros((n_comps, n_models))

    for h in range(n_models):
        sharing[results['comp_list'][:, h], h] = 1

    plt.figure(figsize=(8, 8))
    plt.imshow(sharing, cmap='binary', aspect='auto')
    plt.colorbar()
    plt.xlabel('Model')
    plt.ylabel('Component')
    plt.title('Component Sharing Matrix')
    plt.grid(True)
    plt.tight_layout()


def plot_pdf_fits(
    results_dir: Union[str, Path],
    data: np.ndarray,
    max_comps: int = 10,
    n_points: int = 1000,
    figsize: Optional[Tuple[int, int]] = None,
    compressed: bool = False
) -> None:
    """
    Plot PDF fits for each component.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing results
    data : ndarray
        Data array
    max_comps : int
        Maximum number of components to plot
    n_points : int
        Number of points for PDF evaluation
    figsize : tuple
        Figure size (width, height)
    compressed : bool
        Whether results are compressed
    """
    from amica_pdf import compute_pdf
    results = load_results(results_dir, compressed)

    # Get number of components to plot
    n_comps = min(results['A'].shape[1], max_comps)
    n_mix = results['alpha'].shape[0]

    if figsize is None:
        figsize = (12, 3 * n_comps)

    fig, axes = plt.subplots(n_comps, 1, figsize=figsize)
    if n_comps == 1:
        axes = [axes]

    # Get activations
    S = np.dot(results['W'][:, :, 0], data)

    for i in range(n_comps):
        ax = axes[i]

        # Plot activation histogram
        ax.hist(S[i, :], bins=50, density=True, alpha=0.5, label='Data')

        # Plot fitted PDFs
        x = np.linspace(S[i, :].min(), S[i, :].max(), n_points)
        pdf_total = np.zeros_like(x)

        for j in range(n_mix):
            # Get mixture parameters
            alpha = results['alpha'][j, i]
            mu = results['mu'][j, i]
            beta = results['beta'][j, i]
            rho = results['rho'][j, i]

            # Compute PDF
            y = beta * (x - mu)
            pdf, _ = compute_pdf(y, rho)
            pdf = alpha * beta * pdf

            ax.plot(x, pdf, '--', alpha=0.5,
                   label=f'Mix {j+1} (α={alpha:.2f}, ρ={rho:.2f})')
            pdf_total += pdf

        ax.plot(x, pdf_total, 'r-', label='Total')
        ax.set_title(f'Component {i+1}: PDF Fit')
        ax.set_xlabel('Activation')
        ax.set_ylabel('Density')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()


def create_report(
    results_dir: Union[str, Path],
    data: Optional[np.ndarray] = None,
    output_file: Optional[str] = None,
    compressed: bool = False
) -> None:
    """
    Create comprehensive visualization report combining all analysis plots.

    This function generates a complete analysis report that includes:
    1. Convergence plots showing optimization progress
    2. Component visualizations showing learned sources
    3. Component sharing matrix showing model relationships
    4. Model comparison showing reconstruction quality
    5. PDF fits showing learned source distributions

    The report can be displayed interactively or saved to a file.

    Parameters
    ----------
    results_dir : str or Path
        Directory containing results
    data : ndarray, optional
        Data array
    output_file : str, optional
        Output file path (if None, show plots)
    compressed : bool
        Whether results are compressed
    """
    if output_file is not None:
        import matplotlib
        matplotlib.use('Agg')

    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot convergence
    plt.subplot(2, 2, 1)
    plot_convergence(results_dir, compressed=compressed)

    # Plot components
    plt.subplot(2, 2, 2)
    plot_components(results_dir, data, compressed=compressed)

    # Plot component sharing
    plt.subplot(2, 2, 3)
    plot_component_sharing(results_dir, compressed=compressed)

    if data is not None:
        # Plot model comparison
        plt.subplot(2, 2, 4)
        plot_model_comparison(results_dir, data, compressed=compressed)

        # Plot PDF fits
        plt.figure(figsize=(15, 10))
        plot_pdf_fits(results_dir, data, compressed=compressed)

    if output_file is not None:
        plt.savefig(output_file)
    else:
        plt.show()
