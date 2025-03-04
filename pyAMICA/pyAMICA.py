"""
AMICA (Adaptive Mixture ICA) Implementation
=========================================

This module implements the Adaptive Mixture Independent Component Analysis (AMICA)
algorithm, which performs blind source separation using a mixture of adaptive
independent component analyzers.

Key Features
-----------
* Multiple Source Models: Can learn different mixing models for different parts of the data
* Flexible PDFs: Supports various source distributions including Gaussian, Laplace, and mixtures
* Component Sharing: Automatically identifies and shares similar components across models
* Outlier Rejection: Robust estimation by identifying and excluding outlier samples
* Optimization: Efficient parameter updates using natural gradient and Newton methods
* Preprocessing: Automatic mean removal and data sphering

Mathematical Background
--------------------
AMICA extends traditional ICA by:

1. Using mixture models for source PDFs:
   p(s) = Σ_k α_k p_k(s)
   where α_k are mixture weights and p_k are component PDFs

2. Learning multiple mixing models:
   x = A_m s + c_m
   where m indexes different models and c_m are bias terms

3. Optimizing model parameters via maximum likelihood:
   L = Σ_t log p(x_t)
   where p(x_t) includes all mixture components and models

Usage Example
------------
>>> import numpy as np
>>> from pyAMICA import AMICA
>>>
>>> # Generate random data
>>> X = np.random.randn(64, 1000)  # 64 channels, 1000 samples
>>>
>>> # Initialize and fit model
>>> model = AMICA(num_models=2)  # Use 2 mixing models
>>> model.fit(X)
>>>
>>> # Get separated sources
>>> S = model.transform(X)

The algorithm automatically:
- Removes data mean if requested
- Spheres the data if requested
- Initializes model parameters
- Optimizes parameters using natural gradient
- Switches to Newton optimization if requested
- Identifies shared components across models
- Rejects outliers if requested

See Also
--------
amica_pdf : PDF implementations
amica_utils : Utility functions
amica_viz : Visualization tools
amica_cli : Command-line interface

References
----------
1. Palmer, J. A., et al. "Newton Method for the ICA Mixture Model."
   ICASSP 2008.
2. Palmer, J. A., et al. "AMICA: An Adaptive Mixture of Independent
   Component Analyzers with Shared Components." 2012.
"""

import numpy as np
from scipy import linalg
import logging
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple
from tqdm import tqdm
from .amica_utils import (
    gammaln, determine_block_size, identify_shared_components,
    get_unmixing_matrices
)


def load_default_params(params_file: Optional[str] = None) -> Dict:
    """
    Load default parameters from JSON file.

    Parameters
    ----------
    params_file : str, optional
        Path to JSON parameter file. If None, uses default params.json

    Returns
    -------
    params : dict
        Dictionary of default parameters
    """
    if params_file is None:
        params_file = Path(__file__).parent / 'params.json'

    with open(params_file) as f:
        params = json.load(f)

    # Remove data-specific parameters
    data_params = {'files', 'num_samples', 'data_dim', 'field_dim'}
    return {k: v for k, v in params.items() if k not in data_params}


class AMICA:
    """
    Adaptive Mixture ICA (AMICA) implementation.

    This class implements the AMICA algorithm for blind source separation using
    adaptive mixtures of independent component analyzers.

    The algorithm provides two progress reporting modes:
    1. A modern tqdm progress bar (default) showing overall progress and key metrics
    2. Detailed per-line progress output in the style of the original Fortran implementation
       (enabled with verbose=True or use_tqdm=False)
    """

    def __init__(
        self,
        params_file: Optional[str] = None,
        use_tqdm: bool = True,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize AMICA with parameters.

        Parameters
        ----------
        params_file : str, optional
            Path to JSON parameter file with default values
        use_tqdm : bool, default=True
            Whether to use tqdm progress bar (False will use per-line printing)
        verbose : bool, default=False
            Whether to enable verbose output (will use per-line printing regardless of use_tqdm)
        **kwargs : dict
            Override default parameters with these values
        """
        # Store progress bar settings
        self.use_tqdm = use_tqdm
        self.verbose = verbose
        # Load default parameters
        params = load_default_params(params_file)

        # Override with any provided parameters
        params.update(kwargs)

        # Store parameters
        self.num_models = params.get('num_models', 1)
        self.num_mix = params.get('num_mix', 3)
        self.max_iter = params.get('max_iter', 2000)
        self.do_newton = params.get('do_newton', False)
        self.newt_start = params.get('newt_start', 20)
        self.newt_ramp = params.get('newt_ramp', 10)
        self.newtrate = params.get('newtrate', 0.5)
        self.do_reject = params.get('do_reject', False)
        self.rejsig = params.get('rejsig', 3.0)
        self.rejstart = params.get('rejstart', 2)
        self.rejint = params.get('rejint', 3)
        self.maxrej = params.get('maxrej', 1)
        self.num_comps = params.get('num_comps', -1)
        self.lrate = params.get('lrate', 0.1)
        self.lrate0 = self.lrate
        self.minlrate = params.get('minlrate', 1e-12)
        self.lratefact = params.get('lratefact', 0.5)
        self.rho0 = params.get('rho0', 1.5)
        self.minrho = params.get('minrho', 1.0)
        self.maxrho = params.get('maxrho', 2.0)
        self.rholrate = params.get('rholrate', 0.05)
        self.rholrate0 = self.rholrate
        self.rholratefact = params.get('rholratefact', 0.1)
        self.invsigmax = params.get('invsigmax', 1000.0)
        self.invsigmin = params.get('invsigmin', 1e-4)
        self.do_history = params.get('do_history', False)
        self.histstep = params.get('histstep', 10)
        self.do_opt_block = params.get('do_opt_block', True)
        self.block_size = params.get('block_size', 128)
        self.blk_min = params.get('blk_min', 128)
        self.blk_max = params.get('blk_max', 1024)
        self.blk_step = params.get('blk_step', 128)
        self.share_comps = params.get('share_comps', False)
        self.comp_thresh = params.get('comp_thresh', 0.99)
        self.share_start = params.get('share_start', 100)
        self.share_int = params.get('share_int', 100)
        self.doscaling = params.get('doscaling', True)
        self.scalestep = params.get('scalestep', 1)
        self.do_sphere = params.get('do_sphere', True)
        self.do_mean = params.get('do_mean', True)
        self.do_approx_sphere = params.get('do_approx_sphere', True)
        self.pcakeep = params.get('pcakeep')
        self.pcadb = params.get('pcadb')
        self.writestep = params.get('writestep', 100)
        self.max_decs = params.get('max_decs', 5)
        self.min_dll = params.get('min_dll', 1e-9)
        self.min_grad_norm = params.get('min_grad_norm', 1e-7)
        self.use_min_dll = params.get('use_min_dll', True)
        self.use_grad_norm = params.get('use_grad_norm', True)
        self.pdftype = params.get('pdftype', 1)
        self.outdir = Path(params.get('outdir', 'output'))

        # Initialize random state
        self.rng = np.random.RandomState(params.get('seed'))

        # Initialize model parameters
        self.A = None  # Mixing matrix
        self.W = None  # Unmixing matrix
        self.c = None  # Bias terms
        self.mu = None  # Means of mixture components
        self.alpha = None  # Mixture weights
        self.beta = None  # Scale parameters
        self.rho = None  # Shape parameters
        self.gm = None  # Model weights

        # Initialize data parameters
        self.data_dim = None
        self.num_samples = None
        self.mean = None
        self.sphere = None
        self.comp_list = None
        self.comp_used = None

        # Initialize optimization state
        self.iter = 0
        self.ll = []  # Log likelihood history
        self.nd = []  # Gradient norm history

        # Initialize Newton optimization parameters
        if self.do_newton:
            self.sigma2 = None
            self.lambda_ = None
            self.kappa = None
            self.baralpha = None

        # Setup logging
        self._setup_logging()

    def _setup_logging(self):
        """Setup logging configuration."""
        self.logger = logging.getLogger("AMICA")
        
        # Add console handler for stdout
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # Add file handler for out.txt
        self.outdir = Path(self.outdir)
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)
        file_handler = logging.FileHandler(self.outdir / 'out.txt', mode='w')
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Prevent propagation to avoid duplicate logging
        self.logger.propagate = False

    def fit(self, data: np.ndarray) -> "AMICA":
        """
        Fit the AMICA model to the data.

        Parameters
        ----------
        data : ndarray of shape (n_channels, n_samples)
            The input data to fit the model to.

        Returns
        -------
        self : AMICA
            The fitted model.
        """
        self.logger.info("Starting AMICA fitting...")

        # Initialize dimensions
        self.data_dim = data.shape[0]
        self.num_samples = data.shape[1]

        if self.num_comps == -1:
            self.num_comps = self.data_dim * self.num_models

        # Preprocess data
        self._preprocess_data(data)

        # Initialize parameters
        self._initialize_parameters()

        # Optimize block size if requested
        if self.do_opt_block:
            self.block_size = determine_block_size(
                self.data, self.blk_min, self.blk_max, self.blk_step)
            self.logger.info(f"Optimal block size: {self.block_size}")

        # Main optimization loop
        self._optimize()

        return self

    def _preprocess_data(self, data: np.ndarray):
        """Preprocess the data by removing mean and sphering."""
        # Remove mean if requested
        if self.do_mean:
            self.mean = np.mean(data, axis=1, keepdims=True)
            data = data - self.mean
        else:
            self.mean = np.zeros((self.data_dim, 1))

        # Compute sphering matrix if requested
        if self.do_sphere:
            # Compute covariance
            cov = np.cov(data)

            # Eigenvalue decomposition
            evals, evecs = linalg.eigh(cov)

            # Sort in descending order
            idx = np.argsort(evals)[::-1]
            evals = evals[idx]
            evecs = evecs[:, idx]

            # Determine number of components to keep
            if self.pcakeep is not None:
                n_comp = min(self.pcakeep, len(evals))
            elif self.pcadb is not None:
                db = 10 * np.log10(evals / evals[0])
                n_comp = np.sum(db > -self.pcadb)
            else:
                n_comp = len(evals)

            # Create sphering matrix
            if self.do_approx_sphere:
                # Approximate sphering (faster but less accurate)
                self.sphere = np.dot(
                    np.diag(1.0 / np.sqrt(evals[:n_comp])),
                    evecs[:, :n_comp].T)
            else:
                # Exact sphering
                self.sphere = linalg.inv(
                    np.dot(np.diag(np.sqrt(evals[:n_comp])),
                           evecs[:, :n_comp].T))

            # Apply sphering
            data = np.dot(self.sphere, data)
        else:
            self.sphere = np.eye(self.data_dim)

        self.data = data

    def _initialize_parameters(self):
        """Initialize all model parameters."""
        # Initialize mixing/unmixing matrices
        if self.A is None:
            self.A = np.zeros((self.data_dim, self.num_comps))
            for h in range(self.num_models):
                if not hasattr(self, 'fix_init') or not self.fix_init:
                    self.A[:, h * self.data_dim:(h + 1) * self.data_dim] = (
                        np.eye(self.data_dim) + 0.01 * (
                            0.5 - self.rng.rand(self.data_dim, self.data_dim))
                    )
                else:
                    self.A[:, h * self.data_dim:(h + 1) * self.data_dim] = np.eye(self.data_dim)

        # Initialize component assignments
        self.comp_list = np.zeros((self.data_dim, self.num_models), dtype=int)
        self.comp_used = np.ones(self.num_comps, dtype=bool)
        for h in range(self.num_models):
            self.comp_list[:, h] = np.arange(h * self.data_dim, (h + 1) * self.data_dim)

        # Initialize mixture parameters
        if self.mu is None:
            self.mu = np.zeros((self.num_mix, self.num_comps))
            for k in range(self.num_comps):
                self.mu[:, k] = np.linspace(-1, 1, self.num_mix)
                if not hasattr(self, 'fix_init') or not self.fix_init:
                    self.mu[:, k] += 0.05 * (1 - 2 * self.rng.rand(self.num_mix))

        if self.alpha is None:
            self.alpha = np.ones((self.num_mix, self.num_comps)) / self.num_mix

        if self.beta is None:
            self.beta = np.ones((self.num_mix, self.num_comps))
            if not hasattr(self, 'fix_init') or not self.fix_init:
                self.beta += 0.1 * (0.5 - self.rng.rand(self.num_mix, self.num_comps))

        if self.rho is None:
            self.rho = self.rho0 * np.ones((self.num_mix, self.num_comps))

        if self.gm is None:
            self.gm = np.ones(self.num_models) / self.num_models

        # Initialize bias terms
        if self.c is None:
            self.c = np.zeros((self.data_dim, self.num_models))

        # Initialize Newton optimization parameters
        if self.do_newton:
            self.sigma2 = np.ones((self.data_dim, self.num_models))
            self.lambda_ = np.zeros((self.data_dim, self.num_models))
            self.kappa = np.zeros((self.data_dim, self.num_models))
            self.baralpha = np.zeros((self.num_mix, self.data_dim, self.num_models))

        # Get initial unmixing matrices
        self._update_unmixing_matrices()

    def _update_unmixing_matrices(self):
        """Update unmixing matrices from mixing matrix."""
        self.W = get_unmixing_matrices(self.A, self.comp_list)

    def _compute_log_pdf(self, y: np.ndarray, rho: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute log PDF value and its derivative for given activation.

        Parameters
        ----------
        y : ndarray
            Activation values
        rho : float
            Shape parameter

        Returns
        -------
        log_pdf : ndarray
            Log PDF values
        dpdf : ndarray
            PDF derivatives (not in log space)
        """
        if rho == 1.0:
            # Laplace distribution
            log_pdf = -np.abs(y) - np.log(2.0)
            pdf = np.exp(log_pdf)
            dpdf = -np.sign(y) * pdf
        elif rho == 2.0:
            # Gaussian distribution
            log_pdf = -y * y - 0.5 * np.log(np.pi)
            pdf = np.exp(log_pdf)
            dpdf = -2 * y * pdf
        else:
            # Generalized Gaussian distribution
            log_pdf = -np.power(np.abs(y), rho) - np.log(2.0) - gammaln(1.0 + 1.0 / rho)
            pdf = np.exp(log_pdf)
            dpdf = -rho * np.power(np.abs(y), rho - 1) * np.sign(y) * pdf

        return log_pdf, dpdf

    def _get_updates_and_likelihood(self) -> Dict:
        """
        Compute parameter updates and data likelihood.

        Returns
        -------
        updates : dict
            Dictionary containing parameter updates and likelihood
        """
        # Initialize update accumulators
        updates = {
            'dgm': np.zeros(self.num_models),
            'dalpha': np.zeros((self.num_mix, self.num_comps)),
            'dmu': np.zeros((self.num_mix, self.num_comps)),
            'dbeta': np.zeros((self.num_mix, self.num_comps)),
            'drho': np.zeros((self.num_mix, self.num_comps)),
            'dA': np.zeros((self.data_dim, self.data_dim, self.num_models)),
            'dc': np.zeros((self.data_dim, self.num_models)),
            'll': 0.0
        }

        if self.do_newton:
            updates.update({
                'dsigma2': np.zeros((self.data_dim, self.num_models)),
                'dlambda': np.zeros((self.data_dim, self.num_models)),
                'dkappa': np.zeros((self.data_dim, self.num_models)),
                'dbaralpha': np.zeros((self.num_mix, self.data_dim, self.num_models))
            })

        # Process data in blocks
        for start in range(0, self.data.shape[1], self.block_size):
            end = min(start + self.block_size, self.data.shape[1])
            X = self.data[:, start:end]

            # Get block updates
            block_updates = self._get_block_updates(X)

            # Accumulate updates
            for key in updates:
                updates[key] += block_updates[key]

        return updates

    def _get_block_updates(self, X: np.ndarray) -> Dict:
        """
        Compute parameter updates for a data block.

        Parameters
        ----------
        X : ndarray
            Data block to process

        Returns
        -------
        updates : dict
            Parameter updates for this block
        """
        batch_size = X.shape[1]
        updates = {
            'dgm': np.zeros(self.num_models),
            'dalpha': np.zeros((self.num_mix, self.num_comps)),
            'dmu': np.zeros((self.num_mix, self.num_comps)),
            'dbeta': np.zeros((self.num_mix, self.num_comps)),
            'drho': np.zeros((self.num_mix, self.num_comps)),
            'dA': np.zeros((self.data_dim, self.data_dim, self.num_models)),
            'dc': np.zeros((self.data_dim, self.num_models)),
            'll': 0.0
        }

        if self.do_newton:
            updates.update({
                'dsigma2': np.zeros((self.data_dim, self.num_models)),
                'dlambda': np.zeros((self.data_dim, self.num_models)),
                'dkappa': np.zeros((self.data_dim, self.num_models)),
                'dbaralpha': np.zeros((self.num_mix, self.data_dim, self.num_models))
            })

        # Compute activations for each model
        b = np.zeros((batch_size, self.data_dim, self.num_models))
        for h in range(self.num_models):
            b[:, :, h] = np.dot(X.T, self.W[:, :, h]) - self.c[:, h]

        # Compute mixture probabilities and responsibilities
        z = np.zeros((batch_size, self.data_dim, self.num_mix, self.num_models))
        for h in range(self.num_models):
            for i in range(self.data_dim):
                k = self.comp_list[i, h]
                for j in range(self.num_mix):
                    y = self.beta[j, k] * (b[:, i, h] - self.mu[j, k])

                    # Compute log PDF and its derivative
                    log_pdf, dpdf = self._compute_log_pdf(y, self.rho[j, k])

                    # Compute log probability directly in log space
                    z[:, i, j, h] = np.log(self.alpha[j, k]) + np.log(self.beta[j, k]) + log_pdf

        # Normalize responsibilities
        z = np.exp(z - np.max(z, axis=2, keepdims=True))
        z /= np.sum(z, axis=2, keepdims=True)

        # Compute model probabilities and log likelihood
        v = np.zeros((batch_size, self.num_models))
        ll = np.zeros(batch_size)
        for h in range(self.num_models):
            v[:, h] = np.log(self.gm[h])
            for i in range(self.data_dim):
                k = self.comp_list[i, h]
                # Sum log probabilities across mixture components
                ll_i = np.log(np.sum(np.exp(z[:, i, :, h]), axis=1))
                v[:, h] += ll_i
                ll += ll_i

        v = np.exp(v - np.max(v, axis=1, keepdims=True))
        v /= np.sum(v, axis=1, keepdims=True)

        # Accumulate parameter updates
        updates['ll'] = np.sum(ll)

        for h in range(self.num_models):
            # Model weights
            updates['dgm'][h] = np.sum(v[:, h])

            for i in range(self.data_dim):
                k = self.comp_list[i, h]

                # Mixture weights
                for j in range(self.num_mix):
                    updates['dalpha'][j, k] += np.sum(v[:, h] * z[:, i, j, h])

                    # Component means
                    y = self.beta[j, k] * (b[:, i, h] - self.mu[j, k])
                    log_pdf, dpdf = self._compute_log_pdf(y, self.rho[j, k])
                    updates['dmu'][j, k] += np.sum(v[:, h] * z[:, i, j, h] * dpdf)

                    # Scale parameters
                    updates['dbeta'][j, k] += np.sum(
                        v[:, h] * z[:, i, j, h] * y * dpdf)

                    # Shape parameters
                    if self.rho[j, k] not in (1.0, 2.0):
                        logy = np.log(np.abs(y))
                        updates['drho'][j, k] += np.sum(
                            v[:, h] * z[:, i, j, h] * np.power(
                                np.abs(y), self.rho[j, k]) * logy)

                    if self.do_newton:
                        # Newton optimization parameters
                        updates['dbaralpha'][j, i, h] += np.sum(v[:, h] * z[:, i, j, h])
                        updates['dsigma2'][i, h] += np.sum(v[:, h] * b[:, i, h]**2)
                        updates['dlambda'][i, h] += np.sum(
                            v[:, h] * z[:, i, j, h] * (dpdf * y - 1)**2)
                        updates['dkappa'][i, h] += np.sum(
                            v[:, h] * z[:, i, j, h] * dpdf**2)

            # Unmixing matrices
            g = np.zeros((batch_size, self.data_dim))
            for i in range(self.data_dim):
                k = self.comp_list[i, h]
                for j in range(self.num_mix):
                    y = self.beta[j, k] * (b[:, i, h] - self.mu[j, k])
                    _, dpdf = self._compute_log_pdf(y, self.rho[j, k])
                    g[:, i] += self.beta[j, k] * z[:, i, j, h] * dpdf

            updates['dA'][:, :, h] += np.dot(X, v[:, h: h + 1] * g)

            # Bias terms
            updates['dc'][:, h] += np.sum(v[:, h: h + 1] * g, axis=0)

        return updates

    def _update_parameters(self, updates: Dict):
        """
        Update model parameters using computed updates.

        Parameters
        ----------
        updates : dict
            Dictionary containing parameter updates
        """
        # Update model weights
        if self.do_reject:
            self.gm = updates['dgm'] / self.num_good_samples
        else:
            self.gm = updates['dgm'] / self.num_samples

        # Update mixture weights
        self.alpha = updates['dalpha'] / np.sum(updates['dalpha'], axis=0)

        # Update component means
        dmu = updates['dmu'] / updates['dalpha']
        self.mu += self.lrate * dmu

        # Update scale parameters
        dbeta = updates['dbeta'] / updates['dalpha']
        self.beta *= np.sqrt(1 + self.lrate * dbeta)
        self.beta = np.clip(self.beta, self.invsigmin, self.invsigmax)

        # Update shape parameters
        if not np.all(self.rho == 1.0) and not np.all(self.rho == 2.0):
            drho = updates['drho'] / updates['dalpha']
            self.rho += self.rholrate * (1 - self.rho * drho)
            self.rho = np.clip(self.rho, self.minrho, self.maxrho)

        # Update unmixing matrices
        if self.do_newton and self.iter >= self.newt_start:
            # Update Newton parameters
            self.sigma2 = updates['dsigma2'] / updates['dgm'][:, None]
            self.lambda_ = updates['dlambda'] / updates['dgm'][:, None]
            self.kappa = updates['dkappa'] / updates['dgm'][:, None]
            self.baralpha = updates['dbaralpha'] / updates['dgm'][:, None, None]

            # Newton updates
            self.lrate = min(self.newtrate,
                             self.lrate + min(1.0 / self.newt_ramp, self.lrate))

            for h in range(self.num_models):
                dA = -updates['dA'][:, :, h] / updates['dgm'][h]
                dA[np.diag_indices_from(dA)] += 1

                # Compute Newton direction
                H = np.zeros_like(dA)
                for i in range(self.data_dim):
                    for j in range(self.data_dim):
                        if i == j:
                            H[i, i] = dA[i, i] / self.lambda_[i, h]
                        else:
                            sk1 = self.sigma2[i, h] * self.kappa[j, h]
                            sk2 = self.sigma2[j, h] * self.kappa[i, h]
                            if sk1 * sk2 > 1.0:
                                H[i, j] = (sk1 * dA[i, j] - dA[j, i]) / (sk1 * sk2 - 1.0)

                self.A[:, self.comp_list[:, h]] += self.lrate * np.dot(
                    self.A[:, self.comp_list[:, h]], H)
        else:
            # Natural gradient updates
            self.lrate = min(self.lrate0,
                             self.lrate + min(1.0 / self.newt_ramp, self.lrate))

            for h in range(self.num_models):
                dA = -updates['dA'][:, :, h] / updates['dgm'][h]
                dA[np.diag_indices_from(dA)] += 1

                self.A[:, self.comp_list[:, h]] += self.lrate * np.dot(
                    self.A[:, self.comp_list[:, h]], dA)

        # Update bias terms
        self.c += self.lrate * updates['dc'] / updates['dgm'][:, None]

        # Rescale parameters if requested
        if self.doscaling and self.iter % self.scalestep == 0:
            for k in range(self.num_comps):
                scale = np.sqrt(np.sum(self.A[:, k]**2))
                if scale > 0:
                    self.A[:, k] /= scale
                    self.mu[:, k] *= scale
                    self.beta[:, k] /= scale

        # Update unmixing matrices
        self._update_unmixing_matrices()

        # Store likelihood
        self.ll.append(updates['ll'])

        # Compute gradient norm
        if self.use_grad_norm:
            dA = np.zeros_like(self.A)
            for h in range(self.num_models):
                dA[:, self.comp_list[:, h]] += self.gm[h] * updates['dA'][:, :, h]
            self.nd.append(np.sqrt(np.sum(dA**2) / (self.data_dim * self.num_comps)))

    def _optimize(self):
        """Main optimization loop."""
        self.logger.info("Starting optimization...")

        # Initialize optimization variables
        numdecs = 0
        numincs = 0
        numrej = 0
        start_time = time.time()
        convergence_reason = None
        final_iter = 0

        # Determine whether to use tqdm or per-line printing
        use_tqdm_progress = self.use_tqdm and not self.verbose
        
        # Create iterator (with or without tqdm)
        if use_tqdm_progress:
            # Use minimal progress bar with dynamic width and ASCII characters for better compatibility
            progress_bar = tqdm(
                range(self.max_iter), 
                desc="AMICA", 
                unit="it",
                ncols=60,  # Smaller fixed width
                dynamic_ncols=True,  # Adapt to terminal width
                ascii=True,  # Use ASCII characters for better compatibility
                miniters=1,  # Update on every iteration
                leave=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'  # Simpler format
            )
            iterator = progress_bar
        else:
            iterator = range(self.max_iter)

        try:
            for iter in iterator:
                self.iter = iter
                final_iter = iter

                # Get updates and likelihood
                updates = self._get_updates_and_likelihood()

                # Update parameters
                self._update_parameters(updates)

                # Calculate metrics for logging/progress
                elapsed_time = time.time() - start_time
                seconds_per_iter = elapsed_time / (iter + 1) if iter > 0 else elapsed_time
                total_seconds = seconds_per_iter * self.max_iter
                total_hours = total_seconds / 3600
                current_seconds = (elapsed_time / 3600 - int(elapsed_time / 3600)) * 3600
                
                if len(self.ll) > 1:
                    ll_diff = self.ll[-1] - self.ll[-2]
                    
                    # Update tqdm progress bar - only show iteration count in the bar itself
                    # We'll display LL in a separate line after completion
                    
                    # Log detailed per-line format if verbose or not using tqdm
                    if self.verbose or not self.use_tqdm:
                        if self.use_grad_norm:
                            self.logger.info(
                                f" iter {iter+1:5d} lrate = {self.lrate:12.10f} "
                                f"LL = {self.ll[-1]:13.10f} "
                                f"nd = {self.nd[-1]:11.10f}, "
                                f"D = {ll_diff:11.5e} {ll_diff:11.5e}  "
                                f"({current_seconds:5.2f} s, {total_hours:4.1f} h)"
                            )

                # Check convergence
                converged, reason = self._check_convergence(numdecs, numincs)
                if converged:
                    convergence_reason = reason
                    break

                # Reject outliers if requested
                if (self.do_reject and self.maxrej > 0 and (
                        (iter == self.rejstart) or (
                            ((iter - self.rejstart) % self.rejint == 0) and (numrej < self.maxrej)))):
                    self._reject_outliers()
                    numrej += 1

                # Share components if requested
                if (self.share_comps and iter >= self.share_start and (iter - self.share_start) % self.share_int == 0):
                    self.comp_list, self.comp_used = identify_shared_components(
                        self.A, self.W, self.comp_list, self.comp_thresh)

                # Write intermediate results if requested
                if self.writestep > 0 and iter % self.writestep == 0:
                    self._write_results()

                # Write history if requested
                if self.do_history and iter % self.histstep == 0:
                    self._write_history()
        finally:
            # Close the progress bar if using tqdm
            if use_tqdm_progress:
                progress_bar.close()
                
                # Display final metrics after progress bar is closed
                if len(self.ll) > 0 and self.use_grad_norm and len(self.nd) > 0:
                    self.logger.info(f"Final LL: {self.ll[-1]:.6e}, Gradient norm: {self.nd[-1]:.6e}")
            
            # Log convergence reason if available
            if convergence_reason:
                self.logger.info(convergence_reason)
                
            self.logger.info(f"Optimization finished after {final_iter+1} iterations")

    def _check_convergence(self, numdecs: int, numincs: int) -> Tuple[bool, Optional[str]]:
        """
        Check convergence criteria.

        Parameters
        ----------
        numdecs : int
            Number of consecutive likelihood decreases
        numincs : int
            Number of consecutive small likelihood increases

        Returns
        -------
        converged : bool
            True if optimization should stop
        reason : str or None
            Reason for convergence if converged, None otherwise
        """
        if self.iter == 0:
            return False, None

        # Check for NaN
        if np.isnan(self.ll[-1]):
            return True, "NaN encountered in likelihood"

        # Check for likelihood decrease
        if self.ll[-1] < self.ll[-2]:
            numdecs += 1
            if self.lrate <= self.minlrate or numdecs >= self.max_decs:
                return True, "Converged due to likelihood decrease"
            self.lrate *= self.lratefact

        # Check for small likelihood increase
        if self.use_min_dll:
            if self.ll[-1] - self.ll[-2] < self.min_dll:
                numincs += 1
                if numincs > self.max_decs:
                    return True, "Converged due to small likelihood increase"
            else:
                numincs = 0

        # Check gradient norm
        if self.use_grad_norm and self.nd[-1] < self.min_grad_norm:
            return True, "Converged due to small gradient norm"

        return False, None

    def _reject_outliers(self):
        """Reject outlier data points based on likelihood."""
        if not self.do_reject:
            return

        # Compute likelihood statistics
        ll_mean = np.mean(self.ll[-1])
        ll_std = np.std(self.ll[-1])

        # Identify outliers
        outliers = self.ll[-1] < (ll_mean - self.rejsig * ll_std)

        # Update data mask
        self.data_mask[outliers] = False
        self.num_good_samples = np.sum(~outliers)

        self.logger.info(f"Rejected {np.sum(outliers)} samples")

    def _write_results(self):
        """Write current results to disk."""
        if not self.outdir.exists():
            self.outdir.mkdir(parents=True)

        # Save parameters
        np.save(self.outdir / "A.npy", self.A)
        np.save(self.outdir / "W.npy", self.W)
        np.save(self.outdir / "c.npy", self.c)
        np.save(self.outdir / "mu.npy", self.mu)
        np.save(self.outdir / "alpha.npy", self.alpha)
        np.save(self.outdir / "beta.npy", self.beta)
        np.save(self.outdir / "rho.npy", self.rho)
        np.save(self.outdir / "gm.npy", self.gm)
        np.save(self.outdir / "mean.npy", self.mean)
        np.save(self.outdir / "sphere.npy", self.sphere)
        np.save(self.outdir / "comp_list.npy", self.comp_list)

        # Save optimization history
        np.save(self.outdir / "ll.npy", self.ll)
        if self.use_grad_norm:
            np.save(self.outdir / "nd.npy", self.nd)

    def _write_history(self):
        """Write optimization history at current iteration."""
        if not self.do_history:
            return

        hist_dir = self.outdir / "history" / f"{self.iter:06d}"
        if not hist_dir.exists():
            hist_dir.mkdir(parents=True)

        # Save current state
        np.save(hist_dir / "A.npy", self.A)
        np.save(hist_dir / "W.npy", self.W)
        np.save(hist_dir / "c.npy", self.c)
        np.save(hist_dir / "mu.npy", self.mu)
        np.save(hist_dir / "alpha.npy", self.alpha)
        np.save(hist_dir / "beta.npy", self.beta)
        np.save(hist_dir / "rho.npy", self.rho)
        np.save(hist_dir / "gm.npy", self.gm)
        np.save(hist_dir / "mean.npy", self.mean)
        np.save(hist_dir / "sphere.npy", self.sphere)
        np.save(hist_dir / "comp_list.npy", self.comp_list)
        np.save(hist_dir / "ll.npy", self.ll)
        if self.use_grad_norm:
            np.save(hist_dir / "nd.npy", self.nd)

    def transform(self, data: np.ndarray) -> np.ndarray:
        """
        Apply the learned unmixing matrices to new data.

        Parameters
        ----------
        data : ndarray of shape (n_channels, n_samples)
            The data to transform

        Returns
        -------
        S : ndarray of shape (n_components, n_samples, n_models)
            The unmixed sources for each model
        """
        if self.mean is not None:
            data = data - self.mean

        if self.sphere is not None:
            data = np.dot(self.sphere, data)

        S = np.zeros((self.num_comps, data.shape[1], self.num_models))
        for h in range(self.num_models):
            idx = self.comp_list[:, h]
            S[idx, :, h] = np.dot(self.W[:, :, h], data)

        return S
