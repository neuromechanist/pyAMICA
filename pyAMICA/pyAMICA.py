"""AMICA (Adaptive Mixture ICA) implementation."""

import numpy as np
from scipy import linalg
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import warnings

from amica_utils import (
    gammaln, psifun, determine_block_size, identify_shared_components,
    get_unmixing_matrices, reject_outliers
)


class AMICA:
    """
    Adaptive Mixture ICA (AMICA) implementation.
    
    This class implements the AMICA algorithm for blind source separation using
    adaptive mixtures of independent component analyzers.
    """
    
    def __init__(
        self,
        num_models: int = 1,
        num_mix: int = 3,
        max_iter: int = 2000,
        do_newton: bool = False,
        newt_start: int = 20,
        newt_ramp: int = 10,
        newtrate: float = 0.5,
        do_reject: bool = False,
        rejsig: float = 3.0,
        rejstart: int = 2,
        rejint: int = 3,
        maxrej: int = 1,
        num_comps: int = -1,
        lrate: float = 0.1,
        minlrate: float = 1e-12,
        lratefact: float = 0.5,
        rho0: float = 1.5,
        minrho: float = 1.0,
        maxrho: float = 2.0,
        rholrate: float = 0.05,
        rholratefact: float = 0.1,
        invsigmax: float = 1000.0,
        invsigmin: float = 1e-4,
        do_history: bool = False,
        histstep: int = 10,
        do_opt_block: bool = True,
        block_size: int = 128,
        blk_min: int = 128,
        blk_max: int = 1024,
        blk_step: int = 128,
        share_comps: bool = False,
        comp_thresh: float = 0.99,
        share_start: int = 100,
        share_int: int = 100,
        doscaling: bool = True,
        scalestep: int = 1,
        do_sphere: bool = True,
        do_mean: bool = True,
        do_approx_sphere: bool = True,
        pcakeep: Optional[int] = None,
        pcadb: Optional[float] = None,
        writestep: int = 100,
        max_decs: int = 5,
        min_dll: float = 1e-9,
        min_grad_norm: float = 1e-7,
        use_min_dll: bool = True,
        use_grad_norm: bool = True,
        pdftype: int = 1,
        outdir: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Initialize AMICA with given parameters."""
        # Store parameters
        self.num_models = num_models
        self.num_mix = num_mix
        self.max_iter = max_iter
        self.do_newton = do_newton
        self.newt_start = newt_start
        self.newt_ramp = newt_ramp
        self.newtrate = newtrate
        self.do_reject = do_reject
        self.rejsig = rejsig
        self.rejstart = rejstart
        self.rejint = rejint
        self.maxrej = maxrej
        self.num_comps = num_comps
        self.lrate = lrate
        self.lrate0 = lrate
        self.minlrate = minlrate
        self.lratefact = lratefact
        self.rho0 = rho0
        self.minrho = minrho
        self.maxrho = maxrho
        self.rholrate = rholrate
        self.rholrate0 = rholrate
        self.rholratefact = rholratefact
        self.invsigmax = invsigmax
        self.invsigmin = invsigmin
        self.do_history = do_history
        self.histstep = histstep
        self.do_opt_block = do_opt_block
        self.block_size = block_size
        self.blk_min = blk_min
        self.blk_max = blk_max
        self.blk_step = blk_step
        self.share_comps = share_comps
        self.comp_thresh = comp_thresh
        self.share_start = share_start
        self.share_int = share_int
        self.doscaling = doscaling
        self.scalestep = scalestep
        self.do_sphere = do_sphere
        self.do_mean = do_mean
        self.do_approx_sphere = do_approx_sphere
        self.pcakeep = pcakeep
        self.pcadb = pcadb
        self.writestep = writestep
        self.max_decs = max_decs
        self.min_dll = min_dll
        self.min_grad_norm = min_grad_norm
        self.use_min_dll = use_min_dll
        self.use_grad_norm = use_grad_norm
        self.pdftype = pdftype
        self.outdir = Path(outdir) if outdir else Path.cwd() / "output"
        
        # Initialize random state
        self.rng = np.random.RandomState(seed)
        
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
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

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
                    np.diag(1.0/np.sqrt(evals[:n_comp])), 
                    evecs[:,:n_comp].T)
            else:
                # Exact sphering
                self.sphere = linalg.inv(
                    np.dot(np.diag(np.sqrt(evals[:n_comp])),
                          evecs[:,:n_comp].T))
            
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
                    self.A[:, h*self.data_dim:(h+1)*self.data_dim] = (
                        np.eye(self.data_dim) + 
                        0.01 * (0.5 - self.rng.rand(self.data_dim, self.data_dim))
                    )
                else:
                    self.A[:, h*self.data_dim:(h+1)*self.data_dim] = np.eye(self.data_dim)
                    
        # Initialize component assignments
        self.comp_list = np.zeros((self.data_dim, self.num_models), dtype=int)
        self.comp_used = np.ones(self.num_comps, dtype=bool)
        for h in range(self.num_models):
            self.comp_list[:,h] = np.arange(h*self.data_dim, (h+1)*self.data_dim)
            
        # Initialize mixture parameters
        if self.mu is None:
            self.mu = np.zeros((self.num_mix, self.num_comps))
            for k in range(self.num_comps):
                self.mu[:,k] = np.linspace(-1, 1, self.num_mix)
                if not hasattr(self, 'fix_init') or not self.fix_init:
                    self.mu[:,k] += 0.05 * (1 - 2*self.rng.rand(self.num_mix))
                    
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

    def _compute_pdf(self, y: np.ndarray, rho: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute PDF value and its derivative for given activation.
        
        Parameters
        ----------
        y : ndarray
            Activation values
        rho : float
            Shape parameter
            
        Returns
        -------
        pdf : ndarray
            PDF values
        dpdf : ndarray
            PDF derivatives
        """
        if rho == 1.0:
            # Laplace distribution
            pdf = np.exp(-np.abs(y)) / 2.0
            dpdf = -np.sign(y) * pdf
        elif rho == 2.0:
            # Gaussian distribution
            pdf = np.exp(-y*y) / np.sqrt(np.pi)
            dpdf = -2 * y * pdf
        else:
            # Generalized Gaussian distribution
            pdf = np.exp(-np.power(np.abs(y), rho)) / (
                2.0 * gammaln(1.0 + 1.0/rho))
            dpdf = -rho * np.power(np.abs(y), rho-1) * np.sign(y) * pdf
            
        return pdf, dpdf

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
            X = self.data[:,start:end]
            
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
            b[:,:,h] = np.dot(X.T, self.W[:,:,h]) - self.c[:,h]
            
        # Compute mixture probabilities and responsibilities
        z = np.zeros((batch_size, self.data_dim, self.num_mix, self.num_models))
        for h in range(self.num_models):
            for i in range(self.data_dim):
                k = self.comp_list[i,h]
                for j in range(self.num_mix):
                    y = self.beta[j,k] * (b[:,i,h] - self.mu[j,k])
                    
                    # Compute PDF and its derivative
                    pdf, dpdf = self._compute_pdf(y, self.rho[j,k])
                    
                    # Compute log probability
                    z[:,i,j,h] = np.log(self.alpha[j,k]) + np.log(self.beta[j,k]) + np.log(pdf)
                        
        # Normalize responsibilities
        z = np.exp(z - np.max(z, axis=2, keepdims=True))
        z /= np.sum(z, axis=2, keepdims=True)
        
        # Compute model probabilities
        v = np.zeros((batch_size, self.num_models))
        for h in range(self.num_models):
            v[:,h] = np.log(self.gm[h])
            for i in range(self.data_dim):
                k = self.comp_list[i,h]
                v[:,h] += np.sum(z[:,i,:,h] * z[:,i,:,h], axis=1)
                
        v = np.exp(v - np.max(v, axis=1, keepdims=True))
        v /= np.sum(v, axis=1, keepdims=True)
        
        # Accumulate parameter updates
        updates['ll'] = np.sum(np.log(np.sum(v, axis=1)))
        
        for h in range(self.num_models):
            # Model weights
            updates['dgm'][h] = np.sum(v[:,h])
            
            for i in range(self.data_dim):
                k = self.comp_list[i,h]
                
                # Mixture weights
                for j in range(self.num_mix):
                    updates['dalpha'][j,k] += np.sum(v[:,h] * z[:,i,j,h])
                    
                    # Component means
                    y = self.beta[j,k] * (b[:,i,h] - self.mu[j,k])
                    pdf, dpdf = self._compute_pdf(y, self.rho[j,k])
                    updates['dmu'][j,k] += np.sum(v[:,h] * z[:,i,j,h] * dpdf)
                    
                    # Scale parameters
                    updates['dbeta'][j,k] += np.sum(
                        v[:,h] * z[:,i,j,h] * y * dpdf)
                    
                    # Shape parameters
                    if self.rho[j,k] not in (1.0, 2.0):
                        logy = np.log(np.abs(y))
                        updates['drho'][j,k] += np.sum(
                            v[:,h] * z[:,i,j,h] * 
                            np.power(np.abs(y), self.rho[j,k]) * logy)
                        
                    if self.do_newton:
                        # Newton optimization parameters
                        updates['dbaralpha'][j,i,h] += np.sum(v[:,h] * z[:,i,j,h])
                        updates['dsigma2'][i,h] += np.sum(v[:,h] * b[:,i,h]**2)
                        updates['dlambda'][i,h] += np.sum(
                            v[:,h] * z[:,i,j,h] * (dpdf * y - 1)**2)
                        updates['dkappa'][i,h] += np.sum(
                            v[:,h] * z[:,i,j,h] * dpdf**2)
                        
            # Unmixing matrices
            g = np.zeros((batch_size, self.data_dim))
            for i in range(self.data_dim):
                k = self.comp_list[i,h]
                for j in range(self.num_mix):
                    y = self.beta[j,k] * (b[:,i,h] - self.mu[j,k])
                    _, dpdf = self._compute_pdf(y, self.rho[j,k])
                    g[:,i] += self.beta[j,k] * z[:,i,j,h] * dpdf
                        
            updates['dA'][:,:,h] += np.dot(X, v[:,h:h+1] * g)
            
            # Bias terms
            updates['dc'][:,h] += np.sum(v[:,h:h+1] * g, axis=0)
            
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
            self.sigma2 = updates['dsigma2'] / updates['dgm'][:,None]
            self.lambda_ = updates['dlambda'] / updates['dgm'][:,None]
            self.kappa = updates['dkappa'] / updates['dgm'][:,None]
            self.baralpha = updates['dbaralpha'] / updates['dgm'][:,None,None]
            
            # Newton updates
            self.lrate = min(self.newtrate, 
                           self.lrate + min(1.0/self.newt_ramp, self.lrate))
                           
            for h in range(self.num_models):
                dA = -updates['dA'][:,:,h] / updates['dgm'][h]
                dA[np.diag_indices_from(dA)] += 1
                
                # Compute Newton direction
                H = np.zeros_like(dA)
                for i in range(self.data_dim):
                    for j in range(self.data_dim):
                        if i == j:
                            H[i,i] = dA[i,i] / self.lambda_[i,h]
                        else:
                            sk1 = self.sigma2[i,h] * self.kappa[j,h]
                            sk2 = self.sigma2[j,h] * self.kappa[i,h]
                            if sk1*sk2 > 1.0:
                                H[i,j] = (sk1*dA[i,j] - dA[j,i]) / (sk1*sk2 - 1.0)
                            
                self.A[:,self.comp_list[:,h]] += self.lrate * np.dot(
                    self.A[:,self.comp_list[:,h]], H)
        else:
            # Natural gradient updates
            self.lrate = min(self.lrate0,
                           self.lrate + min(1.0/self.newt_ramp, self.lrate))
            
            for h in range(self.num_models):
                dA = -updates['dA'][:,:,h] / updates['dgm'][h]
                dA[np.diag_indices_from(dA)] += 1
                
                self.A[:,self.comp_list[:,h]] += self.lrate * np.dot(
                    self.A[:,self.comp_list[:,h]], dA)
            
        # Update bias terms
        self.c += self.lrate * updates['dc'] / updates['dgm'][:,None]
        
        # Rescale parameters if requested
        if self.doscaling and self.iter % self.scalestep == 0:
            for k in range(self.num_comps):
                scale = np.sqrt(np.sum(self.A[:,k]**2))
                if scale > 0:
                    self.A[:,k] /= scale
                    self.mu[:,k] *= scale
                    self.beta[:,k] /= scale
                    
        # Update unmixing matrices
        self._update_unmixing_matrices()
        
        # Store likelihood
        self.ll.append(updates['ll'])
        
        # Compute gradient norm
        if self.use_grad_norm:
            dA = np.zeros_like(self.A)
            for h in range(self.num_models):
                dA[:,self.comp_list[:,h]] += self.gm[h] * updates['dA'][:,:,h]
            self.nd.append(np.sqrt(np.sum(dA**2) / 
                                 (self.data_dim * self.num_comps)))

    def _optimize(self):
        """Main optimization loop."""
        self.logger.info("Starting optimization...")
        
        # Initialize optimization variables
        numdecs = 0
        numincs = 0
        numrej = 0
        
        for iter in range(self.max_iter):
            self.iter = iter
            
            # Get updates and likelihood
            updates = self._get_updates_and_likelihood()
            
            # Update parameters
            self._update_parameters(updates)
            
            # Check convergence
            if self._check_convergence(numdecs, numincs):
                break
                
            # Reject outliers if requested
            if (self.do_reject and self.maxrej > 0 and 
                ((iter == self.rejstart) or 
                 (((iter - self.rejstart) % self.rejint == 0) and 
                  (numrej < self.maxrej)))):
                self._reject_outliers()
                numrej += 1
                
            # Share components if requested
            if (self.share_comps and iter >= self.share_start and
                (iter - self.share_start) % self.share_int == 0):
                self.comp_list, self.comp_used = identify_shared_components(
                    self.A, self.W, self.comp_list, self.comp_thresh)
                
            # Write intermediate results if requested
            if self.writestep > 0 and iter % self.writestep == 0:
                self._write_results()
                
            # Write history if requested
            if self.do_history and iter % self.histstep == 0:
                self._write_history()
                
        self.logger.info(f"Optimization finished after {iter+1} iterations")

    def _check_convergence(self, numdecs: int, numincs: int) -> bool:
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
        bool
            True if optimization should stop
        """
        if self.iter == 0:
            return False
            
        # Check for NaN
        if np.isnan(self.ll[-1]):
            self.logger.warning("NaN encountered in likelihood")
            return True
            
        # Check for likelihood decrease
        if self.ll[-1] < self.ll[-2]:
            numdecs += 1
            if self.lrate <= self.minlrate or numdecs >= self.max_decs:
                self.logger.info("Converged due to likelihood decrease")
                return True
            self.lrate *= self.lratefact
            
        # Check for small likelihood increase
        if self.use_min_dll:
            if self.ll[-1] - self.ll[-2] < self.min_dll:
                numincs += 1
                if numincs > self.max_decs:
                    self.logger.info("Converged due to small likelihood increase")
                    return True
            else:
                numincs = 0
                
        # Check gradient norm
        if self.use_grad_norm and self.nd[-1] < self.min_grad_norm:
            self.logger.info("Converged due to small gradient norm")
            return True
            
        return False
        
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
            idx = self.comp_list[:,h]
            S[idx,:,h] = np.dot(self.W[:,:,h], data)
            
        return S
