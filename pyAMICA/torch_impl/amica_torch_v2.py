"""
Updated PyTorch AMICA implementation with Fortran-style output.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, List, Union
from pathlib import Path
from tqdm import tqdm

from .amica_torch import AMICATorch as AMICATorchBase
from .fortran_output import FortranStyleOutput


class AMICATorch(AMICATorchBase):
    """
    Extended PyTorch AMICA with Fortran-style output and debugging.
    
    Inherits from base AMICATorch and adds:
    - Fortran-style convergence output
    - Debug mode with detailed information
    - tqdm progress bars (optional)
    """
    
    def fit(
        self,
        X: np.ndarray,
        max_iter: int = 100,
        lrate: float = 0.1,
        min_lrate: float = 1e-8,
        lrate_decay: float = 0.5,
        do_newton: bool = True,
        newton_start: int = 20,
        newton_ramp: int = 10,
        verbose: bool = True,
        debug: bool = False,
        use_tqdm: bool = True,
        output_dir: Optional[str] = None,
        write_step: int = 10,
        # Convergence criteria
        min_dll: float = 1e-9,
        min_grad_norm: float = 1e-7,
        max_decs: int = 3,
        # Preprocessing
        do_mean: bool = True,
        do_sphere: bool = True,
        **kwargs
    ) -> 'AMICATorch':
        """
        Fit AMICA model with Fortran-style output.
        
        Parameters
        ----------
        X : np.ndarray
            Input data of shape (n_channels, n_samples)
        max_iter : int, default=100
            Maximum iterations
        lrate : float, default=0.1
            Initial learning rate
        min_lrate : float, default=1e-8
            Minimum learning rate
        lrate_decay : float, default=0.5
            Learning rate decay factor
        do_newton : bool, default=True
            Use Newton optimization
        newton_start : int, default=20
            Iteration to start Newton
        newton_ramp : int, default=10
            Newton ramp iterations
        verbose : bool, default=True
            Print progress to stdout
        debug : bool, default=False
            Enable debug output
        use_tqdm : bool, default=True
            Use tqdm progress bar
        output_dir : str, optional
            Directory for output files
        write_step : int, default=10
            Iterations between file writes
        min_dll : float, default=1e-9
            Minimum log-likelihood change for convergence
        min_grad_norm : float, default=1e-7
            Minimum gradient norm for convergence
        max_decs : int, default=3
            Maximum learning rate decreases
        do_mean : bool, default=True
            Remove mean
        do_sphere : bool, default=True
            Apply sphering
        **kwargs
            Additional preprocessing arguments
            
        Returns
        -------
        self : AMICATorch
            Fitted model
        """
        # Set up output directory
        if output_dir is None:
            output_dir = Path.cwd() / 'amica_output'
        else:
            output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize output writer
        output_file = output_dir / 'out.txt'
        with FortranStyleOutput(str(output_file), verbose=verbose, debug=debug) as output:
            
            # Write header
            config = {
                'n_channels': self.n_channels,
                'n_sources': self.n_sources,
                'n_samples': X.shape[1],
                'n_models': self.n_models,
                'n_mix': self.n_mix,
                'max_iter': max_iter,
                'lrate': lrate,
                'do_newton': do_newton,
                'newton_start': newton_start,
                'newton_ramp': newton_ramp,
                'device': str(self.device)
            }
            output.write_header(config)
            
            # Preprocess data
            output.write_info("Starting preprocessing...")
            
            # Store eigenvalues if sphering
            eigenvalues = None
            if do_sphere:
                # Temporarily compute eigenvalues for output
                X_tensor = torch.from_numpy(X).to(self.device, dtype=self.dtype)
                if do_mean:
                    X_centered = X_tensor - X_tensor.mean(dim=1, keepdim=True)
                else:
                    X_centered = X_tensor
                cov = torch.cov(X_centered)
                eigvals, _ = torch.linalg.eigh(cov)
                eigenvalues = eigvals.cpu().numpy().tolist()
            
            output.write_preprocessing(do_mean, do_sphere, eigenvalues)
            
            # Preprocess data
            X_prep = self.preprocess_data(X, do_mean=do_mean, do_sphere=do_sphere, **kwargs)
            
            # Determine block size
            block_size = self._determine_block_size(X_prep.shape[1])
            output.write_block_size(block_size, optimal=True)
            
            # Initialize optimizers
            output.write_info("Starting optimization...")
            from .optimizers import NaturalGradientOptimizer, NewtonOptimizer
            
            nat_grad_opt = NaturalGradientOptimizer(
                self.parameters(), 
                lr=lrate
            )
            
            if do_newton:
                newton_opt = NewtonOptimizer(self)
            
            # Training state
            ll_history = []
            nd_history = []
            dll_old = 0.0
            num_decs = 0
            current_lrate = lrate
            converged = False
            convergence_reason = ""
            
            # Progress bar
            if use_tqdm and not verbose:
                pbar = tqdm(range(max_iter), desc="AMICA", unit="iter")
            else:
                pbar = range(max_iter)
            
            # Training loop
            for iter in pbar:
                # Check if switching to Newton
                use_newton_now = do_newton and iter >= newton_start
                if use_newton_now and iter == newton_start:
                    output.write_newton_switch(iter)
                
                # Compute gradients
                nat_grad_opt.zero_grad()
                
                # Forward pass and loss
                neg_ll = -self.compute_log_likelihood(X_prep)
                
                # Check for NaN
                if torch.isnan(neg_ll):
                    output.write_warning(f"NaN encountered at iteration {iter}")
                    converged = True
                    convergence_reason = "NaN in log-likelihood"
                    break
                
                # Backward pass
                neg_ll.backward()
                
                # Compute gradient norm
                grad_norm = self._compute_gradient_norm()
                
                # Natural gradient step
                nat_grad_opt.step()
                
                # Newton step if applicable
                if use_newton_now:
                    # Apply Newton with ramping
                    if iter < newton_start + newton_ramp:
                        newton_weight = (iter - newton_start + 1) / newton_ramp
                    else:
                        newton_weight = 1.0
                    
                    if newton_weight > 0:
                        newton_opt.step(X_prep)
                
                # Get current values
                with torch.no_grad():
                    ll = -neg_ll.item()
                    nd = grad_norm.item()
                    
                    # Compute LL change
                    if len(ll_history) > 0:
                        dll = ll - ll_history[-1]
                    else:
                        dll = 0.0
                    
                    # Store history
                    ll_history.append(ll)
                    nd_history.append(nd)
                    
                    # Write iteration info
                    if iter % write_step == 0 or iter < 10:
                        output.write_iteration(
                            iter=iter,
                            lrate=current_lrate,
                            ll=ll,
                            nd=nd,
                            dll=dll,
                            dll_old=dll_old,
                            is_newton=use_newton_now
                        )
                    
                    # Check convergence
                    if iter > 10:
                        # Check gradient norm
                        if nd < min_grad_norm:
                            converged = True
                            convergence_reason = f"Gradient norm < {min_grad_norm}"
                            
                        # Check LL change
                        elif abs(dll) < min_dll:
                            # Decrease learning rate
                            if current_lrate > min_lrate and num_decs < max_decs:
                                old_lrate = current_lrate
                                current_lrate *= lrate_decay
                                num_decs += 1
                                
                                output.write_learning_rate_decrease(iter, old_lrate, current_lrate)
                                
                                # Update optimizer learning rate
                                for param_group in nat_grad_opt.param_groups:
                                    param_group['lr'] = current_lrate
                            else:
                                converged = True
                                convergence_reason = f"LL change < {min_dll}"
                    
                    # Update progress bar
                    if use_tqdm and not verbose:
                        pbar.set_postfix({'LL': f'{ll:.4f}', 'nd': f'{nd:.2e}'})
                    
                    dll_old = dll
                
                if converged:
                    break
            
            # Write convergence info
            if converged:
                output.write_convergence(convergence_reason, iter, ll_history[-1], nd_history[-1])
            else:
                output.write_convergence("Maximum iterations reached", max_iter, ll_history[-1], nd_history[-1])
            
            # Save results
            if write_step > 0:
                self._save_checkpoint(output_dir, ll_history, nd_history)
        
        # Store history
        self.ll_history = ll_history
        self.nd_history = nd_history
        
        return self
    
    def _determine_block_size(self, n_samples: int) -> int:
        """Determine optimal block size for processing."""
        # Simple heuristic - can be improved
        if n_samples < 1000:
            return 128
        elif n_samples < 10000:
            return 256
        else:
            return 512
    
    def _compute_gradient_norm(self) -> torch.Tensor:
        """Compute the norm of all gradients."""
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        return torch.tensor(total_norm)
    
    def _save_checkpoint(self, output_dir: Path, ll_history: list, nd_history: list):
        """Save checkpoint with current state."""
        # Save model state
        torch.save(self.state_dict(), output_dir / 'checkpoint.pth')
        
        # Save histories
        np.savetxt(output_dir / 'LL.txt', ll_history)
        np.savetxt(output_dir / 'nd.txt', nd_history)
        
        # Save current parameters in NumPy format
        for h in range(self.n_models):
            W = self.get_unmixing_matrix(h)
            A = self.get_mixing_matrix(h)
            np.save(output_dir / f'W_{h}.npy', W)
            np.save(output_dir / f'A_{h}.npy', A)
        
        # Save mixture parameters
        np.save(output_dir / 'alpha.npy', self.alpha.detach().cpu().numpy())
        np.save(output_dir / 'mu.npy', self.mu.detach().cpu().numpy())
        np.save(output_dir / 'beta.npy', self.beta.detach().cpu().numpy())
        np.save(output_dir / 'rho.npy', self.rho.detach().cpu().numpy())