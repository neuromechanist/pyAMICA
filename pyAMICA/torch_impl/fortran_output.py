"""
Fortran-style output formatting for PyTorch AMICA.
Produces output similar to the original Fortran implementation for debugging and comparison.
"""

import sys
import time
from pathlib import Path
from typing import Optional, TextIO
from datetime import datetime


class FortranStyleOutput:
    """
    Produces Fortran AMICA-style output for convergence monitoring.
    
    Parameters
    ----------
    output_file : str, optional
        Path to output file (default: 'out.txt')
    verbose : bool, default=True
        Print to stdout in addition to file
    debug : bool, default=False
        Enable detailed debug output
    """
    
    def __init__(
        self,
        output_file: str = 'out.txt',
        verbose: bool = True,
        debug: bool = False
    ):
        self.output_file = Path(output_file)
        self.verbose = verbose
        self.debug = debug
        self.start_time = None
        self.file_handle: Optional[TextIO] = None
        
        # Open output file
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        self.file_handle = open(self.output_file, 'w', buffering=1)  # Line buffering
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
            
    def write(self, message: str, to_file: bool = True, to_stdout: bool = None):
        """Write message to file and/or stdout."""
        if to_stdout is None:
            to_stdout = self.verbose
            
        if to_file and self.file_handle:
            self.file_handle.write(message)
            self.file_handle.flush()
            
        if to_stdout:
            sys.stdout.write(message)
            sys.stdout.flush()
            
    def write_header(self, config: dict):
        """Write header similar to Fortran output."""
        self.start_time = time.time()
        
        header = []
        header.append("=" * 70 + "\n")
        header.append(f"PyTorch AMICA - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        header.append("=" * 70 + "\n")
        
        # Configuration summary
        header.append(f"Data dimensions: {config.get('n_channels', 0)} channels, {config.get('n_samples', 0)} samples\n")
        header.append(f"Number of models: {config.get('n_models', 1)}\n")
        header.append(f"Number of sources: {config.get('n_sources', config.get('n_channels', 0))}\n")
        header.append(f"Mixture components: {config.get('n_mix', 3)}\n")
        header.append(f"Max iterations: {config.get('max_iter', 100)}\n")
        header.append(f"Learning rate: {config.get('lrate', 0.1):.6f}\n")
        header.append(f"Newton optimization: {config.get('do_newton', False)}\n")
        
        if config.get('do_newton', False):
            header.append(f"Newton start: {config.get('newt_start', 50)}\n")
            header.append(f"Newton ramp: {config.get('newt_ramp', 10)}\n")
            
        header.append(f"Device: {config.get('device', 'cpu')}\n")
        header.append("-" * 70 + "\n")
        
        for line in header:
            self.write(line)
            
    def write_preprocessing(self, do_mean: bool, do_sphere: bool, eigenvalues: Optional[list] = None):
        """Write preprocessing information."""
        lines = []
        
        if do_mean:
            lines.append("Removing mean from data\n")
            
        if do_sphere:
            lines.append("Sphering data\n")
            
            if eigenvalues is not None and len(eigenvalues) > 0:
                min_eig = min(eigenvalues)
                max_eig = max(eigenvalues)
                lines.append(f" minimum eigenvalues = {min_eig:.14f}\n")
                lines.append(f" maximum eigenvalues = {max_eig:.14f}\n")
                lines.append(f" num eigs kept = {len(eigenvalues)}\n")
                
        for line in lines:
            self.write(line)
            
    def write_iteration(
        self,
        iter: int,
        lrate: float,
        ll: float,
        nd: float,
        dll: float = 0.0,
        dll_old: float = 0.0,
        is_newton: bool = False
    ):
        """
        Write iteration information in Fortran format.
        
        Format matches:
        iter    10 lrate =  0.0500000000 LL =  -3.4503691075 nd =  0.0225617133, D =   0.57517E-01  0.57517E-01  (  0.01 s,   0.0 h)
        """
        elapsed = time.time() - self.start_time if self.start_time else 0
        elapsed_h = elapsed / 3600.0
        
        # Format the line similar to Fortran output
        line = f" iter {iter:5d} lrate = {lrate:13.10f} LL = {ll:18.10f} "
        
        # Add gradient norm
        line += f"nd = {nd:13.10f}, "
        
        # Add delta LL (D parameter in Fortran)
        if dll != 0 or dll_old != 0:
            # Use scientific notation for small values
            if abs(dll) < 0.01:
                line += f"D = {dll:12.5e} {dll_old:12.5e}"
            else:
                line += f"D = {dll:12.5f} {dll_old:12.5f}"
        
        # Add timing
        line += f"  ({elapsed:6.2f} s, {elapsed_h:5.1f} h)"
        
        # Add Newton indicator
        if is_newton:
            line += " [Newton]"
            
        line += "\n"
        
        self.write(line)
        
        # Additional debug information
        if self.debug:
            self.write(f"   [DEBUG] LL change: {dll:.6e}, Gradient norm: {nd:.6e}\n", to_stdout=False)
            
    def write_convergence(self, reason: str, iter: int, ll: float, nd: float):
        """Write convergence information."""
        lines = []
        lines.append("-" * 70 + "\n")
        lines.append(f"Convergence: {reason}\n")
        lines.append(f"Final iteration: {iter}\n")
        lines.append(f"Final LL: {ll:.10f}\n")
        lines.append(f"Final gradient norm: {nd:.10e}\n")
        
        if self.start_time:
            elapsed = time.time() - self.start_time
            lines.append(f"Total time: {elapsed:.2f} seconds\n")
            
        lines.append("=" * 70 + "\n")
        
        for line in lines:
            self.write(line)
            
    def write_warning(self, message: str):
        """Write warning message."""
        line = f" WARNING: {message}\n"
        self.write(line)
        
    def write_info(self, message: str):
        """Write info message."""
        line = f" {message}\n"
        self.write(line)
        
    def write_block_size(self, block_size: int, optimal: bool = False):
        """Write block size information."""
        if optimal:
            line = f"Optimal block size: {block_size}\n"
        else:
            line = f"Using block size: {block_size}\n"
        self.write(line)
        
    def write_rejection_info(self, iter: int, num_rejected: int, total: int):
        """Write outlier rejection information."""
        percent = 100.0 * num_rejected / total if total > 0 else 0
        line = f" iter {iter:5d}: Rejected {num_rejected}/{total} samples ({percent:.2f}%)\n"
        self.write(line)
        
    def write_component_sharing(self, iter: int, shared_pairs: list):
        """Write component sharing information."""
        if shared_pairs:
            line = f" iter {iter:5d}: Sharing {len(shared_pairs)} component pairs\n"
            self.write(line)
            
            if self.debug:
                for i, j, corr in shared_pairs:
                    self.write(f"   Components {i} and {j}: correlation = {corr:.4f}\n", to_stdout=False)
                    
    def write_newton_switch(self, iter: int):
        """Write Newton optimization switch message."""
        line = f" iter {iter:5d}: Switching to Newton optimization\n"
        self.write(line)
        
    def write_learning_rate_decrease(self, iter: int, old_lr: float, new_lr: float):
        """Write learning rate decrease message."""
        line = f" iter {iter:5d}: Decreasing learning rate from {old_lr:.6f} to {new_lr:.6f}\n"
        self.write(line)
        
    def close(self):
        """Close output file."""
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None