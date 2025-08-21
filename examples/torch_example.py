#!/usr/bin/env python
"""
Example of using PyTorch AMICA implementation.
"""

import numpy as np
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Import PyTorch AMICA
from pyAMICA.torch_impl import AMICATorch, setup_device
from pyAMICA.torch_impl.utils import load_eeglab_data, save_pytorch_results, compare_with_fortran


def main():
    """Run PyTorch AMICA on sample data."""
    
    # Paths
    sample_dir = Path(__file__).parent.parent / 'pyAMICA' / 'sample_data'
    data_file = sample_dir / 'eeglab_data.fdt'
    params_file = sample_dir / 'sample_params.json'
    fortran_output = sample_dir / 'amicaout'
    pytorch_output = Path('pytorch_results')
    
    # Load parameters
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    print(f"Loading data from {data_file}")
    
    # Load data
    data = load_eeglab_data(
        str(data_file),
        data_dim=params['data_dim'],
        field_dim=params['field_dim'][0],
        dtype=np.float32
    )
    
    print(f"Data shape: {data.shape}")
    
    # Set up device (auto-selects GPU/MPS if available)
    device = setup_device()
    print(f"Using device: {device}")
    
    # Create model
    model = AMICATorch(
        n_channels=params['data_dim'],
        n_sources=params.get('num_comps', params['data_dim']),
        n_models=params.get('num_models', 1),
        n_mix=params.get('num_mix', 3),
        device=device
    )
    
    print(f"Model initialized with {model.n_models} models, {model.n_mix} mixture components")
    
    # Fit model
    print("Fitting model...")
    model.fit(
        data,
        max_iter=params.get('max_iter', 100),
        lrate=params.get('lrate', 0.05),
        do_newton=params.get('do_newton', True),
        newton_start=params.get('newt_start', 50),
        verbose=True,
        # Preprocessing options
        do_mean=params.get('do_mean', True),
        do_sphere=params.get('do_sphere', True)
    )
    
    print(f"Training complete. Final LL: {model.ll_history[-1]:.6f}")
    
    # Save results
    print(f"Saving results to {pytorch_output}")
    results = save_pytorch_results(model, str(pytorch_output))
    
    # Compare with Fortran if available
    if fortran_output.exists():
        print("\nComparing with Fortran results...")
        metrics = compare_with_fortran(results, str(fortran_output))
        
        print("Comparison metrics:")
        print(f"  Mean W correlation: {metrics.get('W_mean_corr', 0):.4f}")
        print(f"  Min W correlation:  {metrics.get('W_min_corr', 0):.4f}")
        print(f"  PyTorch LL:         {metrics.get('ll_pytorch', 0):.4f}")
        print(f"  Fortran LL:         {metrics.get('ll_fortran', 0):.4f}")
        print(f"  LL difference:      {metrics.get('ll_diff', 0):.4f}")
    
    # Transform data to get sources
    sources = model.transform(data)
    print(f"\nExtracted sources shape: {sources.shape}")
    
    # Save model for later use
    model_path = pytorch_output / 'model.pth'
    model.save(str(model_path))
    print(f"Model saved to {model_path}")
    
    return model, results


if __name__ == '__main__':
    model, results = main()