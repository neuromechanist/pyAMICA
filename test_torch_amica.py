#!/usr/bin/env python
"""
Test PyTorch AMICA with Fortran-style output on sample data.
"""

import numpy as np
import json
from pathlib import Path

from pyAMICA.torch_impl import AMICATorch, setup_device
from pyAMICA.torch_impl.utils import load_eeglab_data


def main():
    # Paths
    sample_dir = Path('pyAMICA/sample_data')
    data_file = sample_dir / 'eeglab_data.fdt'
    params_file = sample_dir / 'sample_params.json'
    output_dir = Path('pytorch_output_test')
    
    # Load parameters
    print("Loading parameters...")
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    # Load data
    print(f"Loading data from {data_file}")
    data = load_eeglab_data(
        str(data_file),
        data_dim=params['data_dim'],
        field_dim=params['field_dim'][0],
        dtype=np.float32
    )
    print(f"Data shape: {data.shape}")
    
    # Set up device
    device = setup_device()
    print(f"Using device: {device}")
    
    # Create model
    print("\nInitializing AMICA model...")
    model = AMICATorch(
        n_channels=params['data_dim'],
        n_sources=params.get('num_comps', params['data_dim']),
        n_models=params.get('num_models', 1),
        n_mix=params.get('num_mix', 3),
        device=device
    )
    
    # Fit with limited iterations for testing
    print("\nStarting optimization with Fortran-style output...")
    print("=" * 70)
    
    model.fit(
        data,
        max_iter=50,  # Limited for testing
        lrate=params.get('lrate', 0.05),
        min_lrate=params.get('minlrate', 1e-8),
        lrate_decay=params.get('lratefact', 0.5),
        do_newton=params.get('do_newton', True),
        newton_start=params.get('newt_start', 20),
        newton_ramp=params.get('newt_ramp', 10),
        verbose=True,  # Show Fortran-style output
        debug=False,   # Set to True for detailed debug info
        use_tqdm=False,  # Disable tqdm to show Fortran output
        output_dir=str(output_dir),
        write_step=5,  # Write every 5 iterations
        # Convergence criteria
        min_dll=params.get('min_dll', 1e-9),
        min_grad_norm=params.get('min_grad_norm', 1e-7),
        max_decs=params.get('max_decs', 3),
        # Preprocessing
        do_mean=params.get('do_mean', True),
        do_sphere=params.get('do_sphere', True)
    )
    
    print("\n" + "=" * 70)
    print("Test completed!")
    print(f"Output written to: {output_dir}/out.txt")
    print(f"Final LL: {model.ll_history[-1]:.6f}")
    
    # Test with debug mode
    print("\n" + "=" * 70)
    print("Testing with debug mode (first 5 iterations)...")
    print("=" * 70)
    
    model_debug = AMICATorch(
        n_channels=params['data_dim'],
        n_sources=params.get('num_comps', params['data_dim']),
        n_models=params.get('num_models', 1),
        n_mix=params.get('num_mix', 3),
        device=device
    )
    
    output_dir_debug = Path('pytorch_output_debug')
    model_debug.fit(
        data,
        max_iter=5,
        lrate=params.get('lrate', 0.05),
        verbose=True,
        debug=True,  # Enable debug output
        use_tqdm=False,
        output_dir=str(output_dir_debug),
        write_step=1
    )
    
    print("\n" + "=" * 70)
    print(f"Debug output written to: {output_dir_debug}/out.txt")


if __name__ == '__main__':
    main()