#!/usr/bin/env python
"""
Test the fixed PyTorch AMICA implementation.
"""

import numpy as np
import json
from pathlib import Path

from pyAMICA.torch_impl import AMICATorch, setup_device
from pyAMICA.torch_impl.utils import load_eeglab_data


def test_normal_mode():
    """Test normal mode with tqdm progress bar."""
    print("=" * 70)
    print("Testing NORMAL MODE (tqdm progress bar)")
    print("=" * 70)
    
    # Load sample data
    sample_dir = Path('pyAMICA/sample_data')
    data_file = sample_dir / 'eeglab_data.fdt'
    params_file = sample_dir / 'sample_params.json'
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    data = load_eeglab_data(
        str(data_file),
        data_dim=params['data_dim'],
        field_dim=params['field_dim'][0],
        dtype=np.float32
    )
    
    # Create model
    device = setup_device()
    print(f"Device: {device}")
    
    model = AMICATorch(
        n_channels=params['data_dim'],
        n_sources=params.get('num_comps', params['data_dim']),
        n_models=params.get('num_models', 1),
        n_mix=params.get('num_mix', 3),
        device=device
    )
    
    # Fit with normal mode (tqdm)
    print("\nRunning optimization...")
    model.fit(
        data,
        max_iter=50,  # Limited for testing
        lrate=params.get('lrate', 0.05),
        debug=False,  # Normal mode with tqdm
        do_mean=params.get('do_mean', True),
        do_sphere=params.get('do_sphere', True)
    )
    
    print(f"\nFinal LL: {model.ll_history[-1]:.4f}")
    print("Normal mode test completed!")
    

def test_debug_mode():
    """Test debug mode with Fortran-style output."""
    print("\n" + "=" * 70)
    print("Testing DEBUG MODE (Fortran-style output)")
    print("=" * 70)
    
    # Load sample data
    sample_dir = Path('pyAMICA/sample_data')
    data_file = sample_dir / 'eeglab_data.fdt'
    params_file = sample_dir / 'sample_params.json'
    
    with open(params_file, 'r') as f:
        params = json.load(f)
    
    data = load_eeglab_data(
        str(data_file),
        data_dim=params['data_dim'],
        field_dim=params['field_dim'][0],
        dtype=np.float32
    )
    
    # Create model
    device = setup_device()
    model = AMICATorch(
        n_channels=params['data_dim'],
        n_sources=params.get('num_comps', params['data_dim']),
        n_models=params.get('num_models', 1),
        n_mix=params.get('num_mix', 3),
        device=device
    )
    
    # Fit with debug mode (Fortran-style)
    output_dir = Path('pytorch_debug_test')
    model.fit(
        data,
        max_iter=50,  # Limited for testing
        lrate=params.get('lrate', 0.05),
        debug=True,  # Debug mode with Fortran output
        output_dir=str(output_dir),
        do_mean=params.get('do_mean', True),
        do_sphere=params.get('do_sphere', True)
    )
    
    print(f"\nDebug output written to: {output_dir}/out.txt")
    print("Debug mode test completed!")


def main():
    """Run both test modes."""
    try:
        # Test normal mode
        test_normal_mode()
        
        # Test debug mode
        test_debug_mode()
        
        print("\n" + "=" * 70)
        print("ALL TESTS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())