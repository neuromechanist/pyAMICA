#!/usr/bin/env python
"""
Test the enhanced PyTorch AMICA with Newton optimization and adaptive PDFs.
"""

import numpy as np
import json
from pathlib import Path

from pyAMICA.torch_impl.amica_torch_v2 import AMICATorchV2
from pyAMICA.torch_impl.utils import load_eeglab_data, setup_device


def test_enhanced_features():
    """Test Newton optimization and adaptive PDFs."""
    print("=" * 70)
    print("Testing Enhanced AMICA Features")
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
    
    # Setup device
    device = setup_device()
    print(f"Device: {device}")
    
    # Create enhanced model with fixed seed
    model = AMICATorchV2(
        n_channels=params['data_dim'],
        n_sources=params.get('num_comps', params['data_dim']),
        n_models=params.get('num_models', 1),
        n_mix=params.get('num_mix', 3),
        adaptive_pdf=True,  # Enable adaptive PDFs
        device=device,
        seed=42  # Fixed seed for reproducibility
    )
    
    print("\nConfiguration:")
    print(f"  Channels: {model.n_channels}")
    print(f"  Sources: {model.n_sources}")
    print(f"  Models: {model.n_models}")
    print(f"  Mixtures: {model.n_mix}")
    print(f"  Adaptive PDF: Enabled")
    print(f"  Seed: 42")
    
    # Test with debug output to see Newton ramping
    output_dir = Path('enhanced_test_output')
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "-" * 70)
    print("Running with Newton optimization and adaptive PDFs...")
    print("-" * 70)
    
    model.fit(
        data,
        max_iter=100,
        lrate=0.05,
        do_newton=True,  # Enable Newton
        newt_start=50,   # Start Newton at iter 50 (like Fortran)
        debug=True,      # Use debug output to see details
        output_dir=str(output_dir),
        do_mean=params.get('do_mean', True),
        do_sphere=params.get('do_sphere', True)
    )
    
    print(f"\nOutput written to: {output_dir}/out.txt")
    
    # Check convergence
    final_ll = model.ll_history[-1]
    print(f"\nFinal log-likelihood: {final_ll:.6f}")
    
    # Check PDF adaptation
    if model.adaptive_pdf:
        pdf_types = model.adaptive_pdf.get_pdf_info()
        pdf_counts = {t: pdf_types.count(t) for t in set(pdf_types)}
        print(f"PDF type distribution: {pdf_counts}")
        
        # Show source statistics
        print(f"Mean kurtosis: {model.adaptive_pdf.kurtosis.mean().item():.3f}")
        print(f"Mean |skewness|: {model.adaptive_pdf.skewness.abs().mean().item():.3f}")
    
    # Compare with basic version
    print("\n" + "=" * 70)
    print("Comparing with Basic Implementation")
    print("=" * 70)
    
    from pyAMICA.torch_impl.amica_torch import AMICATorch
    
    basic_model = AMICATorch(
        n_channels=params['data_dim'],
        n_sources=params.get('num_comps', params['data_dim']),
        n_models=params.get('num_models', 1),
        n_mix=params.get('num_mix', 3),
        device=device
    )
    
    # Set same seed
    torch.manual_seed(42)
    np.random.seed(42)
    
    print("Running basic model (no Newton, no adaptive PDF)...")
    basic_model.fit(
        data,
        max_iter=100,
        lrate=0.05,
        do_newton=False,
        debug=False,
        do_mean=params.get('do_mean', True),
        do_sphere=params.get('do_sphere', True)
    )
    
    basic_ll = basic_model.ll_history[-1]
    print(f"Basic model final LL: {basic_ll:.6f}")
    print(f"Enhanced model final LL: {final_ll:.6f}")
    print(f"Improvement: {final_ll - basic_ll:.6f}")
    
    if final_ll > basic_ll:
        print("✓ Enhanced model achieved better likelihood!")
    else:
        print("⚠ Basic model performed better (may need tuning)")
    
    return model


def compare_with_fortran():
    """Compare enhanced model with Fortran results."""
    print("\n" + "=" * 70)
    print("Comparing Enhanced Model with Fortran")
    print("=" * 70)
    
    # Run validation script
    import subprocess
    import sys
    
    result = subprocess.run(
        [sys.executable, 'validate_implementations.py', '--seed', '42', '--max-iter', '100'],
        capture_output=True,
        text=True
    )
    
    if result.returncode == 0:
        # Parse output for correlation
        for line in result.stdout.split('\n'):
            if 'Mean Correlation:' in line:
                print(line)
            elif 'Min Correlation:' in line:
                print(line)
            elif 'Max Correlation:' in line:
                print(line)
    else:
        print("Validation script failed:")
        print(result.stderr)


if __name__ == '__main__':
    import torch
    
    # Test enhanced features
    model = test_enhanced_features()
    
    # Compare with Fortran
    try:
        compare_with_fortran()
    except Exception as e:
        print(f"\nCouldn't run Fortran comparison: {e}")
    
    print("\n" + "=" * 70)
    print("Enhanced AMICA Testing Complete!")
    print("=" * 70)