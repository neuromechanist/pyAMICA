#!/usr/bin/env python
"""
Example of using the PyTorch AMICA implementation.

Uses the public :class:`~pamica.AMICA` interface, which wraps the
natural-gradient EM backend (``AMICATorchNG``) that matches the Fortran
reference.

Note: model persistence (``AMICA.save``/``load``) is not yet implemented for
this backend (issue #36), so this example fits and inspects the model
in-process rather than writing it to disk.
"""

import json
import logging
from pathlib import Path

import numpy as np

# Public AMICA interface (wraps the natural-gradient EM backend)
from pamica import AMICA
from pamica.torch_impl.utils import load_eeglab_data, compare_with_fortran

# Set up logging
logging.basicConfig(level=logging.INFO)


def main():
    """Run PyTorch AMICA on sample data."""

    # Paths
    sample_dir = Path(__file__).parent.parent / "pamica" / "sample_data"
    data_file = sample_dir / "eeglab_data.fdt"
    params_file = sample_dir / "sample_params.json"
    fortran_output = sample_dir / "amicaout"

    # Load parameters
    with open(params_file, "r") as f:
        params = json.load(f)

    print(f"Loading data from {data_file}")

    # Load data
    data = load_eeglab_data(
        str(data_file),
        data_dim=params["data_dim"],
        field_dim=params["field_dim"][0],
        dtype=np.float32,
    )

    print(f"Data shape: {data.shape}")

    # Create model (device auto-selected; the float64 parity default falls
    # back to CPU when only MPS is available).
    model = AMICA(
        n_models=params.get("num_models", 1),
        n_mix=params.get("num_mix", 3),
    )

    # Fit model
    print("Fitting model...")
    model.fit(
        data,
        max_iter=params.get("max_iter", 100),
        lrate=params.get("lrate", 0.05),
        do_mean=params.get("do_mean", True),
        do_sphere=params.get("do_sphere", True),
        do_newton=params.get("do_newton", True),
        newt_start=params.get("newt_start", 50),
        seed=42,
    )

    print(f"Training complete. Final LL: {model.ll_history_[-1]:.6f}")

    # Compare with Fortran if available
    if fortran_output.exists():
        print("\nComparing with Fortran results...")
        results = {
            "W": model.get_unmixing_matrix(0),
            "A": model.get_mixing_matrix(0),
            "ll": model.ll_history_[-1],
        }
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

    return model


if __name__ == "__main__":
    model = main()
