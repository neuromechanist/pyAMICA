#!/usr/bin/env python3
"""
Command-line interface for AMICA (Adaptive Mixture ICA).

This module provides a command-line interface for running the AMICA algorithm.
It handles:
1. Parameter loading from JSON configuration files
2. Data loading from binary files
3. Model initialization and training
4. Result saving

Example usage:
    python -m pyAMICA.amica_cli params.json  --outdir results --seed 42  # Use -m flag to run as module

The parameter file should be in JSON format and must include:
- files: List of binary data files to process
- data_dim: Number of channels/dimensions
- field_dim: Number of samples per channel for each file

Optional parameters can be included in the JSON file:
- num_models: Number of models (default: 1)
- num_mix: Number of mixture components (default: 3)
- max_iter: Maximum iterations (default: 2000)
And many others as documented in the AMICA class.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

from .pyAMICA import AMICA
from .amica_data import load_multiple_files


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for AMICA execution.

    Returns
    -------
    args : argparse.Namespace
        Parsed command line arguments with the following attributes:
        - paramfile: Path to JSON parameter file
        - outdir: Output directory for results (default: 'output')
        - seed: Random seed for reproducibility (optional)
        - verbose: Flag for detailed logging output
    """
    parser = argparse.ArgumentParser(
        description='AMICA: Adaptive Mixture ICA'
    )

    # Required arguments
    parser.add_argument(
        'paramfile',
        help='Parameter file in JSON format'
    )

    # Optional arguments
    parser.add_argument(
        '--outdir',
        help='Output directory (default: output)',
        default='output'
    )
    parser.add_argument(
        '--seed',
        help='Random seed',
        type=int
    )
    parser.add_argument(
        '--verbose',
        help='Verbose output',
        action='store_true'
    )

    return parser.parse_args()


def load_params(paramfile: str, default_paramfile: str = None) -> Dict[str, Any]:
    """
    Load and validate AMICA parameters from JSON configuration file.

    The function loads default parameters from default_paramfile (if provided),
    then updates them with user-provided parameters from paramfile.

    Required parameters in paramfile are:
    - files: List of data files to process
    - data_dim: Number of channels/dimensions
    - field_dim: List of samples per channel for each file

    Parameters
    ----------
    paramfile : str
        Path to JSON parameter file with user settings
    default_paramfile : str, optional
        Path to JSON file with default parameters

    Returns
    -------
    params : dict
        Dictionary of parameters
    """
    # Load default parameters if provided
    if default_paramfile:
        with open(default_paramfile) as f:
            params = json.load(f)
    else:
        params = {}

    # Update with user parameters
    with open(paramfile) as f:
        user_params = json.load(f)
        params.update(user_params)

    # Required parameters
    required = {
        'files',
        'data_dim',
        'field_dim'
    }

    missing = required - set(params.keys())
    if missing:
        raise ValueError(
            f"Missing required parameters: {', '.join(missing)}")

    return params


def setup_logging(verbose: bool = False):
    """
    Configure logging for AMICA execution.

    Sets up logging with appropriate level and format. When verbose is True,
    DEBUG level messages are included, otherwise only INFO and above are shown.

    Parameters
    ----------
    verbose : bool
        Whether to enable verbose (DEBUG level) logging
    """
    # Get the root logger and remove any existing handlers
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    # Get the AMICA logger and remove any existing handlers
    logger = logging.getLogger('AMICA')
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Set level based on verbose flag
    level = logging.DEBUG if verbose else logging.INFO
    logger.setLevel(level)


def main():
    """
    Main entry point for AMICA command-line execution.

    This function orchestrates the AMICA workflow:
    1. Parses command line arguments
    2. Sets up logging
    3. Loads and validates parameters
    4. Loads data from binary files
    5. Initializes and fits the AMICA model
    6. Saves results to the specified output directory

    The execution can be customized through the parameter file and
    command line arguments.
    """
    # Parse arguments
    args = parse_args()

    # Setup logging
    setup_logging(args.verbose)
    logger = logging.getLogger('AMICA')

    # Load parameters
    logger.info(f"Loading parameters from {args.paramfile}")
    params = load_params(args.paramfile)

    # Load data
    logger.info("Loading data files:")
    for f in params['files']:
        logger.info(f"  {f}")

    data = load_multiple_files(
        params['files'],
        params['data_dim'],
        params['field_dim']
    )

    # Create output directory
    outdir = Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)

    # Initialize AMICA
    model = AMICA(
        params_file=args.paramfile,
        outdir=str(outdir),
        seed=args.seed
    )

    # Fit model
    logger.info("Fitting AMICA model")
    model.fit(data)

    logger.info(f"Results saved to {outdir}")


if __name__ == '__main__':
    main()
