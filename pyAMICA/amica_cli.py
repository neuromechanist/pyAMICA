#!/usr/bin/env python3
"""Command-line interface for AMICA."""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import numpy as np

from amica import AMICA
from amica_data import load_multiple_files


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
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


def load_params(paramfile: str) -> Dict[str, Any]:
    """
    Load parameters from JSON file.
    
    Parameters
    ----------
    paramfile : str
        Path to JSON parameter file
        
    Returns
    -------
    params : dict
        Dictionary of parameters
    """
    with open(paramfile) as f:
        params = json.load(f)
        
    # Required parameters
    required = {
        'files',
        'num_samples',
        'data_dim',
        'field_dim'
    }
    
    missing = required - set(params.keys())
    if missing:
        raise ValueError(
            f"Missing required parameters: {', '.join(missing)}")
            
    return params


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def main():
    """Main entry point."""
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
        params['field_dim'],
        params['num_samples']
    )
    
    # Create output directory
    outdir = Path(args.outdir)
    if not outdir.exists():
        outdir.mkdir(parents=True)
        
    # Initialize AMICA
    model_params = {
        k: v for k, v in params.items()
        if k not in {'files', 'num_samples', 'data_dim', 'field_dim'}
    }
    model_params.update({
        'outdir': str(outdir),
        'seed': args.seed
    })
    
    model = AMICA(**model_params)
    
    # Fit model
    logger.info("Fitting AMICA model")
    model.fit(data)
    
    logger.info(f"Results saved to {outdir}")
    

if __name__ == '__main__':
    main()
