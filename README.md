# [WIP] AMICA: Adaptive Mixture ICA

Python implementation of the Adaptive Mixture ICA algorithm, based on the original Fortran implementation. This implementation is designed to be more user-friendly and easier to integrate with other Python libraries.

NOTE: This is a work in progress and may not be fully functional yet. User should not rely on this implementation for any research or production purposes, as the results may not be accurate or reliable.

## Overview

AMICA (Adaptive Mixture ICA) is an advanced blind source separation algorithm that uses adaptive mixtures of independent component analyzers. This implementation provides:

- Multiple source models
- Different PDF types
- Newton optimization
- Component sharing
- Outlier rejection
- Data preprocessing (mean removal, sphering)

## Installation

```bash
# Clone the repository
git clone https://github.com/neuromechanist/pyAMICA.git
cd pyAMICA

# install the package
pip install -e .
```

## Usage

1. Create a parameter file (e.g., params.json):
```json
{
    "files": ["data1.bin", "data2.bin"],
    "num_samples": [100, 100],
    "data_dim": 64,
    "field_dim": [1000, 1000],
    "num_models": 1,
    "num_mix": 3,
    "max_iter": 2000
}
```

2. Run AMICA:
```bash
python amica_cli.py params.json --outdir results
```

## Parameters

### Required Parameters
- `files`: List of binary data files
- `num_samples`: Number of samples per file
- `data_dim`: Number of channels/dimensions
- `field_dim`: Number of samples per field for each file

### Optional Parameters

#### Model Parameters
- `num_models`: Number of models (default: 1)
- `num_mix`: Number of mixture components (default: 3)
- `num_comps`: Number of components (-1 for data_dim * num_models)
- `pdftype`: PDF type (1: Generalized Gaussian, 2: Logistic, etc.)

#### Optimization Parameters
- `max_iter`: Maximum iterations (default: 2000)
- `lrate`: Initial learning rate (default: 0.1)
- `minlrate`: Minimum learning rate (default: 1e-12)
- `lratefact`: Learning rate decay factor (default: 0.5)

#### Newton Optimization
- `do_newton`: Use Newton optimization (default: false)
- `newt_start`: Iteration to start Newton (default: 20)
- `newt_ramp`: Newton ramp length (default: 10)
- `newtrate`: Newton learning rate (default: 0.5)

#### Component Sharing
- `share_comps`: Enable component sharing (default: false)
- `comp_thresh`: Component correlation threshold (default: 0.99)
- `share_start`: Iteration to start sharing (default: 100)
- `share_int`: Sharing interval (default: 100)

#### Data Preprocessing
- `do_mean`: Remove mean (default: true)
- `do_sphere`: Perform sphering (default: true)
- `do_approx_sphere`: Use approximate sphering (default: true)
- `pcakeep`: Number of PCA components to keep (optional)
- `pcadb`: dB threshold for PCA components (optional)

#### Block Processing
- `do_opt_block`: Optimize block size (default: true)
- `block_size`: Initial block size (default: 128)
- `blk_min`: Minimum block size (default: 128)
- `blk_max`: Maximum block size (default: 1024)
- `blk_step`: Block size step (default: 128)

#### Outlier Rejection
- `do_reject`: Enable outlier rejection (default: false)
- `rejsig`: Rejection threshold in std (default: 3.0)
- `rejstart`: Iteration to start rejection (default: 2)
- `rejint`: Rejection interval (default: 3)
- `maxrej`: Maximum rejections (default: 1)

#### Output Control
- `do_history`: Save optimization history (default: false)
- `histstep`: History saving interval (default: 10)
- `writestep`: Result writing interval (default: 100)

## File Structure

- `amica.py`: Main AMICA implementation
- `amica_pdf.py`: PDF type implementations
- `amica_newton.py`: Newton optimization
- `amica_data.py`: Data loading/preprocessing
- `amica_cli.py`: Command-line interface
- `params.json`: Example parameter file

## Output Files

Results are saved in NumPy format:
- `A.npy`: Mixing matrix
- `W.npy`: Unmixing matrices
- `c.npy`: Bias terms
- `mu.npy`: Component means
- `alpha.npy`: Mixture weights
- `beta.npy`: Scale parameters
- `rho.npy`: Shape parameters
- `gm.npy`: Model weights
- `mean.npy`: Data mean
- `sphere.npy`: Sphering matrix
- `comp_list.npy`: Component assignments
- `ll.npy`: Log likelihood history
- `nd.npy`: Gradient norm history (if use_grad_norm=true)

## References

1. Palmer, J. A., Kreutz-Delgado, K., & Makeig, S. (2012). AMICA: An adaptive mixture of independent component analyzers with shared components.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
