"""Module for loading original AMICA output files."""

import numpy as np
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass
from scipy.special import gamma


@dataclass
class AmicaOutput:
    """Class to hold AMICA output data."""
    num_models: int
    mod_prob: np.ndarray
    W: np.ndarray  # unmixing weights (post-sphering)
    num_pcs: int
    data_dim: int
    data_mean: np.ndarray  # mean of the raw data
    S: np.ndarray  # sphering matrix
    comp_list: Optional[np.ndarray]  # unique component ids if sharing
    Lht: Optional[np.ndarray]  # model posterior log likelihood
    Lt: Optional[np.ndarray]  # total posterior likelihood
    LL: np.ndarray  # likelihood at each iteration
    c: np.ndarray  # model centers
    alpha: np.ndarray  # source mixture proportions
    mu: np.ndarray  # source mixture means
    sbeta: np.ndarray  # source mixture scales
    rho: np.ndarray  # source mixture shapes
    nd: Optional[np.ndarray]  # weight change history by component
    svar: np.ndarray  # data variance explained by comp
    A: np.ndarray  # model component matrices
    origord: np.ndarray  # original order prior to var order
    v: Optional[np.ndarray]  # log10 posterior model odds


def read_binary_file(filepath: Union[str, Path], dtype=np.float64, shape=None) -> Optional[np.ndarray]:
    """Read binary file into numpy array."""
    try:
        with open(filepath, 'rb') as f:
            data = np.fromfile(f, dtype=dtype)
            if shape is not None:
                data = data.reshape(shape)
            return data
    except FileNotFoundError:
        return None


def loadmodout(outdir: Union[str, Path]) -> AmicaOutput:
    """Load AMICA output files from directory.

    Args:
        outdir: Path to directory containing AMICA output files

    Returns:
        AmicaOutput object containing loaded data
    """
    outdir = Path(outdir)

    # Read number of models from gm file
    gm = read_binary_file(outdir / 'gm')
    num_models = len(gm) if gm is not None else 1
    if gm is None:
        print('No gm present, setting num_models to 1')
        gm = np.array([1.0])

    # Read weights
    W = read_binary_file(outdir / 'W')
    if W is None:
        raise FileNotFoundError('No W present, cannot continue')
    nw2 = len(W) // num_models
    nw = int(np.sqrt(nw2))
    W = W.reshape(nw, nw, num_models)

    # Read mean and sphere
    mn = read_binary_file(outdir / 'mean')
    if mn is None:
        print('No mean present, setting mean to zero')
        nx = nw
        mn = np.zeros(nx)
    else:
        nx = len(mn)

    S = read_binary_file(outdir / 'S')
    if S is None:
        if mn is not None:
            S = np.eye(nx)
        else:
            raise FileNotFoundError('No sphere or mean present, cannot continue')
    else:
        if len(S.shape) == 1:
            S = S.reshape(nx, nx)

    # Read component list
    comp_list = read_binary_file(outdir / 'comp_list', dtype=np.int32)
    if comp_list is not None:
        comp_list = comp_list.reshape(nw, num_models)

    # Read log likelihoods
    LLt = read_binary_file(outdir / 'LLt')
    if LLt is not None:
        LLt = LLt.reshape(num_models + 1, -1)
        Lht = LLt[:num_models]
        Lt = LLt[num_models]
    else:
        print('LLt not present')
        Lht = Lt = None

    LL = read_binary_file(outdir / 'LL') or np.array([0])

    # Read model parameters
    c = read_binary_file(outdir / 'c')
    if c is None:
        c = np.zeros((nw, num_models))
    else:
        c = c.reshape(nw, num_models)

    # Read mixture parameters
    alpha_tmp = read_binary_file(outdir / 'alpha')
    if alpha_tmp is not None:
        num_mix = len(alpha_tmp) // (nw * num_models)
        alpha_tmp = alpha_tmp.reshape(num_mix, nw * num_models)
        alpha = np.zeros((num_mix, nw, num_models))
        for h in range(num_models):
            for i in range(nw):
                alpha[:, i, h] = (alpha_tmp[:, comp_list[i, h] - 1] if comp_list is not None
                                  else alpha_tmp[:, i + h * nw])
    else:
        num_mix = 1
        alpha = np.ones((num_mix, nw, num_models))

    # Read mu, sbeta, rho
    mu_tmp = read_binary_file(outdir / 'mu')
    if mu_tmp is not None:
        mu_tmp = mu_tmp.reshape(num_mix, nw * num_models)
        mu = np.zeros((num_mix, nw, num_models))
        for h in range(num_models):
            for i in range(nw):
                mu[:, i, h] = (mu_tmp[:, comp_list[i, h] - 1] if comp_list is not None
                               else mu_tmp[:, i + h * nw])
    else:
        mu = np.zeros((num_mix, nw, num_models))

    sbeta_tmp = read_binary_file(outdir / 'sbeta')
    if sbeta_tmp is not None:
        sbeta_tmp = sbeta_tmp.reshape(num_mix, nw * num_models)
        sbeta = np.zeros((num_mix, nw, num_models))
        for h in range(num_models):
            for i in range(nw):
                sbeta[:, i, h] = (sbeta_tmp[:, comp_list[i, h] - 1] if comp_list is not None
                                  else sbeta_tmp[:, i + h * nw])
    else:
        sbeta = np.ones((num_mix, nw, num_models))

    rho_tmp = read_binary_file(outdir / 'rho')
    if rho_tmp is not None:
        rho_tmp = rho_tmp.reshape(num_mix, nw * num_models)
        rho = np.zeros((num_mix, nw, num_models))
        for h in range(num_models):
            for i in range(nw):
                rho[:, i, h] = (rho_tmp[:, comp_list[i, h] - 1] if comp_list is not None
                                else rho_tmp[:, i + h * nw])
    else:
        rho = 2 * np.ones((num_mix, nw, num_models))

    # Read weight change history
    nd = read_binary_file(outdir / 'nd')
    if nd is not None:
        max_iter = len(nd) // (nw * num_models)
        nd = nd.reshape(max_iter, nw, num_models)

    # Sort models by probability
    gm_ord = np.argsort(gm)[::-1]
    mod_prob = gm[gm_ord]
    W = W[:, :, gm_ord]
    c = c[:, gm_ord]
    alpha = alpha[:, :, gm_ord]
    mu = mu[:, :, gm_ord]
    sbeta = sbeta[:, :, gm_ord]
    rho = rho[:, :, gm_ord]

    if Lht is not None:
        Lht = Lht[gm_ord]
    if comp_list is not None:
        comp_list = comp_list[:, gm_ord]
    if nd is not None:
        nd = nd[:, :, gm_ord]

    # Calculate model matrices and variances
    A = np.zeros((nx, nw, num_models))
    svar = np.zeros((nw, num_models))
    origord = np.zeros((nw, num_models), dtype=int)

    for h in range(num_models):
        A[:, :, h] = np.linalg.pinv(W[:, :, h] @ S[0:nw])

        # Calculate source variances
        num_mix_used = np.sum(alpha[:, 0, 0] > 0)
        for i in range(nw):
            mix_idx = slice(0, num_mix_used)
            r = rho[mix_idx, i, h]
            svar[i, h] = np.sum(
                alpha[mix_idx, i, h] * (
                    mu[mix_idx, i, h]**2 + (gamma(3 / r) / gamma(1 / r)) / sbeta[mix_idx, i, h]**2
                )
            )
            svar[i, h] *= np.linalg.norm(A[:, i, h])**2

        # Sort by variance
        idx = np.argsort(svar[:, h])[::-1]
        origord[:, h] = idx

        # Reorder components
        A[:, :, h] = A[:, idx, h]
        W[:, :, h] = W[idx, :, h]
        alpha[:, :, h] = alpha[:, idx, h]
        mu[:, :, h] = mu[:, idx, h]
        sbeta[:, :, h] = sbeta[:, idx, h]
        rho[:, :, h] = rho[:, idx, h]
        if comp_list is not None:
            comp_list[:, h] = comp_list[idx, h]
        if nd is not None:
            nd[:, :, h] = nd[:, idx, h]

    # Calculate model odds if possible
    v = None
    if Lht is not None and Lt is not None:
        v = 0.4343 * (Lht - Lt)  # log10 odds

    # Normalize components
    for h in range(num_models):
        for i in range(nw):
            na = np.linalg.norm(A[:, i, h])
            A[:, i, h] /= na
            W[i, :, h] *= na
            mu[:, i, h] *= na
            sbeta[:, i, h] /= na

    return AmicaOutput(
        num_models=num_models,
        mod_prob=mod_prob,
        W=W,
        num_pcs=nw,
        data_dim=nx,
        data_mean=mn,
        S=S,
        comp_list=comp_list,
        Lht=Lht,
        Lt=Lt,
        LL=LL,
        c=c,
        alpha=alpha,
        mu=mu,
        sbeta=sbeta,
        rho=rho,
        nd=nd,
        svar=svar,
        A=A,
        origord=origord,
        v=v
    )
