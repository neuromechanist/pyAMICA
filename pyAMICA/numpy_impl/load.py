"""Module for loading original AMICA output files."""

import numpy as np
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass
from scipy.special import gamma
from scipy.io import loadmat


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


def read_binary_file(
    filepath: Union[str, Path], dtype=np.float64, shape=None
) -> Optional[np.ndarray]:
    """Read binary file into numpy array."""
    try:
        with open(filepath, "rb") as f:
            data = np.fromfile(f, dtype=dtype)
            if shape is not None:
                data = data.reshape(shape)
            return data
    except FileNotFoundError:
        return None


def write_amicaout(
    outdir: Union[str, Path],
    *,
    gm,
    W,
    sphere,
    mean,
    c,
    alpha,
    mu,
    sbeta,
    rho,
    comp_list,
    ll,
    A=None,
    Lht=None,
    Lt=None,
):
    """Write a fitted AMICA model as the Fortran/EEGLAB binary output directory.

    Emits the raw little-endian files that :func:`loadmodout` and EEGLAB's
    ``loadmodout15.m`` read: ``gm``, ``W``, ``S``, ``mean``, ``c``, ``alpha``,
    ``mu``, ``sbeta``, ``rho``, ``comp_list`` (1-based ``int32``), ``LL`` and
    (when given) ``LLt``. This is the write counterpart of :func:`loadmodout`,
    so a pyAMICA fit (either backend) drops into an EEGLAB workflow (issue #92).

    Both backends store these arrays in the same convention, so for a single
    model the bytes are identical to the Fortran reference's ``amicaout`` files;
    for ``num_models > 1`` the per-model axis nesting is self-consistent (it
    round-trips through :func:`loadmodout`) but not byte-identical to genuine
    multi-model Fortran output (issue #27).

    Parameters
    ----------
    outdir : str or path-like
        Destination directory (created if absent).
    gm, W, sphere, mean, c, alpha, mu, sbeta, rho : array-like
        Model weights, unmixing, sphere, data mean, per-model centers and the
        mixture-density parameters (``sbeta`` is the scale, pyAMICA's ``beta``).
    comp_list : array-like of int
        0-based component ids; written 1-based to match the Fortran format.
    ll : array-like
        Per-iteration log-likelihood history.
    A : array-like, optional
        Mixing matrix. ``loadmodout15`` derives ``A`` from ``W`` and ``S`` and
        ignores this file; it is written (when given) only so pyAMICA's own
        ``load_results`` can restore ``A`` directly for the viz helpers.
    Lht : array-like of shape (num_models, n_samples), optional
        Per-model per-sample log-likelihood (Fortran ``modloglik``). Written
        together with ``Lt`` as the ``LLt`` file (issue #155); omitted (as
        before) when either is ``None``.
    Lt : array-like of shape (n_samples,), optional
        Total per-sample log-likelihood (Fortran ``loglik``).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def _w(name, arr, dtype=np.float64, order="C"):
        # Fortran dumps arrays column-major; ``order="F"`` reproduces that byte
        # layout so real EEGLAB ``loadmodout15.m`` reads the file correctly.
        np.asarray(arr, dtype=dtype).ravel(order=order).tofile(outdir / name)

    _w("gm", gm)
    if A is not None:
        _w("A", A)
    # W is byte-identical to Fortran in C order: the internal-vs-true-unmixing
    # transpose (issue #24) cancels against Fortran's column-major storage, so a
    # square W written C-order equals the Fortran/EEGLAB column-major bytes. The
    # symmetric sphere S is order-agnostic; mean/gm/LL are 1-D.
    _w("W", W)
    _w("S", sphere)
    _w("mean", mean)
    # The (num_mix, num_comps) mixture params and (num_comps, num_models) c /
    # comp_list are non-square, so their byte layout DOES depend on order: they
    # must be column-major (Fortran) for loadmodout15 to read them correctly
    # (e.g. mixture proportions per component sum to 1). Issue #92.
    _w("c", c, order="F")
    _w("alpha", alpha, order="F")
    _w("mu", mu, order="F")
    _w("sbeta", sbeta, order="F")
    _w("rho", rho, order="F")
    # comp_list is 1-based on disk (loadmodout subtracts 1 when indexing).
    _w("comp_list", np.asarray(comp_list) + 1, dtype=np.int32, order="F")
    _w("LL", np.asarray(ll))
    if (Lht is None) != (Lt is None):
        raise ValueError(
            "write_amicaout: Lht and Lt must be given together (both None to "
            f"omit LLt, or both arrays to write it); got Lht={Lht!r}, "
            f"Lt={Lt!r}."
        )
    if Lht is not None and Lt is not None:
        # Fortran writes, per timepoint, each model's log-likelihood then the
        # total (write_output, amica15.f90:2308-2333) -- a column-major
        # (num_models+1, n_samples) matrix. Stacking Lt as the last row and
        # flattening order="F" reproduces that per-timepoint sequence exactly.
        _w("LLt", np.vstack([np.atleast_2d(Lht), np.atleast_2d(Lt)]), order="F")


def loadmodout(outdir: Union[str, Path]) -> AmicaOutput:
    """Load AMICA output files from directory.

    Args:
        outdir: Path to directory containing AMICA output files

    Returns:
        AmicaOutput object containing loaded data
    """
    outdir = Path(outdir)

    # Read number of models from gm file
    gm = read_binary_file(outdir / "gm")
    num_models = len(gm) if gm is not None else 1
    if gm is None:
        print("No gm present, setting num_models to 1")
        gm = np.array([1.0])

    # Read weights
    W = read_binary_file(outdir / "W")
    if W is None:
        raise FileNotFoundError("No W present, cannot continue")
    nw2 = len(W) // num_models
    nw = int(np.sqrt(nw2))
    W = W.reshape(nw, nw, num_models)

    # Read mean and sphere
    mn = read_binary_file(outdir / "mean")
    if mn is None:
        print("No mean present, setting mean to zero")
        nx = nw
        mn = np.zeros(nx)
    else:
        nx = len(mn)

    S = read_binary_file(outdir / "S")
    if S is None:
        if mn is not None:
            S = np.eye(nx)
        else:
            raise FileNotFoundError("No sphere or mean present, cannot continue")
    else:
        if len(S.shape) == 1:
            S = S.reshape(nx, nx)

    # Read component list
    comp_list = read_binary_file(outdir / "comp_list", dtype=np.int32)
    if comp_list is not None:
        expected = nw * num_models
        # The Fortran writer opens this file with the record length formula
        # used for real*8 arrays (recl=2*nbyte*nw*num_models), which is 2x
        # too large for int32 comp_list data; the direct-access write then
        # zero-pads the record to that length, so a well-formed file is
        # either exactly `expected` values or `2*expected` values with the
        # trailing half all zero. Only accept those two shapes; anything
        # else is a corrupt or unexpected file and should fail loudly
        # rather than silently truncating.
        if comp_list.size == expected:
            comp_list = comp_list.reshape(nw, num_models, order="F")
        elif comp_list.size == 2 * expected and not np.any(comp_list[expected:]):
            comp_list = comp_list[:expected].reshape(nw, num_models, order="F")
        else:
            raise ValueError(
                f"comp_list has {comp_list.size} elements; expected {expected} "
                f"(nw={nw} * num_models={num_models}) or {2 * expected} with a "
                f"zero-padded tail (Fortran recl=2*nbyte*nw*num_models). File may "
                f"be corrupt or from an incompatible run."
            )

    # Read log likelihoods. Stored column-major (Fortran writes, per timepoint,
    # each model's log-likelihood then the total), so reshape order="F" (matches
    # loadmodout15.m's `reshape(LLt, num_models+1, N)` and write_amicaout below).
    LLt = read_binary_file(outdir / "LLt")
    if LLt is not None:
        LLt = LLt.reshape(num_models + 1, -1, order="F")
        Lht = LLt[:num_models]
        Lt = LLt[num_models]
    else:
        print("LLt not present")
        Lht = Lt = None

    LL = read_binary_file(outdir / "LL")
    if LL is None:
        LL = np.array([0])

    # Read model parameters
    c = read_binary_file(outdir / "c")
    if c is None:
        c = np.zeros((nw, num_models))
    else:
        c = c.reshape(nw, num_models, order="F")

    # Read mixture parameters. Stored column-major (Fortran), so reshape order="F"
    # (matches loadmodout15.m and the write_amicaout writer); a C-order read would
    # scramble the (num_mix, num_comps) layout. Issue #92.
    alpha_tmp = read_binary_file(outdir / "alpha")
    if alpha_tmp is not None:
        num_mix = len(alpha_tmp) // (nw * num_models)
        alpha_tmp = alpha_tmp.reshape(num_mix, nw * num_models, order="F")
        alpha = np.zeros((num_mix, nw, num_models))
        for h in range(num_models):
            for i in range(nw):
                alpha[:, i, h] = (
                    alpha_tmp[:, comp_list[i, h] - 1]
                    if comp_list is not None
                    else alpha_tmp[:, i + h * nw]
                )
    else:
        num_mix = 1
        alpha = np.ones((num_mix, nw, num_models))

    # Read mu, sbeta, rho
    mu_tmp = read_binary_file(outdir / "mu")
    if mu_tmp is not None:
        mu_tmp = mu_tmp.reshape(num_mix, nw * num_models, order="F")
        mu = np.zeros((num_mix, nw, num_models))
        for h in range(num_models):
            for i in range(nw):
                mu[:, i, h] = (
                    mu_tmp[:, comp_list[i, h] - 1]
                    if comp_list is not None
                    else mu_tmp[:, i + h * nw]
                )
    else:
        mu = np.zeros((num_mix, nw, num_models))

    sbeta_tmp = read_binary_file(outdir / "sbeta")
    if sbeta_tmp is not None:
        sbeta_tmp = sbeta_tmp.reshape(num_mix, nw * num_models)
        sbeta = np.zeros((num_mix, nw, num_models))
        for h in range(num_models):
            for i in range(nw):
                sbeta[:, i, h] = (
                    sbeta_tmp[:, comp_list[i, h] - 1]
                    if comp_list is not None
                    else sbeta_tmp[:, i + h * nw]
                )
    else:
        sbeta = np.ones((num_mix, nw, num_models))

    rho_tmp = read_binary_file(outdir / "rho")
    if rho_tmp is not None:
        rho_tmp = rho_tmp.reshape(num_mix, nw * num_models)
        rho = np.zeros((num_mix, nw, num_models))
        for h in range(num_models):
            for i in range(nw):
                rho[:, i, h] = (
                    rho_tmp[:, comp_list[i, h] - 1]
                    if comp_list is not None
                    else rho_tmp[:, i + h * nw]
                )
    else:
        rho = 2 * np.ones((num_mix, nw, num_models))

    # Read weight change history
    nd = read_binary_file(outdir / "nd")
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
                alpha[mix_idx, i, h]
                * (
                    mu[mix_idx, i, h] ** 2
                    + (gamma(3 / r) / gamma(1 / r)) / sbeta[mix_idx, i, h] ** 2
                )
            )
            svar[i, h] *= np.linalg.norm(A[:, i, h]) ** 2

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
        v=v,
    )


def read_eeglab_set_metadata(path: Union[str, Path]) -> dict:
    """Read sample rate, channel positions, and labels from an EEGLAB ``.set`` file.

    pyAMICA has no notion of sampling rate or channel geometry anywhere in its
    own data structures (`AmicaOutput`, `load_data_file`, `load_results` all
    lack it); the top-level `pyAMICA.viz` plots need both for a seconds x-axis
    and scalp topography maps. This is a minimal `scipy.io.loadmat` reader for
    exactly those three fields, not a general EEGLAB-format loader.

    Parameters
    ----------
    path : str or path-like
        Path to an EEGLAB ``.set`` file (MATLAB v5/v7 format; ``-v7.3`` ``.set``
        files are HDF5 and are not supported by `scipy.io.loadmat`).

    Returns
    -------
    dict
        ``{"srate": float, "positions": ndarray of shape (n_channels, 3),
        "labels": list[str]}``. ``positions`` columns are EEGLAB's
        ``chanlocs`` ``X``, ``Y``, ``Z`` fields.

    Raises
    ------
    ValueError
        If any channel is missing ``X``/``Y``/``Z`` position data (EEGLAB
        itself allows unlocalized channels; this minimal reader does not,
        since scalp-map plotting cannot use them).
    """
    mat = loadmat(str(path), struct_as_record=False, squeeze_me=True)
    eeg = mat["EEG"]
    chanlocs = np.atleast_1d(eeg.chanlocs)
    if len(chanlocs) == 0:
        # Guard explicitly: the non-finite check below uses np.any(np.isnan(...)),
        # which is False on an empty array, so a .set with no chanlocs would sail
        # through and return an empty (0, 3) positions array. That only surfaces
        # later as a confusing channel-count mismatch in a caller, far from the
        # actual cause.
        raise ValueError(
            f"read_eeglab_set_metadata: {path} has no channel locations "
            "(EEG.chanlocs is empty); scalp topographies need per-channel X/Y/Z."
        )

    positions = np.full((len(chanlocs), 3), np.nan)
    labels = []
    for i, ch in enumerate(chanlocs):
        labels.append(str(ch.labels))
        for j, axis in enumerate(("X", "Y", "Z")):
            val = getattr(ch, axis)
            if isinstance(val, np.ndarray) and val.size == 0:
                continue  # empty MATLAB [] on an unlocalized channel
            positions[i, j] = float(val)

    if np.any(np.isnan(positions)):
        missing = [
            labels[i] for i in range(len(labels)) if np.any(np.isnan(positions[i]))
        ]
        raise ValueError(
            f"read_eeglab_set_metadata: channel(s) {missing} are missing "
            "X/Y/Z position data; this reader requires all channels to be "
            "localized."
        )

    return {"srate": float(eeg.srate), "positions": positions, "labels": labels}
