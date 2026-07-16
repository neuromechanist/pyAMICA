"""Mutual Information Reduction (MIR) ICA separation-quality metric.

MIR measures how much a linear unmixing transform reduces the total mutual
information among channels: it is the (signed) difference between the sum of
per-channel differential entropies before and after unmixing, corrected by the
log-Jacobian of the transform. A well-separated decomposition drives the
unmixed channels toward independence, so MIR should be large and positive.
The identity transform (or any scalar multiple of it) gives MIR exactly zero,
since the bin-partition estimator below is scale-invariant; a generic
non-identity rotation of dependent, non-Gaussian data need not (that
directional sensitivity is the entire mechanism ICA exploits).

Ported from ``getMIR.m`` / its nested ``getent4`` in
`bigdelys/pre_ICA_cleaning
<https://github.com/bigdelys/pre_ICA_cleaning/blob/b6034f03889a6a418968ee119123f3df55251957/getMIR.m>`_
(Apache License 2.0, full text reproduced in ``THIRD_PARTY_NOTICES.md`` per
that license's Section 4(a)). This is a permitted derivative under
Apache-2.0; the port keeps the original's binned-entropy estimator (including
its bin-count-1 correction and eigenvalue-based log-Jacobian) rather than
substituting a different entropy estimator.

A 2023 MATLAB adaptation of ``getMIR`` (the nested ``mir()`` function in
``sccn/NEMAR-pipeline``'s ``eeg_nemar_dataqual.m``, currently dead/commented-out
code there) added an explicit sphering step ahead of this computation, but its
``mir(data, linT)`` references ``eig(W)`` and ``/N``, neither of which are its
own parameters nor ever assigned anywhere in the enclosing function -- it would
raise an undefined-variable error if uncommented, which is why it's dead code.
It also depends on a ``robust_sphering_matrix`` routine that itself calls
GPL-licensed helpers (``block_geometric_median``, ``hlp_memfree``), out of
scope for this BSD-3-Clause project. This port instead takes an explicit
``unmixing`` matrix: callers that want the sphere-then-unmixing composition
(e.g. wiring MIR to a fitted AMICA model) compose it themselves before calling
``mir``.

MIR follows the separation-quality-metric approach introduced by Delorme,
Palmer, Onton, Oostenveld, and Makeig (2012), "Independent EEG sources are
dipolar", PLoS ONE (this repo's ``paper.bib`` key ``delorme2012independent``).
"""

import numpy as np

from ._common import resolve_nbins, validate_signal_matrix

_MIN_ABS_EIGENVALUE = 1e-10


def _marginal_entropies(
    u: np.ndarray, nbins: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Port of getMIR.m's nested getent4: per-row binned differential entropy."""
    validate_signal_matrix(u)
    n_signals, n_samples = u.shape
    nbins = resolve_nbins(n_samples, nbins)

    entropies = np.empty(n_signals)
    variances = np.empty(n_signals)
    for i in range(n_signals):
        row = u[i]
        umin = row.min()
        umax = row.max()
        delta = (umax - umin) / nbins
        # MATLAB's round() is half-away-from-zero; np.round is half-to-even.
        # Immaterial here since exact bin-boundary ties have probability zero
        # on continuous real EEG data.
        binned = 1 + np.round((nbins - 1) * (row - umin) / (umax - umin))
        # np.unique's counts, unlike np.histogram, only ever report occupied
        # bins, so log(p) below never sees a zero-count bin.
        counts = np.unique(binned, return_counts=True)[1]
        p = counts / n_samples
        h = -np.sum(p * np.log(p))
        # Variance uses the uncorrected h, matching getent4's statement order.
        variances[i] = np.sum(p * np.log(p) ** 2) - h**2
        entropies[i] = h + (nbins - 1) / (2 * n_samples) + np.log(delta)
    return entropies, variances


def mir(
    unmixing: np.ndarray, data: np.ndarray, nbins: int | None = None
) -> tuple[float, float]:
    """Mutual Information Reduction of `unmixing` applied to `data`, in nats.

    Parameters
    ----------
    unmixing : np.ndarray
        Square (n, n) linear transform applied to `data` (e.g. an ICA
        unmixing matrix, optionally composed with a sphering matrix).
    data : np.ndarray
        (n, N) data matrix `unmixing` is applied to.
    nbins : int, optional
        Histogram bin count for the marginal-entropy estimator; see
        `_marginal_entropies`. Defaults to ``round(3 * log2(1 + N/10))``.

    Returns
    -------
    mir_nats : float
        Mutual information (in nats) removed by `unmixing` from `data`.
    variance : float
        Variance of the MIR estimate.

    Raises
    ------
    ValueError
        If `unmixing` is singular or near-singular (the log-Jacobian term is
        undefined for a non-invertible transform), or if `data`/`unmixing`
        contain non-finite values or a constant channel.
    """
    eigvals = np.linalg.eigvals(unmixing)
    min_abs_eig = np.min(np.abs(eigvals))
    if min_abs_eig < _MIN_ABS_EIGENVALUE:
        raise ValueError(
            f"mir(): unmixing matrix is singular or near-singular (smallest "
            f"|eigenvalue| = {min_abs_eig:.3e}); the log-Jacobian term is "
            "undefined for a non-invertible transform. Verify `unmixing` is "
            "full rank (e.g. check for duplicated/collinear components from "
            "an unconverged ICA fit)."
        )
    hx, vx = _marginal_entropies(data, nbins)
    y = unmixing @ data
    hy, vy = _marginal_entropies(y, nbins)
    n_samples = data.shape[1]
    # eigvals (not slogdet), matching getMIR.m's own switch from det to eig.
    mir_nats = float(np.sum(np.log(np.abs(eigvals))) + np.sum(hx) - np.sum(hy))
    variance = float((np.sum(vx) + np.sum(vy)) / n_samples)
    return mir_nats, variance
