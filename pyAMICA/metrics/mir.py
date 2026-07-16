"""Mutual Information Reduction (MIR) ICA separation-quality metric.

MIR measures how much a linear unmixing transform reduces the total mutual
information among channels: it is the (signed) difference between the sum of
per-channel differential entropies before and after unmixing, corrected by the
log-Jacobian of the transform. A well-separated decomposition drives the
unmixed channels toward independence, so MIR should be large and positive; an
orthogonal (information-preserving) transform gives MIR near zero.

Ported from ``getMIR.m`` / its nested ``getent4`` in
`bigdelys/pre_ICA_cleaning <https://github.com/bigdelys/pre_ICA_cleaning>`_
(Apache License 2.0). This is a permitted derivative under Apache-2.0's
attribution requirement; the port keeps the original's binned-entropy
estimator (including its bin-count-1 correction and eigenvalue-based
log-Jacobian) rather than substituting a different entropy estimator.

A 2023 MATLAB adaptation of ``getMIR`` (the nested ``mir()`` function in
``sccn/NEMAR-pipeline``'s ``eeg_nemar_dataqual.m``, currently dead/commented-out
code there) added an explicit sphering step ahead of this computation, but its
``mir(data, linT)`` signature only resolves via MATLAB nested-function shared
workspace (it references ``eig(W)`` and ``/N``, neither of which are its own
parameters) and it depends on a GPL-licensed ``robust_sphering_matrix``
routine, out of scope for this BSD-3-Clause project. This port instead takes
an explicit ``unmixing`` matrix: callers that want the sphere-then-unmixing
composition (e.g. wiring MIR to a fitted AMICA model) compose it themselves
before calling ``mir``.
"""

import numpy as np


def _marginal_entropies(
    u: np.ndarray, nbins: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Port of getMIR.m's nested getent4: per-row binned differential entropy."""
    n_signals, n_samples = u.shape
    if nbins is None:
        nbins = round(3 * np.log2(1 + n_samples / 10))

    entropies = np.empty(n_signals)
    variances = np.empty(n_signals)
    for i in range(n_signals):
        row = u[i]
        umin = row.min()
        umax = row.max()
        delta = (umax - umin) / nbins
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
    """Mutual Information Reduction of `unmixing` applied to `data`."""
    hx, vx = _marginal_entropies(data, nbins)
    y = unmixing @ data
    hy, vy = _marginal_entropies(y, nbins)
    n_samples = data.shape[1]
    # eigvals (not slogdet), matching getMIR.m's own switch from det to eig.
    mir_nats = float(
        np.sum(np.log(np.abs(np.linalg.eigvals(unmixing)))) + np.sum(hx) - np.sum(hy)
    )
    variance = float((np.sum(vx) + np.sum(vy)) / n_samples)
    return mir_nats, variance
