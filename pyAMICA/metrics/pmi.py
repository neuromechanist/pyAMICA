"""Pairwise Mutual Information (PMI) ICA separation-quality metric.

PMI measures the mutual information between every pair of AMICA components
(or raw channels): for well-separated ICA components, each pair should show
low pairwise mutual information, since a good decomposition drives the
components toward mutual independence. High residual pairwise MI between two
"independent" components indicates incomplete separation or a shared source
that ICA failed to isolate. `block_diagonal_order` complements the pairwise
matrix by finding a component ordering that clusters high-MI pairs near the
diagonal, useful for visualizing residual structure.

This is an original, clean-room implementation of the general
histogram-binned-entropy-with-Miller-Madow-bias-correction approach described
in issue #135, following the separation-quality-metric approach introduced by
Delorme, Palmer, Onton, Oostenveld, and Makeig (2012), "Independent EEG
sources are dipolar", PLoS ONE (this repo's ``paper.bib`` key
``delorme2012independent``), and the pairwise-MI diagnostic approach used by
Palmer, Balkan, Delorme, and Miyakoshi in their AMICA-related work. It was
NOT derived from, and does not reuse any code or structure from,
``sccn/postAmicaUtility``'s GPL-2.0-or-later MATLAB source
(``minfojp.m``/``get_mi.m``/``arrminf2.m``); that source was not consulted at
any point during this implementation, and the entropy estimator, binning
scheme, and greedy ordering algorithm below are derived solely from standard
statistics (histogram-based plug-in entropy estimation, the Miller-Madow bias
correction) and the algorithm description in issue #135.
"""

import numpy as np

from ._common import resolve_nbins, validate_signal_matrix


def _binned_entropy_from_counts(counts: np.ndarray, n_samples: int) -> float:
    """Miller-Madow-corrected plug-in entropy from occupied-bin counts."""
    counts = counts[counts > 0]
    p = counts / n_samples
    h_plugin = -np.sum(p * np.log(p))
    m = counts.size
    return float(h_plugin + (m - 1) / (2 * n_samples))


def pairwise_mi(sources: np.ndarray, nbins: int | None = None) -> np.ndarray:
    """Symmetric (n, n) pairwise mutual-information matrix, in nats.

    Parameters
    ----------
    sources : np.ndarray
        (n_components, n_samples) signal matrix (e.g. ICA source activations
        or raw channels).
    nbins : int, optional
        Histogram bin count per axis of the joint 2-D histogram. Defaults to
        ``round(3 * log2(1 + N/10))``; see `pyAMICA.metrics._common.resolve_nbins`.

    Returns
    -------
    mi_matrix : np.ndarray
        (n, n) symmetric matrix; `mi_matrix[i, j]` is the mutual information
        (nats) between components `i` and `j`. The diagonal holds each
        component's own entropy (`mi_matrix[i, i] ~= H(X_i)`).

    Raises
    ------
    ValueError
        If `sources` contains non-finite values or a constant channel.
    """
    validate_signal_matrix(sources)
    n, n_samples = sources.shape
    nbins = resolve_nbins(n_samples, nbins)

    mi_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            joint_counts, _, _ = np.histogram2d(sources[i], sources[j], bins=nbins)
            marginal_x = joint_counts.sum(axis=1)
            marginal_y = joint_counts.sum(axis=0)
            h_x = _binned_entropy_from_counts(marginal_x, n_samples)
            h_y = _binned_entropy_from_counts(marginal_y, n_samples)
            h_xy = _binned_entropy_from_counts(joint_counts.ravel(), n_samples)
            mi = h_x + h_y - h_xy
            mi_matrix[i, j] = mi
            mi_matrix[j, i] = mi
    return mi_matrix


def block_diagonal_order(mi_matrix: np.ndarray) -> np.ndarray:
    """Greedy nearest-neighbor-chain permutation clustering high-MI pairs near the diagonal.

    Parameters
    ----------
    mi_matrix : np.ndarray
        (n, n) symmetric mutual-information matrix, e.g. from `pairwise_mi`.

    Returns
    -------
    order : np.ndarray
        Length-`n` permutation of `0..n-1`. Reordering both axes of
        `mi_matrix` by `order` tends to cluster high-MI pairs adjacent to the
        diagonal.
    """
    n = mi_matrix.shape[0]
    if n <= 1:
        return np.arange(n)

    off_diag = mi_matrix.copy()
    np.fill_diagonal(off_diag, -np.inf)
    i, j = np.unravel_index(np.argmax(off_diag), off_diag.shape)
    chain = [int(i), int(j)]
    remaining = set(range(n)) - {int(i), int(j)}

    while remaining:
        best_value = -np.inf
        best_component = -1
        best_at_start = False
        for c in remaining:
            value_start = mi_matrix[chain[0], c]
            if value_start > best_value:
                best_value = value_start
                best_component = c
                best_at_start = True
            value_end = mi_matrix[chain[-1], c]
            if value_end > best_value:
                best_value = value_end
                best_component = c
                best_at_start = False
        if best_at_start:
            chain.insert(0, best_component)
        else:
            chain.append(best_component)
        remaining.remove(best_component)

    return np.array(chain)
