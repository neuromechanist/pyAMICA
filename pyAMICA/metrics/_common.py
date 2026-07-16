"""Shared input-validation helpers for the metrics package (private)."""

import numpy as np


def validate_signal_matrix(u: np.ndarray) -> None:
    """Raise ValueError if `u` has non-finite values or a constant row."""
    if not np.all(np.isfinite(u)):
        raise ValueError(
            "validate_signal_matrix: input contains non-finite (NaN/Inf) "
            "values; differential entropy is undefined for non-finite samples."
        )
    n_samples = u.shape[1]
    for i, row in enumerate(u):
        umin = row.min()
        umax = row.max()
        if umax == umin:
            raise ValueError(
                f"validate_signal_matrix: channel {i} is constant (all "
                f"{n_samples} samples == {umin!r}); differential entropy is "
                "undefined for a zero-variance channel."
            )


def resolve_nbins(n_samples: int, nbins: int | None) -> int:
    """Default nbins = round(3*log2(1+N/10)) if nbins is None; raise if < 1."""
    if nbins is None:
        nbins = round(3 * np.log2(1 + n_samples / 10))
    if nbins < 1:
        raise ValueError(
            f"resolve_nbins: nbins={nbins} is not >= 1 (n_samples="
            f"{n_samples} is too small for the default nbins formula; pass "
            "an explicit nbins or supply more samples)."
        )
    return nbins


def validate_mi_matrix(mi_matrix: np.ndarray) -> None:
    """Raise ValueError if `mi_matrix` isn't square, finite, and symmetric."""
    if mi_matrix.ndim != 2 or mi_matrix.shape[0] != mi_matrix.shape[1]:
        raise ValueError(
            f"validate_mi_matrix: mi_matrix must be square 2-D, got shape "
            f"{mi_matrix.shape}."
        )
    if not np.all(np.isfinite(mi_matrix)):
        raise ValueError(
            "validate_mi_matrix: mi_matrix contains non-finite (NaN/Inf) "
            "values; cannot rank affinities against an undefined value."
        )
    if not np.allclose(mi_matrix, mi_matrix.T):
        raise ValueError("validate_mi_matrix: mi_matrix must be symmetric.")
