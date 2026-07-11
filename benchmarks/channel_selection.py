"""Spatially-distributed channel subset selection (issue #91).

The dimension sweeps reduce channel count by slicing ``full[:nc]`` -- the *first*
nc electrodes in file order. For a montage like ds002718 (EEG001..EEG070) the
first 16/32/48 electrodes are a spatial cluster (one scalp region), not a
whole-head cap, so the reduced-channel decompositions and their IC scalp maps are
not physically meaningful reduced montages.

This module picks nc electrodes **evenly distributed across the scalp** via greedy
farthest-point (k-center) sampling over the real 3D electrode positions from a
BIDS ``electrodes.tsv``, so every subset is a proper whole-head montage. Selection
is deterministic (seeded from the electrode nearest the montage centroid).
"""

from __future__ import annotations

import csv

import numpy as np


def read_electrode_positions(tsv_path):
    """Parse a BIDS ``electrodes.tsv`` into ordered names and 3D positions.

    Rows whose ``x``/``y``/``z`` are non-numeric (e.g. ``n/a`` for unlocalized or
    non-EEG channels) are skipped.

    Parameters
    ----------
    tsv_path : str or path-like
        Path to a BIDS ``electrodes.tsv`` (tab-separated, columns ``name x y z``).

    Returns
    -------
    names : list of str
        Electrode names, in file order, for the localized channels.
    positions : ndarray of shape (n_localized, 3)
        Corresponding 3D coordinates.
    """
    names, pos = [], []
    with open(tsv_path) as f:
        for row in csv.DictReader(f, delimiter="\t"):
            try:
                xyz = [float(row["x"]), float(row["y"]), float(row["z"])]
            except (ValueError, KeyError, TypeError):
                continue
            names.append(row["name"])
            pos.append(xyz)
    return names, np.asarray(pos, dtype=float)


def positions_for_channels(tsv_path, n_channels, name_fmt="EEG{:03d}"):
    """Positions aligned to data-channel index for the first ``n_channels``.

    Data channel ``i`` (0-based) is assumed to be electrode ``name_fmt.format(i+1)``
    (``EEG001``, ``EEG002``, ...), matching the benchmark data layout. Channels
    without a localized position are filled with NaN.

    Returns
    -------
    ndarray of shape (n_channels, 3)
        Row ``i`` is the position of data channel ``i`` (NaN if unlocalized).
    """
    names, pos = read_electrode_positions(tsv_path)
    lookup = dict(zip(names, pos))
    out = np.full((n_channels, 3), np.nan, dtype=float)
    for i in range(n_channels):
        p = lookup.get(name_fmt.format(i + 1))
        if p is not None:
            out[i] = p
    return out


def select_distributed_channels(positions, n):
    """Greedy farthest-point (k-center) selection of ``n`` spread-out channels.

    Starting from the channel nearest the centroid, repeatedly add the channel
    that is farthest (in the max-min sense) from the already-selected set. This
    yields a spatially distributed, whole-head subset rather than a cluster.

    Parameters
    ----------
    positions : ndarray of shape (n_channels, 3)
        Positions aligned to data-channel index; rows may be NaN (unlocalized),
        which are excluded from selection.
    n : int
        Number of channels to select.

    Returns
    -------
    ndarray of int
        Sorted data-channel indices of the selected channels. If ``n`` is at least
        the number of localized channels, all localized indices are returned.
    """
    positions = np.asarray(positions, dtype=float)
    localized = np.where(np.isfinite(positions).all(axis=1))[0]
    pts = positions[localized]
    if n >= len(localized):
        return localized
    centroid = pts.mean(axis=0)
    first = int(np.argmin(((pts - centroid) ** 2).sum(axis=1)))
    selected = [first]
    # running min distance from every point to the selected set
    dist = np.sqrt(((pts - pts[first]) ** 2).sum(axis=1))
    while len(selected) < n:
        nxt = int(np.argmax(dist))
        selected.append(nxt)
        dist = np.minimum(dist, np.sqrt(((pts - pts[nxt]) ** 2).sum(axis=1)))
    return np.sort(localized[np.asarray(selected, dtype=int)])
