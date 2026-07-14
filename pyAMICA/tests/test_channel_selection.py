"""Tests for spatially-distributed channel selection (issue #91).

Uses real electrode coordinates (ds002718 sub-002 BIDS electrodes.tsv, committed
as a fixture) -- never synthetic geometry.
"""

import importlib.util
from itertools import combinations
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_FIXTURE = (
    Path(__file__).resolve().parent / "fixtures" / "ds002718_sub-002_electrodes.tsv"
)


def _load_channel_selection():
    path = _REPO / "benchmarks" / "channel_selection.py"
    spec = importlib.util.spec_from_file_location("channel_selection", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


cs = _load_channel_selection()


def _min_pairwise(pts):
    return min(
        float(np.linalg.norm(pts[i] - pts[j]))
        for i, j in combinations(range(len(pts)), 2)
    )


def test_read_electrode_positions():
    names, pos = cs.read_electrode_positions(_FIXTURE)
    assert len(names) == pos.shape[0]
    assert pos.shape[1] == 3
    assert "EEG001" in names
    assert np.isfinite(pos).all()
    # ds002718 sub-002 fixture: 74 electrode rows, 70 with a localized position;
    # the 4 external channels (n/a coords) are skipped.
    assert len(names) == 70


def test_positions_for_channels_alignment():
    pos = cs.positions_for_channels(_FIXTURE, 70)
    assert pos.shape == (70, 3)
    # ds002718 sub-002: EEG061-064 are external channels with no scalp position
    # (n/a), so 66 of the 70 are localized.
    localized = np.isfinite(pos).all(axis=1)
    assert localized.sum() == 66
    assert not localized[[60, 61, 62, 63]].any()  # EEG061-064 unlocalized


def test_selection_deterministic_unique_sorted():
    pos = cs.positions_for_channels(_FIXTURE, 70)
    idx1 = cs.select_distributed_channels(pos, 16)
    idx2 = cs.select_distributed_channels(pos, 16)
    assert len(idx1) == 16
    assert len(set(idx1.tolist())) == 16  # unique
    assert np.array_equal(idx1, idx2)  # deterministic
    assert np.array_equal(idx1, np.sort(idx1))  # sorted ascending
    assert idx1.max() < 70  # valid channel indices


def test_distributed_more_spread_than_first_n():
    """The whole point of #91: a distributed subset covers the head better than
    the first-N cluster, i.e. its nearest-pair distance is larger."""
    pos = cs.positions_for_channels(_FIXTURE, 70)
    for n in (16, 32, 48):
        idx = cs.select_distributed_channels(pos, n)
        first_n = np.arange(n)
        assert _min_pairwise(pos[idx]) > _min_pairwise(pos[first_n])


def test_n_ge_localized_returns_all_localized():
    pos = cs.positions_for_channels(_FIXTURE, 70)
    # only 66 of 70 are localized; asking for >= that returns the localized set
    idx = cs.select_distributed_channels(pos, 70)
    assert len(idx) == 66
    # the 4 unlocalized external channels are excluded
    assert not (set(idx.tolist()) & {60, 61, 62, 63})


def test_unlocalized_rows_excluded():
    pos = cs.positions_for_channels(_FIXTURE, 70)
    pos[5] = np.nan  # mark channel 5 as unlocalized
    idx = cs.select_distributed_channels(pos, 16)
    assert 5 not in idx.tolist()
    assert len(idx) == 16


def test_n_zero_returns_empty():
    pos = cs.positions_for_channels(_FIXTURE, 70)
    idx = cs.select_distributed_channels(pos, 0)
    assert len(idx) == 0


def test_seed_is_centroid_nearest():
    """The greedy selection seeds from the localized channel nearest the montage
    centroid; that seed must appear in any non-trivial subset."""
    pos = cs.positions_for_channels(_FIXTURE, 70)
    localized = np.where(np.isfinite(pos).all(axis=1))[0]
    pts = pos[localized]
    centroid = pts.mean(axis=0)
    seed = int(localized[np.argmin(((pts - centroid) ** 2).sum(axis=1))])
    assert seed in cs.select_distributed_channels(pos, 8).tolist()


def test_coincident_coordinates_stay_unique():
    """Two channels at the exact same position must not both collapse onto one
    index: the subset stays unique even when candidates tie at distance 0."""
    pos = cs.positions_for_channels(_FIXTURE, 70)
    pos[1] = pos[0]  # duplicate a real electrode position onto another channel
    idx = cs.select_distributed_channels(pos, 16)
    assert len(set(idx.tolist())) == 16
