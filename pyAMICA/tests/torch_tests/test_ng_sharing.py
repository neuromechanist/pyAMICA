"""Component sharing (issue #60) for AMICATorchNG.

Covers the ``share_comps`` reassignment ported from the Fortran
``identify_shared_comps`` (amica15.f90:1898). There is no bit-exact oracle (the
reference's similarity metric ``Spinv2`` is never initialized, like
``do_choose_pdfs``, #26), so the merge *mechanism* is tested with controlled
mixing matrices and the end-to-end behavior on real sample EEG. Sharing is
multi-model only and OFF by default, so single-model parity is unchanged.
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from pyAMICA.amica import AMICA
from pyAMICA.torch_impl.core import AMICATorchNG
from pyAMICA.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504


@pytest.fixture(scope="module")
def real_data() -> np.ndarray:
    if not DATA_FILE.exists():
        pytest.skip("sample data missing")
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _controlled_ng(A: torch.Tensor, n_channels: int, n_models: int) -> AMICATorchNG:
    """An NG with a fixed identity sphere and default comp_list, whose A is the
    given (n_channels, n_channels*n_models) matrix -- so the sharing metric is a
    plain cosine similarity on A's columns. No fit needed."""
    ng = AMICATorchNG(
        n_channels=n_channels,
        n_models=n_models,
        n_mix=3,
        device="cpu",
        share_comps=True,
        comp_thresh=0.99,
    )
    ng.sphere = torch.eye(n_channels, dtype=torch.float64)
    ng._spinv = None
    ng.iteration = 0
    cl = np.zeros((n_channels, n_models), dtype=np.int64)
    for h in range(n_models):
        cl[:, h] = np.arange(h * n_channels, (h + 1) * n_channels)
    ng.comp_list = torch.from_numpy(cl)
    ng.A = A.to(torch.float64)
    return ng


def test_single_model_byte_identical_with_share_toggled(real_data):
    """Sharing is a no-op for n_models=1 (nothing to share), so a single-model
    fit must be byte-for-byte identical with share_comps on vs off."""
    x = real_data[:, :4096]
    off = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    off.fit(x, max_iter=8, seed=42, block_size=1024, do_newton=True)
    on = AMICA(n_models=1, n_mix=3, device="cpu", verbose=False)
    on.fit(
        x,
        max_iter=8,
        seed=42,
        block_size=1024,
        do_newton=True,
        share_comps=True,
        share_start=3,
        share_iter=3,
    )
    np.testing.assert_array_equal(off.get_mixing_matrix(0), on.get_mixing_matrix(0))
    np.testing.assert_array_equal(off.get_unmixing_matrix(0), on.get_unmixing_matrix(0))
    assert off.final_ll_ == on.final_ll_


def test_identify_merges_one_collinear_pair():
    """A single collinear cross-model column pair is merged; the others (random,
    near-orthogonal) are left alone."""
    torch.manual_seed(0)
    n = 4
    A = torch.randn(n, 2 * n, dtype=torch.float64)
    A = A / A.norm(dim=0, keepdim=True)
    # Force model-1 source 1 (global col 5) collinear with model-0 source 2 (col 2).
    A[:, 5] = A[:, 2]
    ng = _controlled_ng(A, n_channels=n, n_models=2)
    assert int(ng.comp_used.sum()) == 2 * n  # all distinct to start
    ng._identify_shared_comps()
    assert int(ng.comp_used.sum()) == 2 * n - 1  # exactly one merge
    # col 5 folded into col 2: source 1 of model 1 now points to comp 2.
    assert int(ng.comp_list[1, 1]) == int(ng.comp_list[2, 0]) == 2


def test_guard_prevents_within_model_collapse():
    """When every column is collinear, sharing must NOT collapse a model's own
    sources together: with 2 sources x 2 models it settles to 2 shared comps,
    not 1 (Fortran's 'common presence in a model' guard, amica15.f90:1918)."""
    n = 2
    v = torch.tensor([1.0, 2.0], dtype=torch.float64)
    A = torch.stack([v] * (2 * n), dim=1)  # all four columns collinear
    ng = _controlled_ng(A, n_channels=n, n_models=2)
    ng._identify_shared_comps()
    assert int(ng.comp_used.sum()) == 2  # two shared comps, one per source slot
    # each model's two sources stay distinct comps
    assert ng.comp_list[0, 0] != ng.comp_list[1, 0]


def test_default_multimodel_leaves_comps_unshared(real_data):
    """With share_comps off (default), a 2-model fit keeps every component
    distinct (comp_list stays the full block layout)."""
    ng = AMICATorchNG(
        n_channels=NW, n_models=2, n_mix=3, device="cpu", block_size=1024, seed=1
    )
    ng.fit(real_data[:, :4096], max_iter=10)
    assert int(ng.comp_used.sum()) == ng.n_comps
    assert torch.isfinite(ng.A).all()


def test_two_model_share_fit_is_finite_and_shares(real_data):
    """A full 2-model fit with sharing enabled runs to completion with finite
    parameters and does not increase the unique-component count."""
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=3,
        device="cpu",
        block_size=1024,
        seed=3,
        do_newton=True,
        share_comps=True,
        share_start=5,
        share_iter=5,
        comp_thresh=0.9,
    )
    ng.fit(real_data[:, :4096], max_iter=25)
    assert torch.isfinite(ng.A).all()
    assert np.isfinite(ng.final_ll_)
    assert int(ng.comp_used.sum()) <= ng.n_comps


def test_share_config_and_comp_list_roundtrip(real_data, tmp_path):
    """save/load preserves the share configuration and the (possibly merged)
    comp_list."""
    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(
        real_data[:, :4096],
        max_iter=12,
        block_size=1024,
        seed=3,
        do_newton=True,
        share_comps=True,
        share_start=3,
        share_iter=3,
        comp_thresh=0.9,
    )
    path = str(tmp_path / "shared.pt")
    model.save(path)
    loaded = AMICA.load(path, device="cpu")
    assert loaded.model_.share_comps is True
    assert loaded.model_.share_start == 3 and loaded.model_.share_iter == 3
    assert torch.equal(loaded.model_.comp_list, model.model_.comp_list)
