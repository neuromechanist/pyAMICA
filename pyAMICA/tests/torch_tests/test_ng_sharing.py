"""Component sharing (issue #60) for AMICATorchNG.

Covers the ``share_comps`` reassignment ported from the Fortran
``identify_shared_comps`` (amica15.f90:1898). There is no bit-exact oracle (the
reference's ``Spinv2`` metric is declared but never allocated, like the dead
``do_choose_pdfs`` switch, #26), so the merge *mechanism* is tested with
controlled mixing matrices and the end-to-end behavior on real sample EEG.
Sharing is multi-model only and OFF by default, so single-model parity is
unchanged.
"""

from pathlib import Path
from typing import Any, Dict

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


# --- default-path parity ----------------------------------------------------


def test_single_model_byte_identical_with_share_toggled(real_data):
    """Sharing is a no-op for n_models=1, so a single-model fit must be
    byte-for-byte identical with share_comps on vs off (the gm-weighted A-update
    reduces to the plain update since gm=[1] cancels exactly)."""
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
        share_start=7,
        share_iter=8,
    )
    np.testing.assert_array_equal(off.get_mixing_matrix(0), on.get_mixing_matrix(0))
    np.testing.assert_array_equal(off.get_unmixing_matrix(0), on.get_unmixing_matrix(0))
    assert off.final_ll_ == on.final_ll_


def test_default_multimodel_leaves_comps_unshared(real_data):
    """With share_comps off (default), a 2-model fit keeps every component
    distinct (comp_list stays the full block layout)."""
    ng = AMICATorchNG(
        n_channels=NW, n_models=2, n_mix=3, device="cpu", block_size=1024, seed=1
    )
    ng.fit(real_data[:, :4096], max_iter=10)
    assert ng.A is not None
    assert int(ng.comp_used.sum()) == ng.n_comps
    assert torch.isfinite(ng.A).all()


# --- merge mechanism (controlled) ------------------------------------------


def test_identify_merges_one_collinear_pair():
    """A single collinear cross-model column pair is merged; the others (random,
    near-orthogonal) are left alone."""
    torch.manual_seed(0)
    n = 4
    A = torch.randn(n, 2 * n, dtype=torch.float64)
    A = A / A.norm(dim=0, keepdim=True)
    A[:, 5] = A[
        :, 2
    ]  # model-1 source 1 (col 5) collinear with model-0 source 2 (col 2)
    ng = _controlled_ng(A, n_channels=n, n_models=2)
    assert ng.comp_list is not None
    assert int(ng.comp_used.sum()) == 2 * n
    ng._identify_shared_comps()
    assert int(ng.comp_used.sum()) == 2 * n - 1
    assert int(ng.comp_list[1, 1]) == int(ng.comp_list[2, 0]) == 2
    assert not bool(ng.comp_used[5])  # col 5 is now unused


def test_guard_prevents_within_model_collapse():
    """When every column is collinear, sharing must NOT collapse a model's own
    sources together: with 2 sources x 2 models it settles to 2 shared comps,
    not 1 (Fortran's 'common presence in a model' guard, amica15.f90:1918)."""
    n = 2
    v = torch.tensor([1.0, 2.0], dtype=torch.float64)
    A = torch.stack([v] * (2 * n), dim=1)
    ng = _controlled_ng(A, n_channels=n, n_models=2)
    assert ng.comp_list is not None
    ng._identify_shared_comps()
    assert int(ng.comp_used.sum()) == 2
    assert ng.comp_list[0, 0] != ng.comp_list[1, 0]


def test_three_model_guard_and_merge():
    """3-model scan (h<hh runs 3 pairs, with sequential mutation): a source
    collinear across all three models collapses to ONE shared comp, while each
    model's other source stays distinct."""
    n = 2
    torch.manual_seed(1)
    A = torch.randn(n, 3 * n, dtype=torch.float64)
    A = A / A.norm(dim=0, keepdim=True)
    # source 0 of every model shares one direction; source 1 stays random.
    shared = A[:, 0].clone()
    A[:, 2] = shared  # model1 source0 (col 2)
    A[:, 4] = shared  # model2 source0 (col 4)
    ng = _controlled_ng(A, n_channels=n, n_models=3)
    assert ng.comp_list is not None
    ng._identify_shared_comps()
    # source-0 column shared by all 3 models -> one comp; three distinct source-1s.
    assert int(ng.comp_list[0, 0]) == int(ng.comp_list[0, 1]) == int(ng.comp_list[0, 2])
    assert int(ng.comp_used.sum()) == 4  # 1 shared + 3 distinct source-1 comps
    # no model shares a comp between its own two sources
    for h in range(3):
        assert ng.comp_list[0, h] != ng.comp_list[1, h]


def test_comp_thresh_one_merges_only_exact_duplicates():
    """comp_thresh=1.0: an exact-duplicate column merges (cos==1), a merely
    similar one does not."""
    n = 3
    torch.manual_seed(2)
    A = torch.randn(n, 2 * n, dtype=torch.float64)
    A = A / A.norm(dim=0, keepdim=True)
    A[:, 3] = A[:, 0]  # exact duplicate across models
    ng = _controlled_ng(A, n_channels=n, n_models=2)
    assert ng.comp_list is not None
    ng.comp_thresh = 1.0
    ng._identify_shared_comps()
    assert int(ng.comp_list[0, 1]) == int(ng.comp_list[0, 0])  # duplicate merged
    assert int(ng.comp_used.sum()) == 2 * n - 1


# --- freeze schedule --------------------------------------------------------


def test_a_frozen_window():
    """A is frozen for the merge iteration + 5 after it, thawed for the rest of
    each cycle, and frozen again at the next cycle boundary."""
    ng = AMICATorchNG(
        n_channels=4,
        n_models=2,
        device="cpu",
        share_comps=True,
        share_start=10,
        share_iter=20,
    )

    def frozen(itf):
        ng.iteration = itf - 1  # itf is the Fortran-style 1-indexed iteration
        return ng._a_frozen()

    assert not any(frozen(i) for i in range(1, 10))  # before share_start
    assert all(frozen(i) for i in range(10, 16))  # merge + 5 (residue 0..5)
    assert not any(frozen(i) for i in range(16, 30))  # thawed rest of cycle
    assert all(frozen(i) for i in range(30, 36))  # next cycle boundary


def test_a_frozen_off_for_single_model():
    ng = AMICATorchNG(
        n_channels=4,
        n_models=1,
        device="cpu",
        share_comps=True,
        share_start=2,
        share_iter=8,
    )
    ng.iteration = 3
    assert ng._a_frozen() is False


# --- validation -------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(share_start=0), "share_start"),
        (dict(share_iter=6), "share_iter"),
        (dict(share_iter=1), "share_iter"),
        (dict(comp_thresh=0.0), "comp_thresh"),
        (dict(comp_thresh=1.5), "comp_thresh"),
        (dict(pcakeep=10), "PCA"),
        (dict(pcadb=30.0), "PCA"),
    ],
)
def test_share_constructor_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        AMICATorchNG(n_channels=8, n_models=2, share_comps=True, **kwargs)


def test_singular_sphere_raises_on_share():
    """Sharing on a rank-deficient (uninvertible) sphere fails loudly rather than
    merging on a garbage metric."""
    ng = AMICATorchNG(
        n_channels=4,
        n_models=2,
        device="cpu",
        share_comps=True,
        share_start=1,
        share_iter=8,
    )
    sphere = torch.zeros(4, 4, dtype=torch.float64)
    sphere[0, 0] = sphere[1, 1] = 1.0  # rank 2 of 4 -> singular
    ng.sphere = sphere
    ng._spinv = None
    ng.A = torch.randn(4, 8, dtype=torch.float64)
    ng.comp_list = torch.tensor([[0, 4], [1, 5], [2, 6], [3, 7]])
    ng.iteration = 0
    with pytest.raises(RuntimeError, match="invertible|singular|non-finite"):
        ng._identify_shared_comps()


# --- end-to-end on real data ------------------------------------------------


def test_two_model_share_fit_survives_merge(real_data):
    """A full 2-model fit with sharing runs to completion with finite parameters
    and the merge SURVIVES to the returned model (keep_best is disabled under
    share_comps, so fit returns the merged last iterate, not a pre-merge peak)."""
    ng = AMICATorchNG(
        n_channels=NW,
        n_models=2,
        n_mix=3,
        device="cpu",
        block_size=1024,
        seed=3,
        do_newton=True,
        share_comps=True,
        share_start=8,
        share_iter=10,
        comp_thresh=0.9,
    )
    ng.fit(real_data[:, :4096], max_iter=40)
    assert ng.A is not None and ng.final_ll_ is not None
    assert torch.isfinite(ng.A).all()
    assert np.isfinite(ng.final_ll_)
    assert int(ng.comp_used.sum()) < ng.n_comps  # at least one merge survived


def test_sharing_reduces_unique_count_without_degrading_ll(real_data):
    """Enabling sharing on matched config strictly reduces the unique-component
    count and does not materially degrade the log-likelihood."""
    x = real_data[:, :4096]
    common: Dict[str, Any] = dict(
        n_channels=NW,
        n_models=2,
        n_mix=3,
        device="cpu",
        block_size=1024,
        seed=7,
        do_newton=True,
    )
    base = AMICATorchNG(**common)
    base.fit(x, max_iter=40)
    shared = AMICATorchNG(
        **common, share_comps=True, share_start=8, share_iter=10, comp_thresh=0.9
    )
    shared.fit(x, max_iter=40)
    assert int(base.comp_used.sum()) == base.n_comps
    assert int(shared.comp_used.sum()) < base.n_comps
    assert shared.final_ll_ is not None and base.final_ll_ is not None
    assert np.isfinite(shared.final_ll_)
    assert shared.final_ll_ > base.final_ll_ - 0.3  # no material degradation


def test_share_config_and_comp_list_roundtrip(real_data, tmp_path):
    """save/load preserves the share configuration and the merged comp_list."""
    model = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    model.fit(
        real_data[:, :4096],
        max_iter=25,
        block_size=1024,
        seed=3,
        do_newton=True,
        share_comps=True,
        share_start=8,
        share_iter=10,
        comp_thresh=0.9,
    )
    assert model.model_ is not None
    assert int(model.model_.comp_used.sum()) < model.model_.n_comps  # a merge happened
    path = str(tmp_path / "shared.pt")
    model.save(path)
    loaded = AMICA.load(path, device="cpu")
    assert loaded.model_ is not None
    assert loaded.model_.share_comps is True
    assert loaded.model_.share_start == 8 and loaded.model_.share_iter == 10
    assert loaded.model_.comp_list is not None and model.model_.comp_list is not None
    assert torch.equal(loaded.model_.comp_list, model.model_.comp_list)
