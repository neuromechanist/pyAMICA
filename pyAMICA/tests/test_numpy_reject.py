"""Outlier rejection (do_reject) on the NumPy backend (issue #123).

Analogous to the torch backend's do_reject tests (test_ng_backend.py): the
mechanism (good_idx shrinks, gm/LL normalize by the good count, degenerate LL
raises) is behavior-validated against the real sample EEG, never synthetic data.
The NumPy port mirrors AMICATorchNG.good_idx / _reject_outliers.
"""

from pathlib import Path

import numpy as np
import pytest

from pyAMICA import AMICA_NumPy as AMICA
from pyAMICA.numpy_impl.data import load_data_file

_FDT = Path(__file__).resolve().parent.parent / "sample_data" / "eeglab_data.fdt"


def _real_data(n_samples: int) -> np.ndarray:
    """A slice of the committed sample EEG (32 channels), float64."""
    data = load_data_file(str(_FDT), 32, 30504, dtype=np.float32)
    return data[:, :n_samples].astype(np.float64)


def test_reject_param_validation():
    """Invalid rejection parameters are refused at construction (matching
    AMICATorchNG), guarding the otherwise-opaque downstream failures: rejint=0
    -> ZeroDivisionError in the reject schedule, rejsig<=0 -> every sample
    dropped."""
    with pytest.raises(ValueError, match="rejint"):
        AMICA(do_reject=True, rejint=0)
    with pytest.raises(ValueError, match="rejsig"):
        AMICA(do_reject=True, rejsig=0.0)
    with pytest.raises(ValueError, match="rejsig"):
        AMICA(do_reject=True, rejsig=-5.0)


def test_do_reject_false_leaves_good_idx_unset():
    """The default path never touches the rejection machinery: good_idx stays
    None and the fit is unaffected (issue #24 parity path)."""
    model = AMICA(num_models=1, num_mix=3, max_iter=3, do_newton=False)
    model.fit(_real_data(2000))
    assert model.good_idx is None


@pytest.mark.skipif(not _FDT.exists(), reason="sample data missing")
def test_rejection_shrinks_good_sample_set():
    """With do_reject enabled, low-log-likelihood samples are permanently
    dropped: the good-sample set shrinks, at most maxrej passes run, and the
    fit stays finite (Fortran-style reject_data, mirroring the torch backend)."""
    data = _real_data(3000)
    n_total = data.shape[1]
    model = AMICA(
        num_models=1,
        num_mix=3,
        max_iter=12,
        do_newton=False,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        rejint=3,
        maxrej=2,
    )
    model.fit(data)

    assert model.good_idx is not None
    n_good = int(model.good_idx.size)
    assert n_good < n_total  # some samples were rejected
    assert n_good == model.num_good_samples  # count tracks the index array
    # good_idx is a strict subset of the original indices, unique and in range.
    assert model.good_idx.min() >= 0 and model.good_idx.max() < n_total
    assert len(set(model.good_idx.tolist())) == n_good
    # The fit stayed finite through the rejection.
    assert model.W is not None
    assert np.all(np.isfinite(model.W))
    assert np.all(np.isfinite(model.ll))


@pytest.mark.skipif(not _FDT.exists(), reason="sample data missing")
def test_gm_normalizes_by_good_count_after_rejection():
    """After rejection, gm (the model weights) is normalized by the good-sample
    count, not the original total. For a single model gm sums to 1 regardless,
    so assert the invariant that survives rejection: gm stays a valid, finite
    distribution over models."""
    data = _real_data(3000)
    model = AMICA(
        num_models=2,
        num_mix=3,
        max_iter=10,
        do_newton=False,
        do_reject=True,
        rejsig=2.0,
        rejstart=2,
        maxrej=1,
    )
    model.fit(data)
    assert model.good_idx is not None and int(model.good_idx.size) < data.shape[1]
    assert model.gm is not None
    assert np.all(np.isfinite(model.gm))
    assert model.gm.min() >= 0.0
    np.testing.assert_allclose(model.gm.sum(), 1.0, atol=1e-8)


def test_rejection_degenerate_ll_raises_clear_error():
    """A degenerate per-sample LL (e.g. all NaN from a diverged fit) makes
    rejection raise a clear ValueError rather than silently emptying the good
    set and crashing downstream on a zero-length normalization."""
    model = AMICA(do_reject=True, rejsig=3.0)
    model.good_idx = np.arange(16)
    model._last_ll_samples = np.full(16, np.nan)
    with pytest.raises(ValueError, match="removed all"):
        model._reject_outliers()
