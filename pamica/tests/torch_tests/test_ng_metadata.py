"""Fitted-parameter metadata accessors (issue #142).

``get_pdftype``/``get_rho``/``shared_components`` expose the source-density
family, GG shape, and component-sharing state that the EEGLAB/MNE exports do not
carry. Real sample EEG only (no synthetic data).
"""

from pathlib import Path

import numpy as np
import pytest

from pamica import AMICATorchNG
from pamica.torch_impl import PDFTYPE_NAMES
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
NMIX = 3
SEED = 42

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def real_data():
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _fit(X, n_models=1, max_iter=8, **kw):
    m = AMICATorchNG(
        n_channels=NW, n_models=n_models, n_mix=NMIX, seed=SEED, device="cpu", **kw
    )
    m.fit(X, max_iter=max_iter, verbose=False)
    return m


def test_pdftype_names_cover_all_codes():
    assert set(PDFTYPE_NAMES) == {0, 1, 2, 3, 4}


def test_get_pdftype_default_is_generalized_gaussian(real_data):
    m = _fit(real_data)
    pdf = m.get_pdftype()
    assert pdf.shape == (NW,)
    assert np.array_equal(np.unique(pdf), [0])  # pdftype=0 default
    assert PDFTYPE_NAMES[int(pdf[0])] == "generalized_gaussian"


def test_get_pdftype_reflects_fixed_family(real_data):
    m = _fit(real_data, pdftype=2)  # Gaussian
    assert np.array_equal(np.unique(m.get_pdftype()), [2])


def test_get_rho_shape_and_bounds(real_data):
    m = _fit(real_data)
    rho = m.get_rho()
    assert rho.shape == (NMIX, NW)
    assert np.all(rho >= m.minrho) and np.all(rho <= m.maxrho)


def test_shared_components_empty_without_sharing(real_data):
    assert _fit(real_data, n_models=1).shared_components() == []
    assert _fit(real_data, n_models=2).shared_components() == []  # share off default


def test_shared_components_reports_partial_merges(real_data):
    """A PARTIAL merge (some columns single-model) exercises both the grouping
    and the exclude-filter: shared groups appear, non-shared columns do not."""
    m = _fit(
        real_data[:, :4096],
        n_models=2,
        max_iter=25,
        block_size=1024,
        share_comps=True,
        share_start=8,
        share_iter=10,
        do_newton=True,
    )
    groups = m.shared_components()
    # Partial: not zero (something merged), and fewer than n_sources (some
    # columns stay single-model and are correctly excluded from the groups).
    assert 0 < len(groups) < NW
    for group in groups:
        models = {h for h, _ in group}
        assert len(models) >= 2  # a shared column spans >= 2 models
        assert all(0 <= h < 2 and 0 <= i < NW for h, i in group)


def test_get_rho_differs_per_model(real_data):
    """get_rho indexes comp_list per model, so a 2-model fit's rho differs."""
    m = _fit(real_data, n_models=2, max_iter=10)
    assert not np.allclose(m.get_rho(model_idx=0), m.get_rho(model_idx=1))


def test_get_pdftype_adaptive_switcher_is_mixed(real_data):
    """The adaptive switcher (pdftype=1) moves sources into families {1, 4};
    get_pdftype must surface those per-source codes, not a uniform array."""
    m = AMICATorchNG(
        n_channels=NW,
        n_models=1,
        n_mix=1,  # adaptive mode is single-component
        seed=SEED,
        device="cpu",
        pdftype=1,
        kurt_start=3,
        num_kurt=5,
        kurt_int=1,
    )
    m.fit(real_data, max_iter=20, verbose=False)
    codes = set(np.unique(m.get_pdftype()).tolist())
    assert codes.issubset({1, 4}) and codes  # only the two cosh families


def test_metadata_accessors_reject_bad_model_idx(real_data):
    m = _fit(real_data, n_models=2, max_iter=5)
    for call in (m.get_pdftype, m.get_rho):
        with pytest.raises(ValueError, match="out of range"):
            call(model_idx=2)
        with pytest.raises(ValueError, match="out of range"):
            call(model_idx=-1)  # negative would silently wrap without the guard


def test_metadata_accessors_require_fit():
    m = AMICATorchNG(n_channels=NW, n_models=1, n_mix=NMIX, seed=SEED, device="cpu")
    for call in (m.get_pdftype, m.get_rho, m.shared_components):
        with pytest.raises(RuntimeError, match="fitted"):
            call()


def test_amica_wrapper_delegates_metadata(real_data):
    """The scikit-learn-style AMICA wrapper exposes the same accessors."""
    from pamica import AMICA

    a = AMICA(n_models=2, n_mix=NMIX, device="cpu", verbose=False)
    a.fit(real_data, max_iter=8, seed=SEED)
    assert a.get_pdftype(model_idx=1).shape == (NW,)
    assert a.get_rho(model_idx=1).shape == (NMIX, NW)
    assert a.shared_components() == []  # share off
    with pytest.raises(ValueError, match="out of range"):
        a.get_pdftype(model_idx=-1)  # guard propagates through the wrapper
