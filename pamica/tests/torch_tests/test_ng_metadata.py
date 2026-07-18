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


def test_shared_components_reports_merges(real_data):
    m = _fit(
        real_data,
        n_models=2,
        max_iter=25,
        share_comps=True,
        share_start=3,
        share_iter=7,
        comp_thresh=0.9,
    )
    groups = m.shared_components()
    assert len(groups) >= 1
    for group in groups:
        models = {h for h, _ in group}
        assert len(models) >= 2  # a shared column spans >= 2 models
        assert all(0 <= h < 2 and 0 <= i < NW for h, i in group)


def test_metadata_accessors_require_fit():
    m = AMICATorchNG(n_channels=NW, n_models=1, n_mix=NMIX, seed=SEED, device="cpu")
    for call in (m.get_pdftype, m.get_rho, m.shared_components):
        with pytest.raises(RuntimeError, match="fitted"):
            call()
