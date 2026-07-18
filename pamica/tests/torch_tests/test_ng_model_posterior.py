"""Live per-model posterior accessor tests (issue #141).

``model_loglik``/``model_probability`` expose, for arbitrary data, the per-model
per-sample log-likelihood the E-step already computes internally. Real sample
EEG only (no synthetic data): the internal training-data ``_llt_lht`` (Fortran's
LLt, issue #155) is an exact oracle for ``model_loglik`` evaluated on that same
data, so the live accessor is pinned bit-for-bit rather than by eyeball. The
equality holds when the fit did not use ``do_reject`` (the default here);
``_compute_full_posterior_ll`` zeroes rejected columns as Fortran's ``load_rej``
sentinel, which ``model_loglik`` (rejection-unaware) does not reproduce.
"""

from pathlib import Path

import numpy as np
import pytest

from pamica import AMICATorchNG
from pamica.torch_impl.utils import load_eeglab_data

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
DATA_FILE = SAMPLE_DIR / "eeglab_data.fdt"
NW = 32
FIELD = 30504
SEED = 42

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="sample data missing")


@pytest.fixture(scope="module")
def real_data():
    return load_eeglab_data(str(DATA_FILE), data_dim=NW, field_dim=FIELD).astype(
        np.float64
    )


def _fit(X, n_models, max_iter=10):
    m = AMICATorchNG(n_channels=NW, n_models=n_models, n_mix=3, seed=SEED, device="cpu")
    m.fit(X, max_iter=max_iter, verbose=False)
    return m


def test_model_loglik_matches_internal_lht(real_data):
    """model_loglik on the training data equals the stored _llt_lht exactly."""
    m = _fit(real_data, n_models=2)
    lht = m.model_loglik(real_data)
    assert lht.shape == (2, real_data.shape[1])
    # _llt_lht is the E-step's own per-model per-sample LL, recomputed post-fit
    # from the same parameters -- an exact oracle for the live accessor.
    np.testing.assert_array_equal(lht, m._llt_lht)


def test_model_probability_is_normalized(real_data):
    m = _fit(real_data, n_models=2)
    prob = m.model_probability(real_data)
    assert prob.shape == (2, real_data.shape[1])
    np.testing.assert_allclose(prob.sum(axis=0), 1.0, atol=1e-12)
    assert np.all(prob >= 0.0) and np.all(prob <= 1.0)


def test_single_model_probability_is_all_ones(real_data):
    m = _fit(real_data, n_models=1, max_iter=5)
    np.testing.assert_allclose(m.model_probability(real_data), 1.0, atol=1e-12)


def test_model_loglik_uses_stored_sphere_not_reprocess(real_data):
    """Scoring new data must not overwrite the fitted sphere/mean."""
    m = _fit(real_data, n_models=2)
    sphere_before = m.sphere.clone()
    mean_before = m.mean.clone()
    _ = m.model_loglik(real_data[:, :1000])  # a different-length slice
    assert np.array_equal(m.sphere.cpu().numpy(), sphere_before.cpu().numpy())
    assert np.array_equal(m.mean.cpu().numpy(), mean_before.cpu().numpy())


def test_model_loglik_requires_fit():
    m = AMICATorchNG(n_channels=NW, n_models=2, n_mix=3, seed=SEED, device="cpu")
    with pytest.raises(RuntimeError, match="fitted"):
        m.model_loglik(np.zeros((NW, 10)))


def test_model_loglik_rejects_non_finite_input(real_data):
    m = _fit(real_data, n_models=2, max_iter=5)
    bad = real_data.copy()
    bad[3, 100] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        m.model_loglik(bad)
    with pytest.raises(ValueError, match="non-finite"):
        m.model_probability(bad)


def test_amica_wrapper_delegates(real_data):
    """The scikit-learn-style AMICA wrapper exposes the same accessors."""
    from pamica import AMICA

    a = AMICA(n_models=2, n_mix=3, device="cpu", verbose=False)
    a.fit(real_data, max_iter=8, seed=SEED)
    lht = a.model_loglik(real_data)
    prob = a.model_probability(real_data)
    assert lht.shape == prob.shape == (2, real_data.shape[1])
    np.testing.assert_allclose(prob.sum(axis=0), 1.0, atol=1e-12)
