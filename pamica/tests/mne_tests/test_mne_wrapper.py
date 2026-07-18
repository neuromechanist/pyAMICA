"""AMICAICA MNE-wrapper tests (issue #139 phase 1, single-model).

Real sample EEG only (no synthetic/mock): the bundled EEGLAB ``eeglab_data.set``
(32 channels, 30504 samples) is loaded through ``mne.io.read_raw_eeglab`` -- the
same entry point a real MNE user would use. The whole module self-skips when the
optional ``mne`` extra is absent, so the base CI env (no mne) skips it; the
dedicated ``test-mne`` CI job installs ``pamica[mne]`` and runs it for real.

The load-bearing check is the round trip: the ICA that ``to_mne_ica`` builds must
reproduce ``AMICA.transform`` bit-for-bit (up to float64 eigh residual), which
pins the mean/sphere/unmixing -> pca_mean_/pca_components_/unmixing_matrix_ map.
"""

from pathlib import Path

import numpy as np
import pytest

mne = pytest.importorskip("mne")

from pamica.mne_compat import AMICAICA  # noqa: E402  (after importorskip)

mne.set_log_level("ERROR")

SAMPLE_DIR = Path(__file__).resolve().parents[2] / "sample_data"
SET_FILE = SAMPLE_DIR / "eeglab_data.set"
SEED = 42
MAX_ITER = 12  # enough to move off the init; parity is orientation, not convergence

pytestmark = pytest.mark.skipif(
    not SET_FILE.exists(), reason="sample eeglab_data.set missing"
)


@pytest.fixture(scope="module")
def raw():
    """Real continuous EEG as an MNE Raw (32 EEG channels, 128 Hz)."""
    return mne.io.read_raw_eeglab(str(SET_FILE), preload=True)


@pytest.fixture(scope="module")
def fitted(raw):
    """A single AMICAICA fit reused across the read-only assertions."""
    return AMICAICA(n_mix=3, random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )


def _picked_data(raw):
    """The exact array AMICAICA fits on (all good data channels, float64)."""
    return raw.copy().pick("data", exclude="bads").get_data().astype(np.float64)


# --- the mapping crux -------------------------------------------------------
def test_get_sources_matches_amica_transform(raw, fitted):
    """to_mne_ica().get_sources == AMICA.transform on the same data."""
    s_mne = fitted.to_mne_ica().get_sources(raw).get_data()
    s_amica = fitted.amica_.transform(_picked_data(raw))
    assert s_mne.shape == s_amica.shape == (fitted.n_components_, raw.n_times)
    np.testing.assert_allclose(s_mne, s_amica, rtol=1e-6, atol=1e-9)


def test_get_components_are_channel_space_maps(fitted):
    """get_components() equals the channel-space mixing inv(sphere) @ inv(W)."""
    comps = fitted.get_components()
    assert comps.shape == (fitted.n_components_, fitted.n_components_)
    sphere = fitted.amica_.model_.sphere.cpu().numpy()
    w_fort = fitted.amica_.get_unmixing_matrix(0)
    maps_ref = np.linalg.inv(sphere) @ np.linalg.inv(w_fort)
    np.testing.assert_allclose(comps, maps_ref, rtol=1e-6, atol=1e-10)


def test_apply_without_exclude_reconstructs(raw, fitted):
    """apply() with no excluded components returns the input unchanged."""
    recon = fitted.apply(raw.copy())
    np.testing.assert_allclose(recon.get_data(), raw.get_data(), rtol=1e-6, atol=1e-12)


def test_apply_excludes_component(raw, fitted):
    """Excluding a component changes the data and zeroes that source."""
    ica = fitted.to_mne_ica()
    cleaned = ica.apply(raw.copy(), exclude=[0])
    # The reconstruction must differ from the input (a component was removed).
    assert not np.allclose(cleaned.get_data(), raw.get_data())
    # Re-deriving sources from the cleaned data: component 0 is gone.
    src_after = ica.get_sources(cleaned).get_data()
    assert np.abs(src_after[0]).max() < 1e-6 * np.abs(src_after).max()


# --- object validity --------------------------------------------------------
def test_to_mne_ica_returns_valid_fitted_ica(raw, fitted):
    ica = fitted.to_mne_ica()
    assert isinstance(ica, mne.preprocessing.ICA)
    assert ica.current_fit != "unfitted"
    assert ica.n_components_ == fitted.n_components_
    assert ica.ch_names == fitted.ch_names_
    # A genuine MNE ICA exposes the full component interface.
    assert ica.get_components().shape == (fitted.n_components_, fitted.n_components_)


def test_export_is_cached_and_invalidated_on_refit(raw):
    ica = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, max_iter=MAX_ITER
    )
    first = ica.to_mne_ica()
    assert ica.to_mne_ica() is first  # cached
    ica.fit(raw, max_iter=MAX_ITER)  # refit invalidates the cache
    assert ica.to_mne_ica() is not first


# --- picks / epochs / plotting ----------------------------------------------
def test_fit_with_channel_subset(raw):
    picks = raw.ch_names[:10]
    ica = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        raw, picks=picks, max_iter=MAX_ITER
    )
    assert ica.n_components_ == 10
    assert ica.ch_names_ == picks
    s_mne = ica.to_mne_ica().get_sources(raw).get_data()
    x = raw.copy().pick(picks).get_data().astype(np.float64)
    assert ica.amica_ is not None
    np.testing.assert_allclose(s_mne, ica.amica_.transform(x), rtol=1e-6, atol=1e-9)


def test_fit_from_epochs(raw):
    epochs = mne.make_fixed_length_epochs(raw, duration=2.0, preload=True)
    ica = AMICAICA(random_state=SEED, device="cpu", verbose=False).fit(
        epochs, max_iter=MAX_ITER
    )
    assert ica._fit_kind == "epochs"
    src = ica.to_mne_ica().get_sources(epochs).get_data()  # (n_ep, n_comp, n_time)
    x = np.hstack(epochs.copy().pick("data", exclude="bads").get_data())
    assert ica.amica_ is not None
    np.testing.assert_allclose(
        np.hstack(src), ica.amica_.transform(x), rtol=1e-6, atol=1e-9
    )


def test_plot_components_returns_figure(fitted):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.figure
    import matplotlib.pyplot as plt

    out = fitted.plot_components(picks=[0, 1], show=False)
    figs = out if isinstance(out, list) else [out]
    assert figs and all(isinstance(f, matplotlib.figure.Figure) for f in figs)
    plt.close("all")


# --- guards -----------------------------------------------------------------
def test_unfitted_calls_raise():
    ica = AMICAICA()
    with pytest.raises(RuntimeError, match="must be fitted"):
        ica.to_mne_ica()


def test_fit_rejects_non_mne_input():
    with pytest.raises(TypeError, match="mne.io.Raw or mne.Epochs"):
        AMICAICA().fit(np.random.randn(8, 500))
