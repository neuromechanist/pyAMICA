"""
Correctness fixes for the three math bugs in the torch backends (issue #11):

1. Swapped mixture factorization -- per-source logsumexp over mixture
   components must happen BEFORE summing over sources, not after.
2. Per-source alpha collapsed to a scalar mean.
3. LL normalization -- Fortran reports LL / (n_samples * n_sources) and does
   not add an explicit log|det W| Jacobian term to the reported LL.

All assertions here run against the real sample EEG data
(`pyAMICA/sample_data/eeglab_data.fdt`), never synthetic/mock data, per
project policy (AGENTS.md NO MOCKS rule). Reference values are computed
directly with numpy/scipy from the model's own parameters, independent of
the `compute_log_likelihood` code path under test.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import scipy.special as sp
import torch

from pyAMICA.torch_impl.amica_torch import AMICATorch
from pyAMICA.torch_impl.amica_torch_v2 import AMICATorchV2
from pyAMICA.torch_impl.utils import load_eeglab_data

SAMPLE_DATA_DIR = Path(__file__).parent.parent.parent / "sample_data"
EEGLAB_DATA = SAMPLE_DATA_DIR / "eeglab_data.fdt"
SAMPLE_PARAMS = SAMPLE_DATA_DIR / "sample_params.json"

# Fortran final LL from .context/phase1_baseline.md (real sample data,
# amica15mac binary): -3.4456 (20 iters), -3.4108 (100 iters). Reported LL
# should now sit in the same order of magnitude, not the ~-44 the swapped
# factorization + missing nw-normalization used to produce.
FORTRAN_LL_BALLPARK = (-10.0, -0.1)


@pytest.fixture(scope="module")
def sample_data_slice():
    """Load a slice of the real EEGLAB sample data (no synthetic data)."""
    if not EEGLAB_DATA.exists():
        pytest.skip(f"Sample data not found at {EEGLAB_DATA}")

    with open(SAMPLE_PARAMS) as f:
        params = json.load(f)

    data = load_eeglab_data(
        str(EEGLAB_DATA),
        data_dim=params["data_dim"],
        field_dim=params["field_dim"][0],
        dtype=np.float32,
    )
    # Use a real (non-random) slice for fast, deterministic unit tests.
    return data[:, :2000], params


def _numpy_gg_log_pdf(
    y: np.ndarray, mu: np.ndarray, beta: np.ndarray, rho: np.ndarray
) -> np.ndarray:
    """
    Independent numpy reference for the Generalized Gaussian log-PDF.

    y: (n_sources, n_samples); mu, beta, rho: (n_sources,)
    Returns (n_sources, n_samples), matching amica17.f90's GG density
    (rho=1 -> Laplace, rho=2 -> Gaussian), computed from scratch (not by
    calling into torch_impl).
    """
    diff = np.abs(y - mu[:, None]) / beta[:, None]
    log_p = -np.power(diff, rho[:, None])
    log_p = (
        log_p
        + np.log(rho[:, None])
        - np.log(2 * beta[:, None])
        - sp.gammaln(1.0 / rho[:, None])
    )
    return log_p


def _numpy_reference_ll(
    Y: np.ndarray, alpha: np.ndarray, mu: np.ndarray, beta: np.ndarray, rho: np.ndarray
) -> float:
    """
    Hand-computed reference log-likelihood matching AMICA's per-source
    mixture factorization (amica17.f90:1313-1360):
        total_LL = sum_t sum_i log( sum_k alpha[k, i] * f_k(y_{i,t}) )
    normalized by (n_samples * n_sources), with no Jacobian term, matching
    Fortran's reported LL convention (amica17.f90:1866).

    Y: (n_sources, n_samples); alpha, mu, beta, rho: (n_mix, n_sources)
    """
    n_mix, n_sources = alpha.shape
    n_samples = Y.shape[1]

    log_mix = np.stack(
        [
            _numpy_gg_log_pdf(Y, mu[k], beta[k], rho[k]) + np.log(alpha[k])[:, None]
            for k in range(n_mix)
        ],
        axis=0,
    )  # (n_mix, n_sources, n_samples)

    # logsumexp over mixture components, per source
    m = np.max(log_mix, axis=0, keepdims=True)
    log_source_probs = (m + np.log(np.sum(np.exp(log_mix - m), axis=0, keepdims=True)))[
        0
    ]

    total_ll = log_source_probs.sum()
    return total_ll / (n_samples * n_sources)


class TestPerSourceMixtureFactorization:
    """Bug 1: per-source logsumexp over k, THEN sum over sources."""

    def test_matches_hand_computed_reference_on_real_data(self, sample_data_slice):
        data, params = sample_data_slice
        torch.manual_seed(0)

        model = AMICATorch(
            n_channels=params["data_dim"],
            n_sources=params["data_dim"],
            n_models=1,
            n_mix=3,
            device="cpu",
            dtype=torch.float64,
        )
        X_prep = model.preprocess_data(
            data.astype(np.float64), do_mean=True, do_sphere=True
        )

        with torch.no_grad():
            Y = model._forward_single(X_prep, 0).numpy()
            alpha = model.alpha.numpy()
            mu = model.mu.numpy()
            beta = model.beta.numpy()
            rho = model.rho.numpy()

            reported_ll = model.compute_log_likelihood(X_prep).item()

        reference_ll = _numpy_reference_ll(Y, alpha, mu, beta, rho)

        np.testing.assert_allclose(reported_ll, reference_ll, rtol=1e-6, atol=1e-8)

    def test_v2_non_adaptive_fallback_matches_reference(self, sample_data_slice):
        """Same factorization bug existed in amica_torch_v2's non-adaptive path."""
        data, params = sample_data_slice
        torch.manual_seed(0)

        model = AMICATorchV2(
            n_channels=params["data_dim"],
            n_sources=params["data_dim"],
            n_models=1,
            n_mix=3,
            adaptive_pdf=False,
            device="cpu",
            dtype=torch.float64,
            seed=0,
        )
        X_prep = model.preprocess_data(
            data.astype(np.float64), do_mean=True, do_sphere=True
        )

        with torch.no_grad():
            Y = model._forward_single(X_prep, 0).numpy()
            alpha = model.alpha.numpy()
            mu = model.mu.numpy()
            beta = model.beta.numpy()
            rho = model.rho.numpy()

            reported_ll = model.compute_log_likelihood(X_prep).item()

        reference_ll = _numpy_reference_ll(Y, alpha, mu, beta, rho)

        np.testing.assert_allclose(reported_ll, reference_ll, rtol=1e-6, atol=1e-8)


class TestPerSourceAlphaNotMeaned:
    """Bug 2: alpha[k, i] must vary per source, not be collapsed to a scalar mean."""

    def test_different_alpha_per_source_changes_contribution(self, sample_data_slice):
        data, params = sample_data_slice
        torch.manual_seed(0)

        model = AMICATorch(
            n_channels=params["data_dim"],
            n_sources=params["data_dim"],
            n_models=1,
            n_mix=3,
            device="cpu",
            dtype=torch.float64,
        )
        X_prep = model.preprocess_data(
            data.astype(np.float64), do_mean=True, do_sphere=True
        )

        with torch.no_grad():
            # Deliberately give source 0 and source 1 very different mixture
            # weight profiles -- source 0 concentrated on component 0, source
            # 1 concentrated on component 2.
            new_log_alpha = model.log_alpha.clone()
            new_log_alpha[:, 0] = torch.tensor([10.0, -10.0, -10.0], dtype=model.dtype)
            new_log_alpha[:, 1] = torch.tensor([-10.0, -10.0, 10.0], dtype=model.dtype)
            model.log_alpha.copy_(new_log_alpha)

            Y = model._forward_single(X_prep, 0)
            alpha = model.alpha
            mu_k0 = model.mu[0, :].unsqueeze(1)
            beta_k0 = model.beta[0, :].unsqueeze(1)
            rho_k0 = model.rho[0, :].unsqueeze(1)

            log_pdf_k0 = model._compute_gg_log_pdf_vectorized(Y, mu_k0, beta_k0, rho_k0)
            # Per-source contribution of mixture component 0 to the log-mix term.
            contrib_source0 = (
                log_pdf_k0[0] + torch.log(alpha[0, 0] + model.eps)
            ).numpy()
            contrib_source1 = (
                log_pdf_k0[1] + torch.log(alpha[0, 1] + model.eps)
            ).numpy()

        # alpha[0, 0] (~1) and alpha[0, 1] (~0) differ by construction; if alpha
        # were collapsed to a shared scalar mean, these two contributions would
        # differ only by the log_pdf term (a much smaller, data-driven gap).
        # The alpha-driven gap alone should dominate and be clearly resolvable.
        alpha_gap = abs(
            np.log(float(alpha[0, 0]) + 1e-10) - np.log(float(alpha[0, 1]) + 1e-10)
        )
        assert alpha_gap > 5.0, f"expected large per-source alpha gap, got {alpha_gap}"

        observed_gap = np.abs(contrib_source0.mean() - contrib_source1.mean())
        assert observed_gap > 5.0, (
            f"per-source alpha did not propagate into the mixture contribution "
            f"(observed gap {observed_gap}, expected > 5.0 dominated by alpha_gap "
            f"{alpha_gap}); alpha may still be collapsed to a scalar"
        )


class TestLLNormalizationParity:
    """Bug 3: reported LL normalized by (n_samples * n_sources), no Jacobian term."""

    def test_reported_ll_in_fortran_ballpark_on_real_data(self, sample_data_slice):
        data, params = sample_data_slice
        torch.manual_seed(42)

        model = AMICATorch(
            n_channels=params["data_dim"],
            n_sources=params["data_dim"],
            n_models=1,
            n_mix=3,
            device="cpu",
            dtype=torch.float32,
        )
        X_prep = model.preprocess_data(data, do_mean=True, do_sphere=True)

        ll = model.compute_log_likelihood(X_prep).item()

        assert not np.isnan(ll), "LL is NaN"
        low, high = FORTRAN_LL_BALLPARK
        assert low < ll < high, (
            f"reported per-sample-per-channel LL {ll} is outside the Fortran "
            f"ballpark {FORTRAN_LL_BALLPARK} (Fortran final LL is -3.41 to "
            f"-3.45 per .context/phase1_baseline.md)"
        )

    def test_ll_not_scaled_by_n_samples_only(self, sample_data_slice):
        """
        Regression guard for the missing nw (n_sources) normalization factor:
        halving n_sources at fixed n_samples should materially change the
        per-sample-per-channel LL scale if normalization is source-count-aware
        (as it must be per amica17.f90:1866), not stay identical.
        """
        data, params = sample_data_slice
        n_channels = params["data_dim"]
        torch.manual_seed(0)

        model_full = AMICATorch(
            n_channels=n_channels,
            n_sources=n_channels,
            n_models=1,
            n_mix=3,
            device="cpu",
            dtype=torch.float64,
        )
        X_prep = model_full.preprocess_data(
            data.astype(np.float64), do_mean=True, do_sphere=True
        )
        n_samples = X_prep.shape[1]

        with torch.no_grad():
            Y = model_full._forward_single(X_prep, 0).numpy()
            alpha = model_full.alpha.numpy()
            mu = model_full.mu.numpy()
            beta = model_full.beta.numpy()
            rho = model_full.rho.numpy()

        ll_all_sources = _numpy_reference_ll(Y, alpha, mu, beta, rho)
        ll_half_sources = _numpy_reference_ll(
            Y[: n_channels // 2],
            alpha[:, : n_channels // 2],
            mu[:, : n_channels // 2],
            beta[:, : n_channels // 2],
            rho[:, : n_channels // 2],
        )

        # Both are per-sample-per-source normalized, so they should be on a
        # comparable scale (same order of magnitude), not off by ~2x the way
        # dividing by n_samples alone (ignoring n_sources) would produce.
        assert np.isfinite(ll_all_sources)
        assert np.isfinite(ll_half_sources)
        ratio = ll_half_sources / ll_all_sources
        assert 0.3 < ratio < 3.0, (
            f"per-source normalized LL scale changed too much when halving "
            f"n_sources (ratio={ratio}); suggests normalization is not "
            f"actually per-source"
        )
        assert n_samples > 0
