"""MLX natural-gradient EM backend for AMICA (issue #76, epic #74 Phase C).

``AMICAMLXNG`` is a v1 MVP port of :class:`pyAMICA.torch_impl.core.AMICATorchNG`
to Apple's MLX array framework, so the per-block E/M-step runs on the Apple GPU.
It is structurally parallel to the PyTorch backend (same method names/order and
Fortran citations) so the two can be diffed method-by-method.

Design constraints (verified against MLX 0.32, see ``.context/mps_pathways.md``):

* **float32 only on the GPU** -- Apple GPUs have no FP64, and MLX raises on GPU
  float64. Full-data float32 converges thanks to the ``ufp/y`` divide-by-zero
  guard from issue #75, which is reproduced here.
* **All ``mlx.core.linalg`` is CPU-stream only** -- ``inv(A)`` and ``slogdet(W)``
  run under ``mx.stream(mx.cpu)``. They are hoisted to once per iteration (the
  PyTorch ``_forward`` recomputes ``slogdet`` inside the block loop; here it is a
  cached constant), so the GPU pipeline sees one cross-stream handoff per
  iteration, not one per block.
* **No ``lgamma``/``digamma`` in MLX** -- the GG normalizer ``lgamma(1+1/rho)``
  and the rho-update ``digamma(1+1/rho)`` depend only on the small ``rho`` array,
  so they are computed host-side with SciPy once per iteration.

MVP scope: single model (``n_models=1``), generalized Gaussian (``pdftype=0``),
natural gradient (``do_newton=False``). Newton, the other PDF families, component
sharing, multi-model, outlier rejection, ``keep_best``, ``transform`` and
save/load are deferred to fast-follows and rejected with a clear error.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import mlx.core as mx
import numpy as np
from scipy.special import digamma, gammaln

logger = logging.getLogger(__name__)

_LOG2 = math.log(2.0)
# Fortran epsdble: zero the rho*ln|y| term when |y|^rho underflows below this
# (amica17.f90:1570), matching AMICATorchNG.
_EPSDBLE = 1e-16
# MLX linalg runs on the CPU stream only (float32-accurate); the GPU stream
# raises "not yet supported on the GPU" for inv/slogdet/eigh/solve.
_CPU = mx.cpu


def _score_gg(y: mx.array, rho: mx.array) -> mx.array:
    """GG score ``fp = rho*sign(y)*|y|^(rho-1)`` (AMICATorchNG ``_score``, GG
    branch, core.py:174-183). ``fp(0)=0`` for ``rho>=1``, which the ``ufp/y``
    guard relies on."""
    abs_y = mx.abs(y)
    sign_y = mx.sign(y)
    fp_gg = rho * sign_y * mx.power(abs_y, rho - 1.0)
    # rho is generically in (1, 2); keep the exact Laplace/Gaussian endpoints.
    return mx.where(rho == 2.0, 2.0 * y, mx.where(rho == 1.0, sign_y, fp_gg))


def _log_pdf_gg(
    y: mx.array, rho: mx.array, lgamma_table: mx.array
) -> tuple[mx.array, mx.array]:
    """GG log-density and ``|y|^rho`` (AMICATorchNG ``_log_pdf_only``, GG branch,
    core.py:214-224). ``lgamma_table = lgamma(1+1/rho)`` is precomputed host-side
    (MLX has no ``lgamma``); it makes the uniform GG form reduce to the exact
    Laplace (rho=1) and Gaussian (rho=2) log-densities."""
    abs_y = mx.abs(y)
    az_rho = mx.power(abs_y, rho)  # reused by the rho-update accumulator
    log_pdf = -az_rho - _LOG2 - lgamma_table
    return log_pdf, az_rho


class AMICAMLXNG:
    """MLX natural-gradient EM backend (single-model GG MVP, issue #76).

    Parameters mirror the subset of :class:`AMICATorchNG` that the MVP supports;
    the same ``seed`` produces the same initial parameters as the PyTorch/NumPy
    backends, so cross-backend equivalence is testable.
    """

    def __init__(
        self,
        n_channels: int,
        n_models: int = 1,
        n_mix: int = 3,
        block_size: int = 512,
        lrate: float = 0.1,
        minlrate: float = 1e-12,
        lratefact: float = 0.5,
        maxdecs: int = 5,
        newt_ramp: int = 10,
        do_newton: bool = False,
        rho0: float = 1.5,
        minrho: float = 1.0,
        maxrho: float = 2.0,
        rholrate: float = 0.05,
        rholratefact: float = 0.1,
        pdftype: int = 0,
        invsigmin: float = 1e-4,
        invsigmax: float = 1000.0,
        doscaling: bool = True,
        scalestep: int = 1,
        do_mean: bool = True,
        do_sphere: bool = True,
        do_approx_sphere: bool = True,
        seed: Optional[int] = None,
    ):
        # --- MVP boundaries: reject the deferred configurations up front. -----
        if n_models != 1:
            raise NotImplementedError(
                "AMICAMLXNG (v1) supports single-model only (n_models=1); "
                "multi-model is a fast-follow. Use AMICATorchNG for n_models>1."
            )
        if pdftype != 0:
            raise NotImplementedError(
                "AMICAMLXNG (v1) supports the generalized-Gaussian family "
                "(pdftype=0) only; the other families are a fast-follow."
            )
        if do_newton:
            raise NotImplementedError(
                "AMICAMLXNG (v1) supports the natural gradient only "
                "(do_newton=False); Newton is a fast-follow."
            )

        self.n_channels = n_channels
        self.n_models = 1
        self.n_mix = n_mix
        self.n_comps = n_channels
        self.block_size = block_size

        self.lrate0 = lrate
        self.lrate = lrate
        self.lrate_cap = lrate
        self.minlrate = minlrate
        self.lratefact = lratefact
        self.maxdecs = maxdecs
        self.newt_ramp = newt_ramp
        self.do_newton = False

        self.rho0 = rho0
        self.minrho = minrho
        self.maxrho = maxrho
        self.rholrate0 = rholrate
        self.rholrate = rholrate
        self.rholratefact = rholratefact
        self.dorho = True  # GG shape adapts (Fortran dorho, pdftype==0)

        self.pdftype = 0
        self.invsigmin = invsigmin
        self.invsigmax = invsigmax
        self.doscaling = doscaling
        self.scalestep = scalestep

        self.do_mean = do_mean
        self.do_sphere = do_sphere
        self.do_approx_sphere = do_approx_sphere
        self.seed = seed

        self.iteration = 0
        self.ll_history: list[float] = []
        self.final_ll_: Optional[float] = None
        self.stop_reason: Optional[str] = None

        # Populated by fit()/_initialize_parameters().
        self.A = self.W = self.mu = self.alpha = self.beta = self.rho = None
        self.gm = self.mean = self.sphere = None
        self.sldet = 0.0
        self._lgamma_table = None  # mx.array (n_mix, n_channels): lgamma(1+1/rho)
        self._logdet_W = None  # mx.array scalar: log|det W|, refreshed per iter

    _DEGENERATE_STOP_REASONS = ("nan_ll", "singular_ll")

    # ------------------------------------------------------------------
    # Preprocessing (host / numpy; mirrors AMICATorchNG._preprocess in float64)
    # ------------------------------------------------------------------
    def _preprocess(self, X: np.ndarray) -> mx.array:
        """Mean-removal + sphering in float64 on the host, then handed to MLX as
        float32. Done in numpy (not MLX) because MLX ``eigh`` is CPU-only and
        float32; keeping it float64-on-host makes the sphere/sldet match the
        PyTorch backend (both derive from the same float64 eigh)."""
        Xc = np.ascontiguousarray(X).astype(np.float64)
        data_dim = Xc.shape[0]

        if self.do_mean:
            mean = Xc.mean(axis=1, keepdims=True)
            Xc = Xc - mean
        else:
            mean = np.zeros((data_dim, 1))

        if self.do_sphere:
            # Population covariance (/N), matching Fortran's DSYRK scatter, not
            # numpy's default sample covariance (/(N-1)) -- see core.py:635-639.
            cov = np.cov(Xc, bias=True)
            evals, evecs = np.linalg.eigh(cov)
            order = np.argsort(evals)[::-1]
            evals = evals[order]
            evecs = evecs[:, order]
            inv_sqrt = np.diag(1.0 / np.sqrt(evals))
            if self.do_approx_sphere:
                # Symmetric ZCA sphere V diag(1/sqrt) V^T (Fortran default).
                sphere = evecs @ inv_sqrt @ evecs.T
            else:
                sphere = inv_sqrt @ evecs.T
            Xc = sphere @ Xc
            sldet = float(-0.5 * np.log(evals).sum())
        else:
            sphere = np.eye(data_dim)
            sldet = 0.0

        self.mean = mx.array(mean.astype(np.float32))
        self.sphere = mx.array(sphere.astype(np.float32))
        self.sldet = sldet
        return mx.array(Xc.astype(np.float32))

    # ------------------------------------------------------------------
    # Initialization (identical RNG draws to AMICATorchNG for cross-backend test)
    # ------------------------------------------------------------------
    def _initialize_parameters(self):
        """Initialize parameters with the *same* ``np.random.RandomState`` draw
        order as AMICATorchNG/AMICA_NumPy (core.py:688-725), so a shared seed
        gives a bit-identical (float32-cast) starting point."""
        rng = np.random.RandomState(self.seed)
        n, ncomp, nmix = self.n_channels, self.n_comps, self.n_mix

        A_np = np.eye(n) + 0.01 * (0.5 - rng.rand(n, n))

        mu_np = np.zeros((nmix, ncomp))
        for k in range(ncomp):
            mu_np[:, k] = np.linspace(-1, 1, nmix)
            mu_np[:, k] += 0.05 * (1 - 2 * rng.rand(nmix))

        alpha_np = np.ones((nmix, ncomp)) / nmix
        beta_np = np.ones((nmix, ncomp)) + 0.1 * (0.5 - rng.rand(nmix, ncomp))
        rho_np = self.rho0 * np.ones((nmix, ncomp))

        self.A = mx.array(A_np.astype(np.float32))
        self.mu = mx.array(mu_np.astype(np.float32))
        self.alpha = mx.array(alpha_np.astype(np.float32))
        self.beta = mx.array(beta_np.astype(np.float32))
        self.rho = mx.array(rho_np.astype(np.float32))
        self.gm = mx.array(np.ones(1, dtype=np.float32))

        self.lrate = self.lrate0
        self.lrate_cap = self.lrate0
        self.rholrate = self.rholrate0
        self.iteration = 0
        self._refresh_lgamma_table()
        self._update_unmixing_matrices()

    def _refresh_lgamma_table(self):
        """Recompute ``lgamma(1+1/rho)`` host-side (MLX has no lgamma). Called at
        init and after every rho update. Cheap: rho is ``(n_mix, n_channels)``."""
        rho_np = np.array(self.rho, dtype=np.float64)
        self._lgamma_table = mx.array(gammaln(1.0 + 1.0 / rho_np).astype(np.float32))

    def _update_unmixing_matrices(self):
        """W = inv(A) and the LL Jacobian log|det W|, both on the CPU stream
        (MLX linalg is CPU-only) and hoisted to once per iteration."""
        self.W = mx.linalg.inv(self.A, stream=_CPU)
        self._logdet_W = mx.linalg.slogdet(self.W, stream=_CPU)[1]

    # ------------------------------------------------------------------
    # E-step
    # ------------------------------------------------------------------
    def _forward(self, Xb: mx.array):
        """E-step forward pass for one block (single model). ``Xb`` is
        ``(n_channels, batch)``. Returns ``(logV, b, z, y, az_rho)``."""
        b = Xb.T @ self.W  # (batch, n_channels); c == 0 for single model
        mu_h = self.mu.T[None]  # (1, n_channels, n_mix)
        beta_h = self.beta.T[None]
        rho_h = self.rho.T[None]
        alpha_h = self.alpha.T[None]
        lgamma_h = self._lgamma_table.T[None]

        y = beta_h * (b[..., None] - mu_h)  # (batch, n_channels, n_mix)
        log_pdf, az_rho = _log_pdf_gg(y, rho_h, lgamma_h)
        z0 = mx.log(alpha_h) + mx.log(beta_h) + log_pdf
        ll_i = mx.logsumexp(z0, axis=-1)  # (batch, n_channels)
        z = mx.softmax(z0, axis=-1)
        # Single model: log(gm)=0; logdet_W + sldet are the Jacobian terms.
        logV = self._logdet_W + self.sldet + ll_i.sum(axis=-1)  # (batch,)
        return logV, b, z, y, az_rho

    def _get_block_updates(self, Xb: mx.array) -> dict:
        """Exact-EM sufficient statistics for one block (single-model,
        non-Newton subset of AMICATorchNG._get_block_updates, core.py:887-947)."""
        logV, b, z, y, az_rho = self._forward(Xb)
        block_ll = logV.sum()  # single model: logsumexp over one model is identity
        beta_h = self.beta.T[None]  # (1, n_channels, n_mix)
        rho_h = self.rho.T[None]

        fp = _score_gg(y, rho_h)
        u = z  # u = v*z with v == 1 (single model)
        ufp = u * fp

        dgm = mx.array(float(Xb.shape[1]), dtype=mx.float32)  # sum_t v_h = n samples
        dalpha_n = u.sum(0).T  # (n_mix, n_channels)
        dmu_n = ufp.sum(0).T
        # Phase A guard: float32 can round y to exactly 0 (fp(0)=0 => ufp=0), so
        # ufp/y is 0/0=NaN; where y==0, 0/1 contributes 0 (issue #75).
        safe_y = mx.where(y == 0, mx.ones_like(y), y)
        dmu_d = (beta_h[0] * (ufp / safe_y).sum(0)).T
        dbeta_n = u.sum(0).T
        dbeta_d = (ufp * y).sum(0).T

        ay = mx.abs(y)
        tiny = float(np.finfo(np.float32).tiny)
        logab = rho_h * mx.log(mx.maximum(ay, tiny))
        logab = mx.where(az_rho < _EPSDBLE, mx.zeros_like(logab), logab)
        drho_n = (u * (az_rho * logab)).sum(0).T

        g = (beta_h * ufp).sum(-1)  # (batch, n_channels)
        dWtmp = g.T @ b  # (n_channels, n_channels)

        return {
            "dgm": dgm,
            "dalpha_n": dalpha_n,
            "dmu_n": dmu_n,
            "dmu_d": dmu_d,
            "dbeta_n": dbeta_n,
            "dbeta_d": dbeta_d,
            "drho_n": drho_n,
            "dWtmp": dWtmp,
            "ll": block_ll,
        }

    def _accumulate_blocks(self, X: mx.array) -> dict:
        """Sum sufficient statistics over all blocks as one lazy graph (no
        per-block ``mx.eval`` -- that over-syncs 2.6x)."""
        n_samples = X.shape[1]
        acc: Optional[dict] = None
        for start in range(0, n_samples, self.block_size):
            end = min(start + self.block_size, n_samples)
            block_acc = self._get_block_updates(X[:, start:end])
            if acc is None:
                acc = block_acc
            else:
                for key in acc:
                    acc[key] = acc[key] + block_acc[key]
        return acc

    # ------------------------------------------------------------------
    # M-step
    # ------------------------------------------------------------------
    def _update_parameters(self, acc: dict, n_samples: int):
        """Exact-EM mixture updates + natural-gradient A-update (single-model,
        non-Newton subset of AMICATorchNG._update_parameters, core.py:1049-1247)."""
        self.gm = acc["dgm"] / n_samples  # == 1 for single model

        self.alpha = acc["dalpha_n"] / acc["dalpha_n"].sum(axis=0, keepdims=True)
        self.mu = self.mu + acc["dmu_n"] / acc["dmu_d"]
        self.beta = mx.clip(
            self.beta * mx.sqrt(acc["dbeta_n"] / acc["dbeta_d"]),
            self.invsigmin,
            self.invsigmax,
        )

        # GG shape update with the 1/psi(1+1/rho) digamma factor (Fortran
        # :2013-2014); digamma is computed host-side (MLX has none).
        if self.dorho:
            drho = acc["drho_n"] / mx.maximum(acc["dalpha_n"], 1e-8)
            rho_np = np.array(self.rho, dtype=np.float64)
            psi = mx.array(digamma(1.0 + 1.0 / rho_np).astype(np.float32))
            new_rho = self.rho + self.rholrate * (1.0 - (self.rho / psi) * drho)
            self.rho = mx.clip(new_rho, self.minrho, self.maxrho)

        # Natural-gradient A-update. A is stored as Fortran's A^T, so the update
        # is a LEFT-multiply by the transposed direction (core.py:1156-1164,
        # #24 root cause). Single-model collapses the gm-weighted column average.
        eye = mx.eye(self.n_channels)
        dA = -acc["dWtmp"] / acc["dgm"] + eye  # I - <g b^T>/dgm
        self.lrate = min(
            self.lrate_cap, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
        )
        self.A = self.A - self.lrate * (dA.T @ self.A)

        if self.doscaling and (self.iteration % self.scalestep == 0):
            scale = mx.sqrt((self.A**2).sum(axis=0))  # (n_comps,)
            safe_scale = mx.where(scale > 0, scale, mx.ones_like(scale))
            self.A = self.A / safe_scale
            self.mu = self.mu * scale
            self.beta = self.beta / safe_scale

        self._refresh_lgamma_table()
        self._update_unmixing_matrices()

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------
    def fit(
        self, X: np.ndarray, max_iter: int = 100, verbose: bool = True
    ) -> "AMICAMLXNG":
        """Fit the model. ``X`` is ``(n_channels, n_samples)``."""
        if X.ndim != 2:
            raise ValueError(f"X must be 2D (n_channels, n_samples), got {X.shape}")
        if X.shape[0] != self.n_channels:
            raise ValueError(
                f"X has {X.shape[0]} channels, model expects {self.n_channels}"
            )

        X_t = self._preprocess(X)
        n_total = X_t.shape[1]
        self._initialize_parameters()
        self.ll_history = []
        self.stop_reason = "max_iter"
        numdecs = 0

        rng = range(max_iter)
        if verbose:
            try:
                from tqdm import tqdm

                rng = tqdm(rng, desc="AMICA-MLX")
            except ImportError:
                pass

        for it in rng:
            self.iteration = it
            acc = self._accumulate_blocks(X_t)

            ll_arr = acc["ll"] / (n_total * self.n_channels)
            mx.eval(ll_arr)  # materialize the accumulate graph once
            ll = float(ll_arr.item())
            if not math.isfinite(ll):
                self.stop_reason = "nan_ll" if math.isnan(ll) else "singular_ll"
                logger.warning(
                    "Non-finite log-likelihood (%s) at iteration %d; stopping.", ll, it
                )
                break

            self._update_parameters(acc, n_total)
            # One eval per iteration bounds the lazy graph to a single iteration's
            # worth of ops (the updated params feed the next accumulate).
            mx.eval(self.A, self.W, self.mu, self.alpha, self.beta, self.rho)

            self.ll_history.append(ll)

            # Learning-rate control (Fortran amica17.f90:1062-1108): anneal on an
            # LL decrease; ratchet the ceiling after maxdecs persistent decreases.
            if len(self.ll_history) > 1 and ll < self.ll_history[-2]:
                if self.lrate <= self.minlrate:
                    logger.warning(
                        "lrate floor (%g) reached at iter %d; stopping.",
                        self.minlrate,
                        it,
                    )
                    self.stop_reason = "lrate_floor"
                    break
                self.lrate *= self.lratefact
                self.rholrate *= self.rholratefact
                numdecs += 1
                if numdecs >= self.maxdecs:
                    self.lrate_cap *= self.lratefact
                    numdecs = 0

        if self.stop_reason in self._DEGENERATE_STOP_REASONS:
            self.final_ll_ = float("nan")
        else:
            self.final_ll_ = self.ll_history[-1] if self.ll_history else float("nan")
        return self
