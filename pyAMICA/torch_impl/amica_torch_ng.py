"""
Natural-gradient EM PyTorch backend for AMICA (ADR 0001).

Unlike ``amica_torch.py``/``amica_torch_v2.py``, which reframe AMICA as
"minimize negative log-likelihood with Adam over reparameterized tensors",
this module is a direct, vectorized port of the closed-form E-step/M-step
fixed-point updates used by the Fortran reference (``amica17.f90``) and the
legacy NumPy implementation (``pyAMICA.pyAMICA.AMICA._get_block_updates`` /
``_update_parameters``, which is this module's line-by-line spec). There is
no autograd and no Adam: every parameter update is a closed-form function of
the E-step responsibilities and simple moments, matching the natural-gradient
EM fixed point instead of a different, Adam-driven trajectory.

Key design points (see ``.context/decisions/0001-torch-backend-natural-gradient-em.md``):

* ``W`` (and ``A``) are stored and mutated directly; ``W`` is recomputed from
  ``A`` once per iteration via a batched ``torch.linalg.inv`` (matching
  ``amica_utils.get_unmixing_matrices``), never via ``pinv`` in the hot path.
* The E-step is vectorized over ``(model, mix, source)`` via broadcasting;
  the only Python loops are over models (typically 1-3) and over blocks.
* Samples are processed in blocks and sufficient statistics are accumulated
  across blocks, so peak memory scales with ``block_size``, not with the
  total number of samples.
* Parameters default to float64 for numerical parity with Fortran's
  double-precision arithmetic; float32 is available for speed.

Log-likelihood: this module computes the per-source log-likelihood from the
pre-normalization mixture logits via ``logsumexp`` plus the ``log|det W|`` +
``sldet`` Jacobian (matching ``amica17.f90:1341-1350``), the mathematically
correct per-source log-density required to hit the Fortran-normalized LL target
(~-3.4/sample-channel). As of issue #24 the legacy NumPy port
(``pyAMICA.pyAMICA.AMICA``) computes it the same way; both backends now converge
to the Fortran solution (component correlation > 0.95).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, Optional, Union

import numpy as np
import torch
from tqdm import tqdm

from .utils import setup_device

logger = logging.getLogger(__name__)

_LOG2 = math.log(2.0)
_HALF_LOG_PI = 0.5 * math.log(math.pi)


def _log_pdf_and_deriv(
    y: torch.Tensor, rho: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized generalized-Gaussian log-density and density derivative.

    Elementwise port of ``pyAMICA.pyAMICA.AMICA._compute_log_pdf``: branches
    on ``rho`` via ``torch.where`` (Laplace/Gaussian/generalized-Gaussian)
    instead of Python control flow so it runs over full
    ``(block, source, mixture)`` tensors with no source/mixture loop. ``y``
    and ``rho`` must be broadcastable to a common shape.
    """
    abs_y = y.abs()
    sign_y = torch.sign(y)

    log_pdf_lap = -abs_y - _LOG2
    dpdf_lap = -sign_y * torch.exp(log_pdf_lap)

    log_pdf_gau = -y * y - _HALF_LOG_PI
    dpdf_gau = -2.0 * y * torch.exp(log_pdf_gau)

    log_pdf_gg = -abs_y.pow(rho) - _LOG2 - torch.lgamma(1.0 + 1.0 / rho)
    dpdf_gg = -rho * abs_y.pow(rho - 1.0) * sign_y * torch.exp(log_pdf_gg)

    is_lap = rho == 1.0
    is_gau = rho == 2.0

    log_pdf = torch.where(
        is_gau, log_pdf_gau, torch.where(is_lap, log_pdf_lap, log_pdf_gg)
    )
    dpdf = torch.where(is_gau, dpdf_gau, torch.where(is_lap, dpdf_lap, dpdf_gg))
    return log_pdf, dpdf


def _score(y: torch.Tensor, rho: torch.Tensor) -> torch.Tensor:
    """Generalized-Gaussian score ``fp = d|y|^rho/dy`` (Fortran ``fp``).

    This is the derivative of the *negative* log-density, ``fp(y) =
    rho*sign(y)*|y|^(rho-1)`` (``sign(y)`` for Laplace, ``2y`` for Gaussian),
    used by the Newton sufficient statistics (``amica17.f90:1455-1467``). It
    is distinct from the density derivative ``dpdf`` returned by
    ``_log_pdf_and_deriv`` (which carries an extra ``pdf`` factor); the Newton
    curvature terms ``kappa``/``lambda`` are defined in terms of ``fp``, not
    ``dpdf``.
    """
    abs_y = y.abs()
    sign_y = torch.sign(y)
    fp_lap = sign_y
    fp_gau = 2.0 * y
    fp_gg = rho * sign_y * abs_y.pow(rho - 1.0)
    is_lap = rho == 1.0
    is_gau = rho == 2.0
    return torch.where(is_gau, fp_gau, torch.where(is_lap, fp_lap, fp_gg))


class AMICATorchNG:
    """
    Natural-gradient EM AMICA, ported from ``pyAMICA.pyAMICA.AMICA``.

    Not an ``nn.Module``: there are no learnable ``nn.Parameter``s and no
    autograd. Parameters (``A``, ``W``, ``c``, ``mu``, ``alpha``, ``beta``,
    ``rho``, ``gm``) are plain tensors mutated in place by closed-form
    E-step/M-step updates each iteration, mirroring
    ``pyAMICA.AMICA._get_block_updates``/``_update_parameters``.

    Parameters
    ----------
    n_channels : int
        Number of input channels (``data_dim`` in the NumPy/Fortran code).
    n_models : int, default=1
        Number of ICA mixture models.
    n_mix : int, default=3
        Number of mixture components per source.
    block_size : int, default=128
        Number of samples processed per accumulation block. Peak memory
        during the E-step scales with this, not with the total sample count.
    lrate : float, default=0.1
        Initial/maximum natural-gradient learning rate (``lrate0`` in NumPy).
    minlrate : float, default=1e-12
        Hard learning-rate floor: once ``lrate`` anneals to it, ``fit`` stops
        (``stop_reason="lrate_floor"``).
    lratefact : float, default=0.5
        Factor by which ``lrate`` (and the ceiling ``lrate_cap``/``newtrate``)
        are annealed when the log-likelihood decreases; see ``fit`` for the
        Fortran-style ``numdecs``/``maxdecs`` ratchet.
    maxdecs : int, default=5
        Number of consecutive log-likelihood decreases after which the
        learning-rate *ceiling* is ratcheted down (Fortran ``maxdecs``).
    newt_ramp : int, default=10
        Denominator of the per-iteration learning-rate ramp toward the current
        ceiling: ``lrate = min(ceiling, lrate + min(1/newt_ramp, lrate))``
        (ceiling is ``lrate_cap`` for natural gradient, ``newtrate`` for
        Newton).
    do_newton : bool, default=False
        Enable the Newton preconditioner for the ``A``/``W`` update once
        ``iteration >= newt_start``. Ported from the Fortran reference
        (``amica17.f90``): natural gradient alone plateaus well short of the
        Fortran solution, and the Newton step (a per-source-pair 2x2 solve
        preconditioning the natural gradient by an approximate Hessian) is
        what closes the gap.
    newt_start : int, default=20
        Iteration at which the Newton step switches on (natural gradient is
        used before it, letting the mixture parameters settle first).
    newtrate : float, default=0.5
        Maximum learning rate the ramp climbs to while Newton is active
        (the natural-gradient phase is capped at ``lrate``/``lrate0``).
    do_reject : bool, default=False
        Enable Fortran-style outlier rejection: after the parameter update,
        samples whose total log-likelihood falls below
        ``mean - rejsig*std`` are permanently excluded from subsequent
        sufficient-statistic accumulation and from the sample count used to
        normalize ``gm`` and the reported log-likelihood.
    rejsig : float, default=3.0
        Rejection threshold in standard deviations of the per-sample
        log-likelihood.
    rejstart, rejint, maxrej : int
        First rejection iteration, interval between rejections, and maximum
        number of rejection passes (matching ``amica17.f90:1141-1146``).
    rho0, minrho, maxrho, rholrate : float
        Generalized-Gaussian shape-parameter initialization, clamp bounds,
        and learning rate.
    invsigmin, invsigmax : float
        Clamp bounds for the mixture scale parameter ``beta``.
    doscaling, scalestep : bool, int
        Whether/how often to rescale ``A`` columns to unit norm each
        iteration (with matching ``mu``/``beta`` rescale).
    do_mean, do_sphere, do_approx_sphere : bool
        Preprocessing options, matching ``pyAMICA.AMICA._preprocess_data``.
    pcakeep, pcadb : int, float, optional
        PCA dimensionality-reduction options (rarely used; see
        ``pyAMICA.AMICA._preprocess_data``).
    seed : int, optional
        Seed for parameter initialization. Uses ``numpy.random.RandomState``
        internally (not ``torch``'s RNG) with the exact same draw order as
        ``pyAMICA.AMICA._initialize_parameters``, so the same seed produces
        bit-identical starting parameters to the NumPy reference.
    device : str or torch.device, optional
        Compute device for the block loop. Preprocessing (mean/cov/eigh) is
        always done in float64 on CPU regardless of device, since eigh is
        not reliably supported on MPS.
    dtype : torch.dtype, default=torch.float64
        Parameter/computation dtype. float64 is required for MPS to raise
        (MPS does not support float64); use dtype=torch.float32 for MPS.
    """

    def __init__(
        self,
        n_channels: int,
        n_models: int = 1,
        n_mix: int = 3,
        block_size: int = 128,
        lrate: float = 0.1,
        minlrate: float = 1e-12,
        lratefact: float = 0.5,
        maxdecs: int = 5,
        newt_ramp: int = 10,
        do_newton: bool = False,
        newt_start: int = 20,
        newtrate: float = 0.5,
        do_reject: bool = False,
        rejsig: float = 3.0,
        rejstart: int = 2,
        rejint: int = 3,
        maxrej: int = 1,
        rho0: float = 1.5,
        minrho: float = 1.0,
        maxrho: float = 2.0,
        rholrate: float = 0.05,
        rholratefact: float = 0.1,
        invsigmin: float = 1e-4,
        invsigmax: float = 1000.0,
        doscaling: bool = True,
        scalestep: int = 1,
        do_mean: bool = True,
        do_sphere: bool = True,
        do_approx_sphere: bool = True,
        pcakeep: Optional[int] = None,
        pcadb: Optional[float] = None,
        seed: Optional[int] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: torch.dtype = torch.float64,
    ):
        self.n_channels = n_channels
        self.n_models = n_models
        self.n_mix = n_mix
        self.n_comps = n_channels * n_models
        self.block_size = block_size

        self.lrate0 = lrate
        self.lrate = lrate
        self.minlrate = minlrate
        self.lratefact = lratefact
        self.maxdecs = maxdecs
        self.newt_ramp = newt_ramp

        self.do_newton = do_newton
        self.newt_start = newt_start
        self.newtrate = newtrate
        self.newtrate0 = newtrate

        self.do_reject = do_reject
        self.rejsig = rejsig
        self.rejstart = rejstart
        self.rejint = rejint
        self.maxrej = maxrej
        if do_reject:
            if rejint < 1:
                raise ValueError(f"rejint must be >= 1, got {rejint}")
            if rejsig <= 0:
                raise ValueError(f"rejsig must be > 0, got {rejsig}")
            if maxrej < 0:
                raise ValueError(f"maxrej must be >= 0, got {maxrej}")

        self.rho0 = rho0
        self.minrho = minrho
        self.maxrho = maxrho
        self.rholrate = rholrate
        self.rholrate0 = rholrate
        self.rholratefact = rholratefact

        self.invsigmin = invsigmin
        self.invsigmax = invsigmax

        self.doscaling = doscaling
        self.scalestep = scalestep

        self.do_mean = do_mean
        self.do_sphere = do_sphere
        self.do_approx_sphere = do_approx_sphere
        self.pcakeep = pcakeep
        self.pcadb = pcadb

        self.seed = seed

        if device is None:
            device = setup_device()
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device
        self.dtype = dtype

        if self.device.type == "mps" and self.dtype == torch.float64:
            raise ValueError(
                "MPS does not support float64. Use dtype=torch.float32 for "
                "device='mps', or device='cpu'/'cuda' for float64 parity runs."
            )

        self.iteration = 0
        self.ll_history: list[float] = []

        # Outlier-rejection bookkeeping (set up in fit()).
        self.numrej = 0
        self.good_idx: Optional[torch.Tensor] = None

        # Set by fit(): why fitting stopped ("max_iter", "nan_ll", "lrate_floor")
        # and how many iterations reverted Newton to natural gradient (Fortran
        # prints this; here it is exposed for parity debugging, see issue #21).
        self.stop_reason: Optional[str] = None
        self.n_newton_fallbacks = 0

        # Populated by fit()/_initialize_parameters().
        self.A = self.W = self.c = None
        self.mu = self.alpha = self.beta = self.rho = None
        self.gm = self.comp_list = None
        self.mean = self.sphere = None
        self.sldet = 0.0

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------
    def _preprocess(self, X: np.ndarray) -> torch.Tensor:
        """Mean-removal + sphering, matching ``pyAMICA.AMICA._preprocess_data``.

        Done in float64 on CPU (eigh is not reliably supported on MPS and
        this is a one-time O(n_channels^3) cost, not the per-block hot
        path), then cast/moved to ``self.device``/``self.dtype``.
        """
        X_cpu = torch.from_numpy(np.ascontiguousarray(X)).to(torch.float64)
        data_dim = X_cpu.shape[0]

        if self.do_mean:
            mean = X_cpu.mean(dim=1, keepdim=True)
            X_cpu = X_cpu - mean
        else:
            mean = torch.zeros(data_dim, 1, dtype=torch.float64)

        if self.do_sphere:
            # Population covariance (divide by N), matching Fortran's DSYRK
            # scatter/N -- NOT torch.cov's default sample covariance (/(N-1)).
            # The two differ by a pure scalar sqrt(N/(N-1)); using /(N-1) leaves
            # a ~5e-6 sphere mismatch vs the reference (issue #24, check [1] of
            # .context/issue-24/root_cause_Aupdate.py).
            cov = torch.cov(X_cpu, correction=0)
            evals, evecs = torch.linalg.eigh(cov)
            order = torch.argsort(evals, descending=True)
            evals = evals[order]
            evecs = evecs[:, order]

            if self.pcakeep is not None:
                n_comp = min(self.pcakeep, evals.shape[0])
            elif self.pcadb is not None:
                db = 10.0 * torch.log10(evals / evals[0])
                n_comp = int((db > -self.pcadb).sum().item())
            else:
                n_comp = evals.shape[0]

            V = evecs[:, :n_comp]
            inv_sqrt = torch.diag(1.0 / torch.sqrt(evals[:n_comp]))
            if self.do_approx_sphere:
                # Symmetric ZCA sphere V diag(1/sqrt(eval)) V^T (Fortran
                # do_approx_sphere=True, amica17.f90:480-481). This is the
                # Fortran default and the parity-validated form; the old
                # diag(1/sqrt)@V^T (PCA whitening) is a different, non-symmetric
                # transform that breaks activation parity.
                sphere = V @ inv_sqrt @ V.T
            else:
                # Non-symmetric PCA whitening D^-1/2 V^T (Fortran
                # do_approx_sphere=False path, amica17.f90:495).
                sphere = inv_sqrt @ V.T

            X_cpu = sphere @ X_cpu
            # Sphering log-determinant term of the data log-likelihood
            # (Fortran ``sldet``, amica17.f90:474): sum over the kept
            # eigenvalues of -0.5*log(eval). For the PCA-reduced-rank case
            # this is a pseudo-determinant, matching Fortran which sums over
            # numeigs kept eigenvalues regardless of full rank.
            sldet = float(-0.5 * torch.log(evals[:n_comp]).sum().item())
        else:
            sphere = torch.eye(data_dim, dtype=torch.float64)
            sldet = 0.0

        self.mean = mean.to(device=self.device, dtype=self.dtype)
        self.sphere = sphere.to(device=self.device, dtype=self.dtype)
        self.sldet = sldet

        return X_cpu.to(device=self.device, dtype=self.dtype)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------
    def _initialize_parameters(self):
        """Initialize parameters, mirroring ``pyAMICA.AMICA._initialize_parameters``
        exactly (same RNG draws, same order) so the same seed gives
        bit-identical starting parameters to the NumPy reference.
        """
        rng = np.random.RandomState(self.seed)
        n, m, ncomp, nmix = self.n_channels, self.n_models, self.n_comps, self.n_mix

        A_np = np.zeros((n, ncomp), dtype=np.float64)
        for h in range(m):
            A_np[:, h * n : (h + 1) * n] = np.eye(n) + 0.01 * (0.5 - rng.rand(n, n))

        comp_list_np = np.zeros((n, m), dtype=np.int64)
        for h in range(m):
            comp_list_np[:, h] = np.arange(h * n, (h + 1) * n)

        mu_np = np.zeros((nmix, ncomp), dtype=np.float64)
        for k in range(ncomp):
            mu_np[:, k] = np.linspace(-1, 1, nmix)
            mu_np[:, k] += 0.05 * (1 - 2 * rng.rand(nmix))

        alpha_np = np.ones((nmix, ncomp), dtype=np.float64) / nmix

        beta_np = np.ones((nmix, ncomp), dtype=np.float64)
        beta_np += 0.1 * (0.5 - rng.rand(nmix, ncomp))

        rho_np = self.rho0 * np.ones((nmix, ncomp), dtype=np.float64)
        gm_np = np.ones(m, dtype=np.float64) / m
        c_np = np.zeros((n, m), dtype=np.float64)

        self.A = torch.from_numpy(A_np).to(self.device, self.dtype)
        self.comp_list = torch.from_numpy(comp_list_np).to(self.device)
        self.mu = torch.from_numpy(mu_np).to(self.device, self.dtype)
        self.alpha = torch.from_numpy(alpha_np).to(self.device, self.dtype)
        self.beta = torch.from_numpy(beta_np).to(self.device, self.dtype)
        self.rho = torch.from_numpy(rho_np).to(self.device, self.dtype)
        self.gm = torch.from_numpy(gm_np).to(self.device, self.dtype)
        self.c = torch.from_numpy(c_np).to(self.device, self.dtype)

        # Reset the mutable optimization state to the pristine constructor
        # values (lrate_cap, newtrate, rholrate are ratcheted down during
        # fit; restore them so a re-fit starts fresh).
        self.lrate = self.lrate0
        self.lrate_cap = self.lrate0
        self.newtrate = self.newtrate0
        self.rholrate = self.rholrate0
        self.iteration = 0
        self._update_unmixing_matrices()

    def _update_unmixing_matrices(self):
        """Recompute W from A via direct (batched) inversion -- never pinv."""
        A_stack = torch.stack(
            [self.A[:, self.comp_list[:, h]] for h in range(self.n_models)], dim=0
        )
        W_stack = torch.linalg.inv(A_stack)
        self.W = W_stack.permute(1, 2, 0).contiguous()

    # ------------------------------------------------------------------
    # E-step / M-step sufficient statistics (the hot path)
    # ------------------------------------------------------------------
    def _forward(self, X: torch.Tensor):
        """Run the E-step forward pass for one data block.

        Computes, for every model ``h``, the activations ``b``, scaled
        activations ``y``, normalized mixture responsibilities ``z``, the
        density derivative ``dpdf``, and the per-sample per-model
        log-likelihood ``logV`` (including the ``log|det W|`` and ``sldet``
        Jacobian terms, matching Fortran's ``Ptmp`` seed, amica17.f90:1273).
        Shared by ``_get_block_updates`` (which reduces it into sufficient
        statistics) and ``_block_sample_ll`` (which only needs ``logV``).

        Returns
        -------
        logV : torch.Tensor of shape (batch, n_models)
        b_list, z_list, y_list, dpdf_list : lists (one entry per model) of
            per-model tensors (``b``: (batch, n_channels); ``z``/``y``/``dpdf``:
            (batch, n_channels, n_mix)).
        """
        batch_size = X.shape[1]
        num_models = self.n_models
        b_list, z_list, y_list, dpdf_list = [], [], [], []
        logV = torch.empty(batch_size, num_models, dtype=self.dtype, device=self.device)

        for h in range(num_models):
            idx = self.comp_list[:, h]
            b = X.T @ self.W[:, :, h] - self.c[:, h]  # (batch, n_channels)

            mu_h = self.mu[:, idx].T.unsqueeze(0)  # (1, n_channels, num_mix)
            beta_h = self.beta[:, idx].T.unsqueeze(0)
            rho_h = self.rho[:, idx].T.unsqueeze(0)
            alpha_h = self.alpha[:, idx].T.unsqueeze(0)

            y = beta_h * (b.unsqueeze(-1) - mu_h)  # (batch, n_channels, num_mix)
            log_pdf, dpdf = _log_pdf_and_deriv(y, rho_h)

            z0 = torch.log(alpha_h) + torch.log(beta_h) + log_pdf
            ll_i = torch.logsumexp(
                z0, dim=-1
            )  # (batch, n_channels) -- per-source log-density
            z = torch.softmax(z0, dim=-1)  # normalized responsibilities

            logdet_W = torch.linalg.slogdet(self.W[:, :, h])[1]
            logV[:, h] = (
                torch.log(self.gm[h]) + logdet_W + self.sldet + ll_i.sum(dim=-1)
            )

            b_list.append(b)
            z_list.append(z)
            y_list.append(y)
            dpdf_list.append(dpdf)

        return logV, b_list, z_list, y_list, dpdf_list

    def _block_sample_ll(self, X: torch.Tensor) -> torch.Tensor:
        """Per-sample total log-likelihood for a data block (the rejection
        statistic; Fortran ``P``/``loglik``, amica17.f90:1372)."""
        logV, *_ = self._forward(X)
        return torch.logsumexp(logV, dim=1)  # (batch,)

    def _get_block_updates(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute sufficient-statistic accumulators for one data block.

        Fortran-faithful exact-EM statistics (amica17.f90:1437-1592), validated
        against the reference binary to machine precision (issue #24). Unlike a
        first-order gradient M-step, the mixture updates use exact-EM numerator/
        denominator pairs and the score ``fp = rho*sign(y)*|y|^(rho-1)`` (``_score``,
        Fortran ``fp``) rather than the density derivative ``dpdf``:

        * ``dmu_n = sum(u*fp)``, ``dmu_d = sbeta*sum(u*fp/y)``   (mu += dmu_n/dmu_d)
        * ``dbeta_n = sum(u)``, ``dbeta_d = sum(u*fp*y)``        (beta *= sqrt(n/d))
        * ``drho_n = rho*sum(u*|y|^rho*ln|y|)``                  (rho digamma update)
        * ``dWtmp = g^T b`` with ``g = sum_j sbeta*u*fp``        (natural gradient)

        where ``u = v*z`` (model x mixture responsibility). ``ll`` is the correct
        pre-normalization ``logsumexp`` (see module docstring).

        Assumes ``rho <= 2`` (the ``maxrho`` default); the ``rho > 2`` denominator
        branches of Fortran (:1539/:1551) are unreachable and not implemented.

        Returns
        -------
        updates : dict with ``dgm`` (n_models,), ``dalpha_n``/``dmu_n``/``dmu_d``/
            ``dbeta_n``/``dbeta_d``/``drho_n`` (n_mix, n_comps), ``dWtmp``
            (n_channels, n_channels, n_models), ``dc`` (n_channels, n_models),
            ``ll`` (scalar), and -- when ``do_newton`` -- ``dsigma2_numer``,
            ``dkappa_numer``, ``dlambda_numer`` (see ``_finalize_newton_stats``).
        """
        num_mix, num_models = self.n_mix, self.n_models
        dev, dt = self.device, self.dtype

        logV, b_list, z_list, y_list, _ = self._forward(X)
        block_ll = torch.logsumexp(logV, dim=1).sum()
        v = torch.softmax(logV, dim=1)  # (batch, num_models)

        def zeros(*shape):
            return torch.zeros(*shape, dtype=dt, device=dev)

        dgm = zeros(num_models)
        dalpha_n = zeros(num_mix, self.n_comps)
        dmu_n = zeros(num_mix, self.n_comps)
        dmu_d = zeros(num_mix, self.n_comps)
        dbeta_n = zeros(num_mix, self.n_comps)
        dbeta_d = zeros(num_mix, self.n_comps)
        drho_n = zeros(num_mix, self.n_comps)
        dWtmp = zeros(self.n_channels, self.n_channels, num_models)
        dc = zeros(self.n_channels, num_models)
        if self.do_newton:
            dsigma2_numer = zeros(self.n_channels, num_models)
            dkappa_numer = zeros(num_mix, self.n_channels, num_models)
            dlambda_numer = zeros(num_mix, self.n_channels, num_models)
        tiny = torch.finfo(dt).tiny

        for h in range(num_models):
            idx = self.comp_list[:, h]
            b, zr, y = b_list[h], z_list[h], y_list[h]
            v_h = v[:, h]
            beta_h = self.beta[:, idx].T.unsqueeze(0)  # sbeta, (1, n_ch, num_mix)
            rho_h = self.rho[:, idx].T  # (n_ch, num_mix)
            fp = _score(y, rho_h.unsqueeze(0))  # score, NOT dpdf (:1467)
            u = v_h.unsqueeze(-1).unsqueeze(-1) * zr  # u = v*z (:1439)
            ufp = u * fp  # (:1485)

            dgm[h] = v_h.sum()
            dalpha_n.index_add_(1, idx, u.sum(0).T)  # sum(u) (:1524)
            dmu_n.index_add_(1, idx, ufp.sum(0).T)  # sum(ufp) (:1532)
            dmu_d.index_add_(
                1, idx, (beta_h.squeeze(0) * (ufp / y).sum(0)).T
            )  # (:1537)
            dbeta_n.index_add_(1, idx, u.sum(0).T)  # sum(u) (:1550)
            dbeta_d.index_add_(1, idx, (ufp * y).sum(0).T)  # sum(ufp*y) (:1556)

            # drho_numer = rho * sum(u*|y|^rho*ln|y|)  (:1560-1578). The leading
            # rho comes from ln(|y|^rho)=rho*ln|y| in the Fortran logab chain
            # (issue #24 Bug 1). Guard only the per-sample underflow (:1570) --
            # no per-component (rho!=1&rho!=2) mask (Bug 2): |y|^rho*ln|y| is 0 at
            # y=0, and clamping the log input makes the product collapse there.
            ay = y.abs()
            ayrho = ay.pow(rho_h.unsqueeze(0))  # |y|^rho
            logab = rho_h.unsqueeze(0) * torch.log(ay.clamp_min(tiny))  # rho*ln|y|
            logab = torch.where(ayrho < tiny, torch.zeros_like(logab), logab)
            drho_n.index_add_(1, idx, (u * (ayrho * logab)).sum(0).T)

            g = (beta_h * ufp).sum(-1)  # g_i = sum_j sbeta*ufp (:1493)
            dWtmp[:, :, h] = g.T @ b  # source-space sum g_t b_t^T (:1592)
            dc[:, h] = g.sum(0)

            if self.do_newton:
                # Newton curvature accumulators (Fortran amica17.f90:1419,
                # 1500-1514), in terms of the score fp (not dpdf).
                dsigma2_numer[:, h] = (v_h.unsqueeze(-1) * b.pow(2)).sum(0)  # (:1419)
                dkappa_numer[:, :, h] = (
                    (u * fp.pow(2)).sum(0) * beta_h.squeeze(0).pow(2)
                ).T  # (:1500)
                dlambda_numer[:, :, h] = (u * (fp * y - 1.0).pow(2)).sum(0).T  # (:1511)

        updates = {
            "dgm": dgm,
            "dalpha_n": dalpha_n,
            "dmu_n": dmu_n,
            "dmu_d": dmu_d,
            "dbeta_n": dbeta_n,
            "dbeta_d": dbeta_d,
            "drho_n": drho_n,
            "dWtmp": dWtmp,
            "dc": dc,
            "ll": block_ll,
        }
        if self.do_newton:
            updates["dsigma2_numer"] = dsigma2_numer
            updates["dkappa_numer"] = dkappa_numer
            updates["dlambda_numer"] = dlambda_numer
        return updates

    def _accumulate_blocks(self, X: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Sum sufficient statistics over all blocks of ``X``.

        Peak memory scales with ``block_size`` (each block's intermediates
        are freed once accumulated), not with ``X.shape[1]``.
        """
        n_samples = X.shape[1]
        acc: Optional[Dict[str, torch.Tensor]] = None
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
    # M-step parameter update
    # ------------------------------------------------------------------
    def _finalize_newton_stats(self, acc: Dict[str, torch.Tensor]):
        """Reduce the Newton block accumulators into ``(sigma2, lambda, kappa)``.

        Ports the Fortran finalization (amica17.f90:1762-1776). The Fortran
        ``baralpha``/``dkappa_denom``/``dlambda_denom`` responsibility masses
        all cancel algebraically against the per-mixture ``dalpha`` weighting,
        leaving simply (with ``dgm = sum_t v_h`` the raw model mass):

            sigma2[i,h] = dsigma2_numer[i,h] / dgm[h]
            kappa[i,h]  = sum_j dkappa_numer[j,i,h] / dgm[h]
            lambda[i,h] = sum_j (dlambda_numer[j,i,h]
                                 + dkappa_numer[j,i,h] * mu[j,comp(i,h)]^2) / dgm[h]

        Returns (sigma2, lambda_, kappa), each (n_channels, n_models).
        """
        dgm = acc["dgm"].unsqueeze(0)  # (1, n_models)
        sigma2 = acc["dsigma2_numer"] / dgm
        kappa = acc["dkappa_numer"].sum(dim=0) / dgm
        # mu at each source's component: mu[j, comp_list[i,h]] -> (n_mix, n_ch, n_models)
        mu_at = self.mu[:, self.comp_list]
        lambda_ = (acc["dlambda_numer"] + acc["dkappa_numer"] * mu_at.pow(2)).sum(
            dim=0
        ) / dgm
        return sigma2, lambda_, kappa

    def _newton_direction(self, dA_h, sigma2_h, lambda_h, kappa_h):
        """Per-model Newton direction ``H`` from the natural gradient ``dA_h``.

        Vectorized port of the per-source-pair 2x2 solve (amica17.f90:1817-1832,
        pyAMICA.py:802-813):

            H[i,i] = dA_h[i,i] / lambda[i]
            sk1 = sigma2[i]*kappa[k];  sk2 = sigma2[k]*kappa[i]   (i != k)
            H[i,k] = (sk1*dA_h[i,k] - dA_h[k,i]) / (sk1*sk2 - 1)  if sk1*sk2 > 1

        Returns ``(H, posdef)``. ``posdef`` is False if any off-diagonal pair
        fails ``sk1*sk2 > 1`` (the positive-definiteness guard); the caller
        then falls back to the natural gradient for this model.
        """
        n = self.n_channels
        sk1 = sigma2_h.unsqueeze(1) * kappa_h.unsqueeze(0)  # [i,k] = sigma2[i]*kappa[k]
        sk2 = sigma2_h.unsqueeze(0) * kappa_h.unsqueeze(1)  # [i,k] = sigma2[k]*kappa[i]
        prod = sk1 * sk2
        valid = prod > 1.0
        denom = torch.where(valid, prod - 1.0, torch.ones_like(prod))
        h_off = (sk1 * dA_h - dA_h.T) / denom
        H = torch.where(valid, h_off, torch.zeros_like(h_off))
        # Diagonal overrides (uses lambda, not the off-diagonal formula).
        diag = torch.diagonal(dA_h) / lambda_h
        H = H - torch.diag(torch.diagonal(H)) + torch.diag(diag)
        # Positive-definite iff every off-diagonal pair passed the guard.
        offdiag = ~torch.eye(n, dtype=torch.bool, device=dA_h.device)
        posdef = bool(valid[offdiag].all().item())
        return H, posdef

    def _update_parameters(self, acc: Dict[str, torch.Tensor], n_samples: int):
        """Apply the M-step parameter update, matching
        ``pyAMICA.AMICA._update_parameters`` (natural-gradient and Newton).

        ``n_samples`` is the number of samples that fed the accumulators (the
        good-sample count when ``do_reject`` is active), so ``gm`` and the
        reported log-likelihood are normalized by the effective sample count.

        The mixture parameters use exact-EM fixed-point updates (no ``lrate``);
        only the ``A``/``W`` step is scaled by ``lrate`` (Fortran amica17.f90:
        1890-2035). ``c`` is not updated: for mean-removed data Fortran's ``c``
        stays at machine zero (issue #24), so the update is a no-op.
        """
        self.gm = acc["dgm"] / n_samples

        self.alpha = acc["dalpha_n"] / acc["dalpha_n"].sum(dim=0, keepdim=True)

        # Exact-EM mixture location/scale (Fortran :1978/:1993). No lrate.
        self.mu = self.mu + acc["dmu_n"] / acc["dmu_d"]
        self.beta = torch.clamp(
            self.beta * torch.sqrt(acc["dbeta_n"] / acc["dbeta_d"]),
            self.invsigmin,
            self.invsigmax,
        )

        # GG shape update with the 1/psi(1+1/rho) digamma factor (Fortran
        # :2013-2014); the divisor is the per-component responsibility mass
        # dalpha_n (floored so a near-empty component cannot poison rho).
        if not torch.all(self.rho == 1.0) and not torch.all(self.rho == 2.0):
            drho = acc["drho_n"] / acc["dalpha_n"].clamp_min(1e-8)
            psi = torch.special.digamma(1.0 + 1.0 / self.rho)
            new_rho = self.rho + self.rholrate * (1.0 - (self.rho / psi) * drho)
            self.rho = torch.clamp(
                torch.nan_to_num(new_rho, nan=self.rho0), self.minrho, self.maxrho
            )

        # --- A / W update: natural gradient, optionally Newton-preconditioned.
        # A is stored as Fortran's A^T (the true unmixing is W^T = inv(A)^T), so
        # Fortran's A_fort -= lrate*A_fort @ dir becomes, transposed,
        # A -= lrate*dir^T @ A (LEFT-multiply by the TRANSPOSED direction). The
        # direction ``dir`` (natural gradient I - <g b^T>/dgm, or its Newton
        # precondition) is built in Fortran's untransposed convention. Getting
        # this wrong (right-multiply by the untransposed dir) is invisible at the
        # fixed point but sends the free-running fit downhill -- issue #24 root
        # cause (.context/issue-24/root_cause_Aupdate.py, machine-exact check).
        newton_active = self.do_newton and self.iteration >= self.newt_start
        if newton_active:
            sigma2, lambda_, kappa = self._finalize_newton_stats(acc)

        eye = torch.eye(self.n_channels, dtype=self.dtype, device=self.device)
        directions = []
        no_newt = False
        for h in range(self.n_models):
            dA_h = -acc["dWtmp"][:, :, h] / acc["dgm"][h] + eye  # I - <g b^T>/dgm
            if newton_active:
                H, posdef = self._newton_direction(
                    dA_h, sigma2[:, h], lambda_[:, h], kappa[:, h]
                )
                if posdef:
                    directions.append(H)
                else:
                    no_newt = True
                    directions.append(dA_h)  # fall back to natural gradient
            else:
                directions.append(dA_h)

        if newton_active and no_newt:
            # Fortran prints "Hessian not positive definite, using natural
            # gradient" here (amica17.f90:1911-1913). Surface the same signal so
            # a silent all-fallback run (the current issue #21 behaviour) is
            # visible without re-instrumenting the code.
            self.n_newton_fallbacks += 1
            logger.warning(
                "Newton not positive definite at iter %d; using natural gradient.",
                self.iteration,
            )

        # Learning-rate ramp: toward newtrate while Newton is active and stable,
        # otherwise toward lrate0 (Fortran amica17.f90:1906-1917). Ramped after
        # mu/beta/rho (which are exact-EM, lrate-free) and before A.
        if newton_active and not no_newt:
            self.lrate = min(
                self.newtrate, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
            )
        else:
            self.lrate = min(
                self.lrate_cap, self.lrate + min(1.0 / self.newt_ramp, self.lrate)
            )

        for h in range(self.n_models):
            idx = self.comp_list[:, h]
            A_cols = self.A[:, idx]
            self.A[:, idx] = A_cols - self.lrate * (directions[h].T @ A_cols)

        if self.doscaling and (self.iteration % self.scalestep == 0):
            scale = torch.sqrt((self.A**2).sum(dim=0))  # (n_comps,)
            nonzero = scale > 0
            self.A[:, nonzero] = self.A[:, nonzero] / scale[nonzero]
            self.mu[:, nonzero] = self.mu[:, nonzero] * scale[nonzero]
            self.beta[:, nonzero] = self.beta[:, nonzero] / scale[nonzero]

        self._update_unmixing_matrices()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit(
        self, X: np.ndarray, max_iter: int = 100, verbose: bool = True
    ) -> "AMICATorchNG":
        """Fit the model to data.

        Parameters
        ----------
        X : np.ndarray of shape (n_channels, n_samples)
            Input data.
        max_iter : int, default=100
            Number of natural-gradient EM iterations.
        verbose : bool, default=True
            Show a tqdm progress bar.

        Returns
        -------
        self : AMICATorchNG
        """
        if X.ndim != 2:
            raise ValueError(
                f"X must be a 2D array (n_channels, n_samples), got shape {X.shape}"
            )
        if X.shape[0] != self.n_channels:
            raise ValueError(
                f"X has {X.shape[0]} channels, model expects {self.n_channels}"
            )

        X_t = self._preprocess(X)
        n_total = X_t.shape[1]

        self._initialize_parameters()
        self.ll_history = []
        self.numrej = 0
        self.n_newton_fallbacks = 0
        self.stop_reason = "max_iter"
        self.good_idx = (
            torch.arange(n_total, device=self.device) if self.do_reject else None
        )
        numdecs = 0

        iterator = tqdm(range(max_iter), desc="AMICA-NG", disable=not verbose)
        for it in iterator:
            self.iteration = it

            X_use = X_t[:, self.good_idx] if self.do_reject else X_t
            n_use = X_use.shape[1]
            acc = self._accumulate_blocks(X_use)

            # Whether rejection fires this iteration (Fortran schedule,
            # amica17.f90:1141-1146). Fortran rejects using the per-sample
            # log-likelihood from THIS iteration's E-step, i.e. the PRE-update
            # parameters (loglik is stored in get_updates_and_likelihood before
            # update_params runs). Capture it here, before _update_parameters,
            # to match that ordering.
            will_reject = (
                self.do_reject
                and self.maxrej > 0
                and (
                    it == self.rejstart
                    or (
                        max(1, it - self.rejstart) % self.rejint == 0
                        and self.numrej < self.maxrej
                    )
                )
            )
            reject_ll = self._sample_ll(self.good_idx, X_t) if will_reject else None

            self._update_parameters(acc, n_use)

            ll = (acc["ll"] / (n_use * self.n_channels)).item()
            if math.isnan(ll):
                logger.warning("NaN log-likelihood at iteration %d; stopping.", it)
                self.stop_reason = "nan_ll"
                break
            self.ll_history.append(ll)

            # Learning-rate control, ported from Fortran (amica17.f90:1062-1108).
            # Natural-gradient/Newton ascent is not monotonic at a fixed rate:
            # when the log-likelihood decreases, anneal the working rates
            # (lrate, rholrate). If decreases persist for maxdecs iterations,
            # ratchet the *ceilings* down (lrate_cap, and newtrate once Newton
            # is running) so the per-iteration ramp can no longer re-inflate
            # lrate back to the overshooting value -- without this the ramp and
            # a one-shot halving just oscillate and the LL drifts down.
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
                    if self.do_newton and it > self.newt_start:
                        self.newtrate *= self.lratefact
                    numdecs = 0
            if self.do_newton and it == self.newt_start:
                numdecs = 0

            # Outlier rejection, after the parameter update (Fortran order,
            # amica17.f90:1141-1146) but using the pre-update per-sample LL
            # captured above.
            if will_reject:
                self._reject_outliers(reject_ll)

            iterator.set_postfix({"LL": f"{ll:.4f}", "lrate": f"{self.lrate:.4g}"})

        return self

    def _sample_ll(self, good_idx: torch.Tensor, X_t: torch.Tensor) -> torch.Tensor:
        """Per-sample total log-likelihood over ``good_idx``, block by block, in
        ``good_idx`` order (so a keep-mask over the result maps back correctly)."""
        parts = [
            self._block_sample_ll(X_t[:, good_idx[start : start + self.block_size]])
            for start in range(0, int(good_idx.numel()), self.block_size)
        ]
        return torch.cat(parts)

    def _reject_outliers(self, ll_vec: torch.Tensor):
        """Permanently drop samples whose (pre-update) log-likelihood is a low
        outlier.

        Fortran ``reject_data`` (amica17.f90:2380-2464): reject any currently-good
        sample with ``loglik < mean - rejsig*std`` (population std). The rejection
        is one-directional; ``good_idx`` only ever shrinks, and the good-sample
        count drives the ``gm``/LL normalization thereafter. ``ll_vec`` is the
        per-sample log-likelihood over the current good set, in ``good_idx`` order.
        """
        good = self.good_idx
        mean = ll_vec.mean()
        std = torch.sqrt((ll_vec.pow(2).mean() - mean.pow(2)).clamp_min(0.0))
        keep = ll_vec >= (mean - self.rejsig * std)

        if not bool(keep.any()):
            raise ValueError(
                f"Outlier rejection removed all {good.numel()} samples "
                f"(rejsig={self.rejsig} too aggressive for this data)."
            )

        self.good_idx = good[keep]
        self.numrej += 1
        n_rejected = int(good.numel() - self.good_idx.numel())
        logger.info(
            "Rejection %d at iter %d: dropped %d samples (%d good remaining).",
            self.numrej,
            self.iteration,
            n_rejected,
            int(self.good_idx.numel()),
        )

    def transform(self, X: np.ndarray, model_idx: int = 0) -> np.ndarray:
        """Apply the learned unmixing matrix to (new) data.

        The internal ``W = inv(A)`` is stored transposed relative to the true
        unmixing (the E-step forms activations as ``X^T @ W``), so the unmixing
        applied here is ``W^T`` (issue #24 transpose convention).
        """
        X_t = torch.from_numpy(np.ascontiguousarray(X)).to(self.device, self.dtype)
        X_t = self.sphere @ (X_t - self.mean)
        S = self.W[:, :, model_idx].T @ X_t - self.c[:, model_idx : model_idx + 1]
        return S.cpu().numpy()

    def get_mixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """True mixing matrix ``A_fort`` = (stored A)^T (issue #24 convention)."""
        return self.A[:, self.comp_list[:, model_idx]].T.cpu().numpy()

    def get_unmixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """True unmixing matrix ``W_fort`` = (stored W)^T (issue #24 convention)."""
        return self.W[:, :, model_idx].T.cpu().numpy()
