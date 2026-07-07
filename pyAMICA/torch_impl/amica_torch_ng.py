"""
Natural-gradient EM PyTorch backend for AMICA (ADR 0001).

Rather than reframing AMICA as "minimize negative log-likelihood with Adam
over reparameterized tensors" (the approach of the earlier Adam/autograd
backends, since removed in issue #32), this module is a direct, vectorized
port of the closed-form E-step/M-step fixed-point updates used by the Fortran
reference (``amica17.f90``) and the legacy NumPy implementation
(``pyAMICA.pyAMICA.AMICA._get_block_updates`` / ``_update_parameters``, which
is this module's line-by-line spec). There is
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

Source-density families (issue #26): the default GG path cites ``amica17.f90``,
but the reference *binary* is ``amica15mac`` = ``amica15.f90``, which (unlike the
GG-only ``amica17.f90``) implements the ``pdtype`` density families. The
``pdftype`` machinery therefore cites ``amica15.f90``; the two Fortran sources
are not interchangeable. See ``.context/decisions/0002-adaptive-pdf-families.md``.
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
_LOG4 = math.log(4.0)  # logistic-family normalizer (amica15.f90:1328)
_HALF_LOG_PI = 0.5 * math.log(math.pi)
# Log-normalizers for the non-GG density families, using Fortran's exact literal
# constants (amica15.f90:1315/1341/1353) so the log-density matches the reference
# binary bit-for-bit: 2.506628274 = sqrt(2*pi) (Gaussian, pdtype 2); 4.132731354 /
# 1.858073988 = the sub-/super-Gaussian cosh normalizers (pdtype 4 / 1).
_LOG_SQRT_2PI = math.log(2.506628274)
_LOG_NORM_COSH_SUB = math.log(4.132731354)
_LOG_NORM_COSH_SUP = math.log(1.858073988)
# Fortran's epsdble (amica17_header.f90:73): the drho-numerator underflow guard
# zeros the rho*ln|y| term when |y|^rho falls below this, matching amica17.f90:1570.
_EPSDBLE = 1e-16

# Best-iterate safeguard (issue #51). The lrate schedule is deliberately
# non-monotone: both NG and Fortran anneal the rate only *after* an LL decrease,
# so a late Newton fallback can overshoot and a run can end below a peak it
# already reached (on the sample EEG this is the sole driver of NG's inflated
# multi-model LL variance -- one seed peaked at -3.357 then crashed to -3.545 in
# its last iterations). fit() therefore tracks the highest-LL iterate and
# restores it when the final LL falls more than this tolerance below that peak.
# Units: mean log-likelihood per sample-channel (same scale as min_dll), so 1e-9
# reads as "numerical noise, not a real overshoot". The threshold also keeps a
# monotone single-model run (issue #24 parity) a bit-exact no-op: its final
# iterate already IS the best, the gap is 0 < tol, and no restore fires.
_KEEP_BEST_TOL = 1e-9


def _logcosh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log cosh(x) = |x| - log2 + log1p(exp(-2|x|))``."""
    ax = x.abs()
    return ax - _LOG2 + torch.log1p(torch.exp(-2.0 * ax))


def _log_pdf_and_deriv(
    y: torch.Tensor, rho: torch.Tensor, pdtype: Optional[torch.Tensor] = None
) -> tuple[torch.Tensor, torch.Tensor]:
    """Vectorized source-density log-density and density derivative.

    Elementwise port of ``pyAMICA.pyAMICA.AMICA._compute_log_pdf``: branches via
    ``torch.where`` instead of Python control flow so it runs over full
    ``(block, source, mixture)`` tensors with no source/mixture loop. ``y``,
    ``rho`` and ``pdtype`` must be broadcastable to a common shape.

    When ``pdtype is None`` (the default ``pdftype=0`` path) this computes only
    the generalized-Gaussian (GG) family, branching on ``rho``
    (Laplace/Gaussian/GG), and is bit-identical to the pre-#26 implementation.
    When ``pdtype`` is given it additionally selects, per source, among the
    fixed density families of ``amica15.f90`` (codes 0/2/3/4/1): GG, Gaussian,
    logistic, sub-Gaussian cosh+, super-Gaussian cosh-. The density derivative
    obeys ``dpdf = -fp * pdf`` for every family (``fp`` = the score from
    ``_score``), which reproduces the GG ``dpdf`` exactly.
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
    if pdtype is None:
        return log_pdf, dpdf

    # Non-GG families (amica15.f90:1309-1353). Each is `-cost - log_norm`, and
    # dpdf = -fp * exp(log_pdf).
    log_pdf_2 = -0.5 * y * y - _LOG_SQRT_2PI  # Gaussian
    log_pdf_3 = -2.0 * _logcosh(0.5 * y) - _LOG4  # logistic (sech^2)
    lc = _logcosh(y)
    log_pdf_4 = -0.5 * y * y + lc - _LOG_NORM_COSH_SUB  # sub-Gaussian cosh+
    log_pdf_1 = -0.5 * y * y - lc - _LOG_NORM_COSH_SUP  # super-Gaussian cosh-

    log_pdf = torch.where(
        pdtype == 2,
        log_pdf_2,
        torch.where(
            pdtype == 3,
            log_pdf_3,
            torch.where(
                pdtype == 4, log_pdf_4, torch.where(pdtype == 1, log_pdf_1, log_pdf)
            ),
        ),
    )
    fp = _score(y, rho, pdtype)
    dpdf = torch.where(pdtype == 0, dpdf, -fp * torch.exp(log_pdf))
    return log_pdf, dpdf


def _score(
    y: torch.Tensor, rho: torch.Tensor, pdtype: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Source-density score ``fp = -d(log pdf)/dy`` (Fortran ``fp``).

    For the GG family this is ``fp(y) = rho*sign(y)*|y|^(rho-1)`` (``sign(y)``
    for Laplace, ``2y`` for Gaussian), used by the exact-EM and Newton
    sufficient statistics (``amica15.f90:1449-1473``). It is distinct from the
    density derivative ``dpdf`` (which carries an extra ``pdf`` factor).

    With ``pdtype is None`` only the GG score is computed (bit-identical to the
    pre-#26 path). With ``pdtype`` given it selects per source among the fixed
    families: 2 Gaussian ``y``; 3 logistic ``tanh(y/2)``; 4 sub-Gaussian
    ``y - tanh(y)``; 1 super-Gaussian ``y + tanh(y)``.
    """
    abs_y = y.abs()
    sign_y = torch.sign(y)
    fp_lap = sign_y
    fp_gau = 2.0 * y
    fp_gg = rho * sign_y * abs_y.pow(rho - 1.0)
    is_lap = rho == 1.0
    is_gau = rho == 2.0
    fp = torch.where(is_gau, fp_gau, torch.where(is_lap, fp_lap, fp_gg))
    if pdtype is None:
        return fp

    tanh_half = torch.tanh(0.5 * y)
    tanh_y = torch.tanh(y)
    return torch.where(
        pdtype == 2,
        y,
        torch.where(
            pdtype == 3,
            tanh_half,
            torch.where(
                pdtype == 4, y - tanh_y, torch.where(pdtype == 1, y + tanh_y, fp)
            ),
        ),
    )


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
    keep_best : bool, default=True
        Return the highest-log-likelihood iterate instead of the last one
        (issue #51). The lrate schedule is non-monotone (it anneals only after
        an LL *decrease*), so a late Newton-fallback overshoot can leave the
        final iterate below a peak the run already reached. When the final LL
        falls more than a small tolerance below that peak, ``fit`` restores the
        peak's parameters. A monotone single-model run (issue #24 parity) is a
        bit-exact no-op. Automatically inactive under ``do_reject`` (the
        good-sample set, and thus the LL normalization, changes across
        iterations, making per-iteration LLs incomparable).
    pdftype : int, default=0
        Source-density family (issue #26), matching Fortran ``amica15.f90``'s
        ``pdtype`` codes: 0 generalized Gaussian (default; rho adapts), 2
        Gaussian, 3 logistic, 4 sub-Gaussian cosh+. ``pdftype=1`` enables the
        extended-Infomax adaptive switcher, which flips each source between the
        super-Gaussian (code 1) and sub-Gaussian (code 4) cosh densities by
        kurtosis sign. For every non-GG family the GG shape update is frozen
        (Fortran ``dorho=.false.``); the single-component families 1/4 (and the
        adaptive mode) require ``n_mix=1``. ``pdftype=0`` is byte-for-byte the
        pre-#26 implementation.
    kurt_start, num_kurt, kurt_int : int
        Adaptive-switch schedule (only used when ``pdftype=1``): first iteration
        to re-estimate kurtosis, number of switch passes, and the iteration
        interval between them. ``num_kurt=0`` disables switching (the family
        stays at its super-Gaussian init).
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
        keep_best: bool = True,
        pdftype: int = 0,
        kurt_start: int = 3,
        num_kurt: int = 5,
        kurt_int: int = 1,
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

        # Best-iterate safeguard (issue #51). When True, fit() restores the
        # highest-log-likelihood iterate if the run ends more than _KEEP_BEST_TOL
        # below it (a late Newton-fallback overshoot). Disabled automatically
        # under do_reject, where the good-sample set (and the LL normalization)
        # changes across iterations, so per-iteration LLs are not comparable.
        self.keep_best = keep_best

        # Source-density family selection (Fortran ``pdftype``, amica15.f90). Values
        # match Fortran's per-source ``pdtype`` codes: 0 generalized Gaussian (the
        # default, GG-mixture with adaptive rho), 2 Gaussian mixture, 3 logistic
        # (sech^2) mixture, 4 sub-Gaussian cosh+ (single component). pdftype=1 enables
        # the extended-Infomax adaptive switcher (Fortran's do_choose_pdfs trigger),
        # which flips each source between the super-Gaussian (code 1) and sub-Gaussian
        # (code 4) cosh densities by kurtosis sign on the kurt_start/num_kurt/kurt_int
        # schedule. Families 1 and 4 are single-component (no alpha mixture).
        if pdftype not in (0, 1, 2, 3, 4):
            raise ValueError(f"pdftype must be one of 0,1,2,3,4; got {pdftype}")
        self.pdftype = pdftype
        # Fortran freezes the GG shape update for every non-GG family (amica15.f90:
        # `if (pdftype /= 0) dorho = .false.`, line 3682).
        self.dorho = pdftype == 0
        # pdftype==1 is Fortran's adaptive trigger (amica15.f90:594).
        self.do_choose_pdfs = pdftype == 1
        self.kurt_start = kurt_start
        self.num_kurt = num_kurt
        self.kurt_int = kurt_int
        # Families 1/4 (and the adaptive mode, which uses only codes 1 and 4) are
        # single-component densities: Fortran's z0 references only mixture component
        # j=1 and omits log(alpha). They are meaningful only with n_mix == 1.
        if pdftype in (1, 4) and n_mix != 1:
            raise ValueError(
                f"pdftype={pdftype} is a single-component density (adaptive mode "
                f"uses codes 1 and 4); it requires n_mix=1, got n_mix={n_mix}."
            )
        # Validate the adaptive-switch schedule up front (mirrors the do_reject
        # checks below): kurt_int==0 would otherwise raise a bare ZeroDivisionError
        # deep in fit(), and a negative kurt_int silently changes the schedule.
        if self.do_choose_pdfs:
            if kurt_int < 1:
                raise ValueError(f"kurt_int must be >= 1, got {kurt_int}")
            if kurt_start < 1:
                raise ValueError(f"kurt_start must be >= 1, got {kurt_start}")
            if num_kurt < 0:
                raise ValueError(f"num_kurt must be >= 0, got {num_kurt}")

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
        # Log-likelihood of the *returned* parameters (issue #51). With
        # keep_best, ``ll_history`` stays the true per-iteration trajectory
        # (which can include a late overshoot), while ``final_ll_`` is the LL of
        # the iterate fit() actually kept -- use this, not ``ll_history[-1]``, as
        # the model's fitted log-likelihood. Set by fit().
        self.final_ll_: Optional[float] = None

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
        # Per-source density-family codes (n_channels, n_models); set in
        # _initialize_parameters and mutated by the adaptive switcher.
        self.pdtype = None
        # Number of adaptive-switch passes already performed (Fortran numchpdf).
        self.n_kurt_done = 0
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

        # Per-source density-family codes, Fortran ``pdtype = pdftype`` (amica15.f90:
        # 593). In adaptive mode (pdftype==1) every source starts as the
        # super-Gaussian code 1 and the switcher may flip it to 4.
        self.pdtype = torch.full(
            (n, m), self.pdftype, dtype=torch.long, device=self.device
        )
        self.n_kurt_done = 0

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

    def _pdtype_h(self, h: int) -> Optional[torch.Tensor]:
        """Per-source density-family codes for model ``h``, shaped for
        broadcasting against ``(batch, n_channels, num_mix)`` tensors, or
        ``None`` on the default ``pdftype=0`` (GG-only) fast path so the E-step
        stays bit-identical to the pre-#26 implementation.
        """
        if self.pdftype == 0:
            return None
        return self.pdtype[:, h].view(1, -1, 1)

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
            # Activation b = W(x - c): c is the per-model data-space center.
            # Fortran subtracts wc in the E-step (amica17.f90:1280-1292), where
            # wc = W@c is precomputed in get_unmixing_matrices (amica17.f90:2178).
            # Subtracting c in data space before W is equivalent and keeps c's
            # semantics identical to Fortran's. For n_models=1, c == 0, so this is
            # bit-identical to the old X.T @ W.
            b = (X - self.c[:, h].unsqueeze(1)).T @ self.W[:, :, h]  # (batch, n_ch)

            mu_h = self.mu[:, idx].T.unsqueeze(0)  # (1, n_channels, num_mix)
            beta_h = self.beta[:, idx].T.unsqueeze(0)
            rho_h = self.rho[:, idx].T.unsqueeze(0)
            alpha_h = self.alpha[:, idx].T.unsqueeze(0)

            y = beta_h * (b.unsqueeze(-1) - mu_h)  # (batch, n_channels, num_mix)
            log_pdf, dpdf = _log_pdf_and_deriv(y, rho_h, self._pdtype_h(h))

            # z0 = log(alpha) + log(beta) + log_pdf. For the single-component
            # families (codes 1/4) n_mix==1 so alpha==1 and log(alpha)==0, which
            # reproduces Fortran's alpha-free z0 (amica15.f90:1340/1352).
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
            (n_channels, n_channels, n_models), ``dc_numer`` (n_channels,
            n_models; the data-space bias numerator ``sum_t v_h*x``, issue #27),
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
        dc_numer = zeros(self.n_channels, num_models)
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
            # score fp; the family select-case is amica15.f90:1449-1473 (amica17
            # is GG-only, so cite the binary's source explicitly here).
            fp = _score(y, rho_h.unsqueeze(0), self._pdtype_h(h))
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
            logab = torch.where(ayrho < _EPSDBLE, torch.zeros_like(logab), logab)
            drho_n.index_add_(1, idx, (u * (ayrho * logab)).sum(0).T)

            g = (beta_h * ufp).sum(-1)  # g_i = sum_j sbeta*ufp (:1493)
            dWtmp[:, :, h] = g.T @ b  # source-space sum g_t b_t^T (:1592)
            # Data-space bias accumulator: dc_numer[i,h] = sum_t v_h(t)*x(i,t)
            # (Fortran :1423-1429). The denominator is dgm[h] = sum_t v_h(t).
            # NOTE: this replaces the old gradient-style bias g.sum(0), which was
            # accumulated but never applied (c was frozen at 0); the Fortran
            # update is the data-space responsibility-weighted mean (issue #27).
            dc_numer[:, h] = X @ v_h

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
            "dc_numer": dc_numer,
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
        1890-2035). The per-model data-space bias ``c`` uses Fortran's exact-EM
        ``update_c`` (amica17.f90:1423-1429/1899-1901): ``c[i,h] = sum_t v_h*x /
        sum_t v_h``, the responsibility-weighted data mean for model ``h``. For
        ``n_models=1`` on mean-removed data ``v == 1`` so ``c`` collapses to the
        (zero) data mean; the update is skipped there so single-model parity stays
        bit-exact (issue #24). For ``n_models>1`` the per-model ``v`` is
        non-uniform and ``c`` moves each iteration (issue #27).
        """
        self.gm = acc["dgm"] / n_samples

        # Per-model data-space bias (Fortran's `update_c` flag, amica17.f90:1423-
        # 1429 numerator / :1899-1901 division). Skipped for a single model to keep
        # the issue #24 parity bit-exact: with v==1 the update would add a ~1e-13
        # float-sum residual of the (mean-removed) data, perturbing the
        # otherwise-exact single-model trajectory. dgm[h] = sum_t v_h(t) is the
        # denominator (Fortran `dc_denom`). A fully-dead model (dgm[h]==0 => v_h==0
        # for all t, so dc_numer[:,h]==0 too) gives 0/0; keep its PRIOR c rather
        # than write a NaN. A NaN c would poison the NEXT iteration's cross-model
        # softmax for EVERY model (unlike log(gm[h])=-inf, which softmax tolerates,
        # so a dead model was previously inert) -- this containment mirrors the
        # mu/beta/rho non-finite guards below. `dgm>0` is also False for a NaN dgm
        # from upstream corruption, so that is contained too.
        if self.n_models > 1:
            dgm = acc["dgm"]
            live = dgm > 0.0
            new_c = acc["dc_numer"] / dgm.clamp_min(torch.finfo(self.dtype).tiny)
            self.c = torch.where(live.unsqueeze(0), new_c, self.c)
            if not bool(live.all()):
                logger.warning(
                    "Zero-responsibility model(s) at iter %d; kept their prior "
                    "bias c (dead-model guard).",
                    self.iteration,
                )

        self.alpha = acc["dalpha_n"] / acc["dalpha_n"].sum(dim=0, keepdim=True)

        # Finalize the Newton curvature with the PRE-update mu. Fortran folds the
        # mu^2 term into lambda during E-step accumulation, before the M-step
        # moves mu (amica17.f90:1762-1774), and the NumPy port bakes it in at
        # accumulation time. Do it here, before self.mu is reassigned below, so
        # lambda uses this iteration's mu rather than the updated one.
        newton_active = self.do_newton and self.iteration >= self.newt_start
        if newton_active:
            sigma2, lambda_, kappa = self._finalize_newton_stats(acc)

        # Exact-EM mixture location/scale (Fortran :1978/:1993). No lrate.
        self.mu = self.mu + acc["dmu_n"] / acc["dmu_d"]
        self.beta = torch.clamp(
            self.beta * torch.sqrt(acc["dbeta_n"] / acc["dbeta_d"]),
            self.invsigmin,
            self.invsigmax,
        )
        # Fortran keeps a live "NaN in sbeta!" canary here (amica17.f90:1996-2000).
        # The exact-EM mu/beta divisions are unguarded (matching Fortran, whose own
        # mu/beta guard is commented out), so surface a non-finite value here
        # instead of letting it propagate to a later, unattributable nan-LL stop.
        if not torch.isfinite(self.mu).all() or not torch.isfinite(self.beta).all():
            logger.warning(
                "Non-finite mu/beta at iter %d (a mixture component's mass likely "
                "collapsed).",
                self.iteration,
            )

        # GG shape update with the 1/psi(1+1/rho) digamma factor (Fortran
        # :2013-2014); the divisor is the per-component responsibility mass
        # dalpha_n (floored so a near-empty component cannot poison rho). A NaN
        # here (e.g. from upstream mu/beta corruption) is reset to rho0 -- but
        # logged first, so the reset does not silently erase the failure origin.
        # Skipped for every non-GG family: Fortran sets dorho=.false. when
        # pdftype/=0 (amica15.f90:3682), freezing rho at rho0.
        if (
            self.dorho
            and not torch.all(self.rho == 1.0)
            and not torch.all(self.rho == 2.0)
        ):
            drho = acc["drho_n"] / acc["dalpha_n"].clamp_min(1e-8)
            psi = torch.special.digamma(1.0 + 1.0 / self.rho)
            new_rho = self.rho + self.rholrate * (1.0 - (self.rho / psi) * drho)
            nan_mask = torch.isnan(new_rho)
            if nan_mask.any():
                logger.warning(
                    "NaN in rho update at iter %d for %d component(s); resetting "
                    "to rho0=%g.",
                    self.iteration,
                    int(nan_mask.sum()),
                    self.rho0,
                )
                new_rho = torch.where(
                    nan_mask, torch.full_like(new_rho, self.rho0), new_rho
                )
            self.rho = torch.clamp(new_rho, self.minrho, self.maxrho)

        # --- A / W update: natural gradient, optionally Newton-preconditioned.
        # A is stored as Fortran's A^T (the true unmixing is W^T = inv(A)^T), so
        # Fortran's A_fort -= lrate*A_fort @ dir becomes, transposed,
        # A -= lrate*dir^T @ A (LEFT-multiply by the TRANSPOSED direction). The
        # direction ``dir`` (natural gradient I - <g b^T>/dgm, or its Newton
        # precondition) is built in Fortran's untransposed convention. Getting
        # this wrong (right-multiply by the untransposed dir) is invisible at the
        # fixed point but sends the free-running fit downhill -- issue #24 root
        # cause (.context/issue-24/root_cause_Aupdate.py, machine-exact check).
        # (newton_active / sigma2 / lambda_ / kappa were finalized above.)
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

    def _choose_pdfs(self, X: torch.Tensor) -> None:
        """Extended-Infomax adaptive PDF switch (Fortran ``do_choose_pdfs``).

        Re-estimates each source's kurtosis from the current model activations
        and sets its density family to the super-Gaussian (code 1) or
        sub-Gaussian (code 4) cosh density by kurtosis sign. This is the
        extended-Infomax rule that ``runamica15.m`` documents for the
        ``kurt_start``/``num_kurt``/``kurt_int`` schedule (the super/sub-Gaussian
        scores ``y +/- tanh(y)`` are exactly the two families 1/4). The
        reference binary declares this (``pdftype==1`` sets ``do_choose_pdfs``,
        amica15.f90:594) but never runs the switch (``m2sum``/``m4sum`` are
        never accumulated), so there is no bit-exact oracle; validated by
        real-data log-likelihood (must not decrease vs the fixed GG default).
        """
        n_ch, n_models = self.n_channels, self.n_models
        m2 = torch.zeros(n_ch, n_models, dtype=self.dtype, device=self.device)
        m4 = torch.zeros_like(m2)
        nsub = torch.zeros(n_models, dtype=self.dtype, device=self.device)
        n_samples = X.shape[1]
        for start in range(0, n_samples, self.block_size):
            block = X[:, start : start + self.block_size]
            logV, b_list, *_ = self._forward(block)
            v = torch.softmax(logV, dim=1)  # (batch, n_models)
            for h in range(n_models):
                b = b_list[h]  # (batch, n_ch)
                vh = v[:, h].unsqueeze(1)
                m2[:, h] += (vh * b.pow(2)).sum(0)
                m4[:, h] += (vh * b.pow(4)).sum(0)
                nsub[h] += v[:, h].sum()

        # Kurtosis = E[b^4]/E[b^2]^2 - 3 = nsub * m4 / m2^2 - 3, per (source, model).
        tiny = torch.finfo(self.dtype).tiny
        kurt = nsub.unsqueeze(0) * m4 / m2.pow(2).clamp_min(tiny) - 3.0
        self.pdtype = self._pdtype_from_kurtosis(kurt, nsub)

    def _pdtype_from_kurtosis(
        self, kurt: torch.Tensor, nsub: torch.Tensor
    ) -> torch.Tensor:
        """Map per-source excess kurtosis to a density-family code (pure).

        Super-Gaussian (positive kurtosis) -> code 1; sub-Gaussian -> code 4.
        Only sources with a meaningful signal switch: a dead model
        (``nsub[h]==0`` => ``kurt==-3.0``, finite) or a numerically blown-up
        source (``kurt`` NaN, and ``NaN>0`` is False) would otherwise be silently
        assigned code 4 with no diagnostic, so those keep their prior ``pdtype``
        and are logged -- mirroring the dead-model / non-finite guards in
        ``_update_parameters``. Split out from ``_choose_pdfs`` so the decision
        (including the sub-Gaussian branch, which real EEG rarely triggers) is
        unit-testable on a constructed ``kurt`` tensor.
        """
        ones = torch.ones_like(self.pdtype)
        new_pdtype = torch.where(kurt > 0.0, ones, ones * 4)
        valid = torch.isfinite(kurt) & (nsub.unsqueeze(0) > 0.0)
        result = torch.where(valid, new_pdtype, self.pdtype)
        if not bool(valid.all()):
            logger.warning(
                "Non-finite or zero-mass kurtosis for %d source/model pair(s) at "
                "iter %d; kept their prior pdtype (adaptive-switch guard).",
                int((~valid).sum()),
                self.iteration,
            )
        return result

    def _snapshot_params(self) -> Dict[str, torch.Tensor]:
        """Clone the fitted parameter tensors for the best-iterate safeguard
        (issue #51). Clones (not aliases) so the live in-place M-step updates do
        not roll the snapshot forward. Covers exactly ``_PARAM_TENSORS``; the
        constant preprocessing tensors (``mean``/``sphere``) are included so a
        restore is a total, unambiguous rollback of model state."""
        return {name: getattr(self, name).clone() for name in self._PARAM_TENSORS}

    def _restore_params(self, snapshot: Dict[str, torch.Tensor]) -> None:
        """Restore parameter tensors captured by :meth:`_snapshot_params`."""
        for name, tensor in snapshot.items():
            setattr(self, name, tensor)

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

        # Best-iterate safeguard (issue #51): track the highest-LL iterate so a
        # late Newton-fallback overshoot cannot leave the returned model below a
        # peak it already reached. Inactive under do_reject, where the good set
        # (and the LL normalization) changes across iterations, so per-iteration
        # LLs are not comparable.
        track_best = self.keep_best and not self.do_reject
        best_ll = -math.inf
        best_snapshot: Optional[Dict[str, torch.Tensor]] = None

        iterator = tqdm(range(max_iter), desc="AMICA-NG", disable=not verbose)
        for it in iterator:
            self.iteration = it

            X_use = X_t[:, self.good_idx] if self.do_reject else X_t
            n_use = X_use.shape[1]
            acc = self._accumulate_blocks(X_use)

            # Log-likelihood of the CURRENT (pre-update) parameters: acc["ll"] is
            # this iteration's E-step total, computed before _update_parameters
            # moves the parameters. A singular W makes logdet -> -inf (not NaN),
            # so guard on isfinite, not isnan alone: a -inf LL would otherwise
            # sail past as a mere "decrease" and the run would "complete"
            # (stop_reason=max_iter) on a degenerate model. Checking here, before
            # the update, stops on the last finite parameters instead of
            # overwriting them with a garbage update first.
            ll = (acc["ll"] / (n_use * self.n_channels)).item()
            if not math.isfinite(ll):
                self.stop_reason = "nan_ll" if math.isnan(ll) else "singular_ll"
                logger.warning(
                    "Non-finite log-likelihood (%s) at iteration %d; stopping.",
                    ll,
                    it,
                )
                break

            # Best-iterate safeguard (issue #51): remember the parameters that
            # produced this LL when it is the best seen, so a later overshoot
            # does not leave the returned model below this peak.
            if track_best and ll > best_ll:
                best_ll = ll
                best_snapshot = self._snapshot_params()

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

            # Extended-Infomax adaptive PDF switch (Fortran do_choose_pdfs). Runs
            # on the kurt_start/num_kurt/kurt_int schedule using the just-updated
            # W; the new per-source families take effect from the next E-step.
            # itf is the Fortran-style 1-indexed iteration. num_kurt=0 disables
            # switching (the family stays at its pdftype=1 super-Gaussian init).
            if self.do_choose_pdfs and self.n_kurt_done < self.num_kurt:
                itf = it + 1
                if (
                    itf >= self.kurt_start
                    and (itf - self.kurt_start) % self.kurt_int == 0
                ):
                    self._choose_pdfs(X_use)
                    self.n_kurt_done += 1

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

        # Log-likelihood of the parameters fit() returns. Defaults to the last
        # trajectory value; overwritten with the best iterate's LL below if the
        # safeguard restores it.
        self.final_ll_ = self.ll_history[-1] if self.ll_history else float("nan")

        # Restore the best iterate if the run ended materially below it (issue
        # #51). Skipped for a degenerate stop (params are already non-finite and
        # state_dict()/the wrapper reject them, so there is nothing good to keep)
        # and when the final LL is within _KEEP_BEST_TOL of the best -- a monotone
        # single-model run has final == best, so no restore fires and issue #24
        # parity stays bit-exact.
        if (
            track_best
            and best_snapshot is not None
            and self.stop_reason not in self._DEGENERATE_STOP_REASONS
            and self.ll_history
            and best_ll - self.ll_history[-1] > _KEEP_BEST_TOL
        ):
            logger.info(
                "Restoring best iterate (LL %.6f) over final LL %.6f "
                "(issue #51 best-iterate safeguard).",
                best_ll,
                self.ll_history[-1],
            )
            self._restore_params(best_snapshot)
            self.final_ll_ = best_ll

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
        unmixing (the E-step forms activations as ``(X-c)^T @ W``, see
        ``_forward``), so the unmixing applied here is ``W^T`` (issue #24
        transpose convention) with the per-model data-space center ``c``
        subtracted first (issue #27).
        """
        X_t = torch.from_numpy(np.ascontiguousarray(X)).to(self.device, self.dtype)
        X_t = self.sphere @ (X_t - self.mean)
        # c is the per-model data-space center: unmix as W(x - c) (issue #27).
        S = self.W[:, :, model_idx].T @ (X_t - self.c[:, model_idx : model_idx + 1])
        return S.cpu().numpy()

    def get_mixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """True mixing matrix ``A_fort`` = (stored A)^T (issue #24 convention)."""
        return self.A[:, self.comp_list[:, model_idx]].T.cpu().numpy()

    def get_unmixing_matrix(self, model_idx: int = 0) -> np.ndarray:
        """True unmixing matrix ``W_fort`` = (stored W)^T (issue #24 convention)."""
        return self.W[:, :, model_idx].T.cpu().numpy()

    # ------------------------------------------------------------------
    # Persistence (issue #36)
    # ------------------------------------------------------------------
    # Full fitted-parameter snapshot. A/W/c/comp_list/mean/sphere are what
    # transform()/get_*matrix() read back; mu/alpha/beta/rho/gm are the
    # mixture-PDF EM state, included for a complete snapshot (and for parity/
    # continued-analysis) even though no public method currently reads them.
    # pdtype is the per-source density-family code (issue #26): a non-default
    # pdftype model, or the adaptive switcher's chosen 1/4 assignments, would
    # otherwise silently revert to GG on reload. comp_list and pdtype are integer
    # tensors (dtype preserved on load); the rest follow self.dtype.
    _PARAM_TENSORS = (
        "A", "W", "c", "mu", "alpha", "beta", "rho", "gm",
        "comp_list", "mean", "sphere", "pdtype",
    )  # fmt: skip
    # Integer tensors in _PARAM_TENSORS: keep their dtype on load, only move device.
    _INT_PARAM_TENSORS = ("comp_list", "pdtype")

    # Stop reasons that mark a fit as degenerate (non-finite log-likelihood).
    # Such a model yields NaN sources, so state_dict() refuses to persist it
    # rather than let it round-trip silently (silent-failure review, PR #44).
    _DEGENERATE_STOP_REASONS = ("nan_ll", "singular_ll")

    def state_dict(self) -> dict:
        """Serialize the fitted model to a plain, device-agnostic dict.

        The returned dict has three parts: ``config`` (the constructor
        arguments needed to rebuild the object), ``params`` (the fitted
        tensors, moved to CPU), and ``extra`` (scalar/schedule state, plus the
        optional ``good_idx`` index tensor). Every value is a tensor or a plain
        Python primitive, so the dict round-trips through
        ``torch.save``/``torch.load`` with ``weights_only=True`` (no custom
        classes or ``torch.dtype`` objects: dtype is stored by name). Rebuild
        with :meth:`from_state_dict`.

        Raises if the model is unfitted or degenerate (a fit that ended on a
        non-finite log-likelihood): a NaN model must not be persisted silently.
        """
        if self.A is None:
            raise RuntimeError(
                "AMICATorchNG.state_dict() requires a fitted model; call fit() first."
            )
        if self.stop_reason in self._DEGENERATE_STOP_REASONS:
            raise RuntimeError(
                f"Refusing to serialize a degenerate model (stop_reason="
                f"{self.stop_reason!r}): fit() hit a non-finite log-likelihood at "
                f"iteration {self.iteration}. Fix the instability (lower lrate, "
                f"disable Newton, or check data conditioning) before saving."
            )
        # Defense-in-depth: catch a non-finite parameter even if stop_reason
        # bookkeeping ever misses it (the codebase has known NaN-suppression
        # risks). isfinite on the integer comp_list is trivially all-True.
        nonfinite = [
            name
            for name in self._PARAM_TENSORS
            if not torch.isfinite(getattr(self, name)).all()
        ]
        if nonfinite:
            raise RuntimeError(
                f"Refusing to serialize a model with non-finite parameters "
                f"{nonfinite} (stop_reason={self.stop_reason!r})."
            )
        config = {
            "n_channels": self.n_channels,
            "n_models": self.n_models,
            "n_mix": self.n_mix,
            "block_size": self.block_size,
            # lrate/newtrate/rholrate are annealed during fit; persist the
            # original constructor values (lrate0/newtrate0/rholrate0) and
            # restore the mutated ones from ``extra`` below.
            "lrate": self.lrate0,
            "minlrate": self.minlrate,
            "lratefact": self.lratefact,
            "maxdecs": self.maxdecs,
            "newt_ramp": self.newt_ramp,
            "do_newton": self.do_newton,
            "newt_start": self.newt_start,
            "newtrate": self.newtrate0,
            "do_reject": self.do_reject,
            "rejsig": self.rejsig,
            "rejstart": self.rejstart,
            "rejint": self.rejint,
            "maxrej": self.maxrej,
            "rho0": self.rho0,
            "minrho": self.minrho,
            "maxrho": self.maxrho,
            "rholrate": self.rholrate0,
            "rholratefact": self.rholratefact,
            # Best-iterate safeguard flag (issue #51); only affects a re-fit, but
            # persisted so a reloaded model reconstructs its exact configuration.
            "keep_best": self.keep_best,
            # Density-family selection (issue #26): needed so a reloaded model
            # rebuilds with the right pdftype/dorho/do_choose_pdfs and switch
            # schedule instead of the GG default.
            "pdftype": self.pdftype,
            "kurt_start": self.kurt_start,
            "num_kurt": self.num_kurt,
            "kurt_int": self.kurt_int,
            "invsigmin": self.invsigmin,
            "invsigmax": self.invsigmax,
            "doscaling": self.doscaling,
            "scalestep": self.scalestep,
            "do_mean": self.do_mean,
            "do_sphere": self.do_sphere,
            "do_approx_sphere": self.do_approx_sphere,
            "pcakeep": self.pcakeep,
            "pcadb": self.pcadb,
            "seed": self.seed,
            # Store dtype by name (e.g. "float64") to keep the payload
            # weights_only-safe; rebuilt via getattr(torch, ...) on load.
            "dtype": str(self.dtype).split(".")[-1],
        }
        # .clone() forces an independent copy even when self.device is already
        # CPU (where .cpu() would alias): fit() mutates A/mu/beta in place each
        # iteration, so an aliased snapshot would silently roll forward if
        # state_dict() were ever called mid-fit (e.g. best-so-far checkpointing).
        params = {
            name: getattr(self, name).detach().cpu().clone()
            for name in self._PARAM_TENSORS
        }
        extra = {
            "sldet": float(self.sldet),
            "iteration": int(self.iteration),
            "ll_history": [float(v) for v in self.ll_history],
            "final_ll": None if self.final_ll_ is None else float(self.final_ll_),
            "stop_reason": self.stop_reason,
            "n_newton_fallbacks": int(self.n_newton_fallbacks),
            "n_kurt_done": int(self.n_kurt_done),
            "numrej": int(self.numrej),
            "good_idx": None
            if self.good_idx is None
            else self.good_idx.detach().cpu().clone(),
            "lrate": float(self.lrate),
            "lrate_cap": float(self.lrate_cap),
            "newtrate": float(self.newtrate),
            "rholrate": float(self.rholrate),
        }
        return {
            "format_version": 3,
            "config": config,
            "params": params,
            "extra": extra,
        }

    @classmethod
    def from_state_dict(
        cls, state: dict, device: Optional[Union[str, torch.device]] = None
    ) -> "AMICATorchNG":
        """Rebuild a fitted :class:`AMICATorchNG` from :meth:`state_dict` output.

        ``device`` overrides where the restored tensors live (the constructor
        picks a default when ``None``); ``dtype`` always comes from the saved
        ``config``.
        """
        version = state.get("format_version")
        if version != 3:
            raise ValueError(
                f"unsupported AMICATorchNG state format_version: {version!r} "
                "(expected 3)"
            )
        for section in ("config", "params", "extra"):
            if section not in state:
                raise ValueError(
                    f"malformed AMICATorchNG state: missing {section!r} section "
                    f"(format_version={version}); the payload may be truncated."
                )
        config = dict(state["config"])
        config["dtype"] = getattr(torch, config["dtype"])
        obj = cls(device=device, **config)
        obj._load_params(state)
        return obj

    def _load_params(self, state: dict) -> None:
        """Restore fitted tensors/scalars from :meth:`state_dict` output onto
        this instance's device/dtype."""
        params = state["params"]
        missing = [name for name in self._PARAM_TENSORS if name not in params]
        if missing:
            raise ValueError(f"malformed AMICATorchNG state: missing params {missing}")
        # Guard against config/params drift: A and comp_list must match the
        # dimensions the constructor just derived, or transform()/the E-step
        # would fail later with a confusing matmul error far from load().
        if tuple(params["A"].shape) != (self.n_channels, self.n_comps):
            raise ValueError(
                f"restored A has shape {tuple(params['A'].shape)}, expected "
                f"{(self.n_channels, self.n_comps)} for n_channels="
                f"{self.n_channels}, n_models={self.n_models}"
            )
        if tuple(params["comp_list"].shape) != (self.n_channels, self.n_models):
            raise ValueError(
                f"restored comp_list has shape {tuple(params['comp_list'].shape)}, "
                f"expected {(self.n_channels, self.n_models)}"
            )
        for name in self._PARAM_TENSORS:
            tensor = params[name]
            # comp_list/pdtype hold integer indices/codes; preserve their dtype
            # and only move devices. The float parameters follow self.dtype.
            if name in self._INT_PARAM_TENSORS:
                setattr(self, name, tensor.to(self.device))
            else:
                setattr(self, name, tensor.to(self.device, self.dtype))

        extra = state["extra"]
        self.sldet = extra["sldet"]
        self.iteration = extra["iteration"]
        self.ll_history = list(extra["ll_history"])
        self.final_ll_ = extra["final_ll"]
        self.stop_reason = extra["stop_reason"]
        self.n_newton_fallbacks = extra["n_newton_fallbacks"]
        self.n_kurt_done = extra["n_kurt_done"]
        self.numrej = extra["numrej"]
        good_idx = extra["good_idx"]
        self.good_idx = None if good_idx is None else good_idx.to(self.device)
        self.lrate = extra["lrate"]
        self.lrate_cap = extra["lrate_cap"]
        self.newtrate = extra["newtrate"]
        self.rholrate = extra["rholrate"]
