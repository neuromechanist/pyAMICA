"""``AMICANative``: run the native AMICA Fortran binary as a pyAMICA backend.

The fourth run engine, alongside the NumPy, PyTorch (``AMICATorchNG``) and MLX
backends. It writes the data and an ``input.param``, runs the dependency-free
binary resolved by :mod:`pyAMICA.native.resolver`, and reads the result back
through :func:`pyAMICA.numpy_impl.load.loadmodout` -- so its output is an
``AmicaOutput`` with the same accessors (``.sources``, ``.W``, ``.A``, ...) as a
loaded fit. This is the Fortran reference itself, so it is the parity oracle the
Python backends are validated against.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

from ..numpy_impl.load import AmicaOutput, loadmodout
from . import resolver

# Full default parameter set (mirrors sample_data/input.param) so the engine does
# not depend on the sample data being installed. ``files``/``outdir``/``data_dim``/
# ``field_dim`` are set per fit; everything else is a tunable default. Keys are the
# Fortran param names; friendly aliases are mapped in ``fit``.
_DEFAULT_PARAMS: dict[str, object] = {
    "block_size": 512,
    "do_opt_block": 0,
    "blk_min": 256,
    "blk_step": 256,
    "blk_max": 1024,
    "num_models": 1,
    "max_threads": 10,
    "use_min_dll": 1,
    "min_dll": 1e-09,
    "use_grad_norm": 1,
    "min_grad_norm": 1e-07,
    "num_mix_comps": 3,
    "pdftype": 0,
    "max_iter": 2000,
    "num_samples": 1,
    "field_blocksize": 1,
    "do_history": 0,
    "histstep": 10,
    "share_comps": 0,
    "share_start": 100,
    "comp_thresh": 0.99,
    "share_iter": 100,
    "lrate": 0.05,
    "minlrate": 1e-08,
    "mineig": 1e-12,
    "lratefact": 0.5,
    "rholrate": 0.05,
    "rho0": 1.5,
    "minrho": 1.0,
    "maxrho": 2.0,
    "rholratefact": 0.5,
    "kurt_start": 3,
    "num_kurt": 5,
    "kurt_int": 1,
    "do_newton": 1,
    "newt_start": 50,
    "newt_ramp": 10,
    "newtrate": 1.0,
    "do_reject": 0,
    "numrej": 3,
    "rejsig": 3.0,
    "rejstart": 2,
    "rejint": 3,
    "writestep": 20,
    "write_nd": 0,
    "write_LLt": 1,
    "decwindow": 1,
    "max_decs": 3,
    "fix_init": 0,
    "update_A": 1,
    "update_c": 1,
    "update_gm": 1,
    "update_alpha": 1,
    "update_mu": 1,
    "update_beta": 1,
    "invsigmax": 100.0,
    "invsigmin": 0.0,
    "do_rho": 1,
    "load_rej": 0,
    "load_W": 0,
    "load_c": 0,
    "load_gm": 0,
    "load_alpha": 0,
    "load_mu": 0,
    "load_beta": 0,
    "load_rho": 0,
    "load_comp_list": 0,
    "do_mean": 1,
    "do_sphere": 1,
    "doPCA": 1,
    "pcakeep": 0,
    "pcadb": 30.0,
    "byte_size": 4,
    "doscaling": 1,
    "scalestep": 1,
}

# Friendly kwarg -> Fortran param-name aliases (match the Python backends' names).
_ALIASES = {"n_models": "num_models", "n_mix": "num_mix_comps"}


def _render_param(params: dict[str, object]) -> str:
    return "".join(f"{k} {_fmt(v)}\n" for k, v in params.items())


def _fmt(v: object) -> str:
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        return f"{v:.6e}" if (v != 0 and abs(v) < 1e-3) else f"{v:.6f}"
    return str(v)


class AMICANative:
    """Run the native AMICA binary on data and expose the result as an
    ``AmicaOutput``.

    Parameters
    ----------
    binary : path-like, optional
        Explicit binary path; otherwise resolved for the host (downloaded from the
        release on first use). Equivalent to setting ``PAMICA_NATIVE_BINARY``.
    version : str, default "latest"
        Release tag to resolve the binary from when not given explicitly.
    threads : int, optional
        ``OMP_NUM_THREADS`` for the run (default: the binary's own default).
    timeout : float, optional
        Seconds before the subprocess is killed (default: no timeout).
    **params
        Any Fortran ``input.param`` field (or a friendly alias: ``n_models``,
        ``n_mix``), overriding the defaults; e.g. ``max_iter``, ``lrate``,
        ``pdftype``, ``do_newton``.
    """

    def __init__(
        self,
        binary: Optional[Union[str, Path]] = None,
        *,
        version: str = "latest",
        threads: Optional[int] = None,
        timeout: Optional[float] = None,
        **params: object,
    ) -> None:
        # resolve() now: the subprocess runs with cwd set to a tempdir, so a
        # relative binary path would pass the existence check but fail to launch.
        self.binary = Path(binary).resolve() if binary is not None else None
        self.version = version
        self.threads = threads
        self.timeout = timeout
        self.params = params
        self.output_: Optional[AmicaOutput] = None

    def _resolve_binary(self) -> Path:
        if self.binary is not None:
            if not self.binary.exists():
                raise FileNotFoundError(f"binary not found: {self.binary}")
            return self.binary
        return resolver.resolve(self.version)

    def fit(self, X: np.ndarray, **params: object) -> "AMICANative":
        """Run AMICA on ``X`` (shape ``(n_channels, n_samples)``) and store the
        result as ``self.output_`` (an ``AmicaOutput``)."""
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D (n_channels, n_samples); got {X.shape}")
        n_channels, n_samples = X.shape

        merged = dict(_DEFAULT_PARAMS)
        for src in (self.params, params):
            for key, value in src.items():
                merged[_ALIASES.get(key, key)] = value
        merged["data_dim"] = n_channels
        merged["field_dim"] = n_samples
        if not merged.get("pcakeep"):
            merged["pcakeep"] = n_channels  # default: keep all components

        binary = self._resolve_binary()

        with tempfile.TemporaryDirectory(prefix="amica_native_") as td:
            work = Path(td)
            # AMICA reads the data as raw byte_size floats in column-major order
            # (numpy_impl/data.py: reshape order="F"); write it that way.
            byte_size = merged["byte_size"]
            dtype = (
                np.float32
                if isinstance(byte_size, int) and byte_size == 4
                else np.float64
            )
            X.astype(dtype).ravel(order="F").tofile(work / "data.fdt")

            outdir = work / "amicaout"
            outdir.mkdir()
            # `files` must come first: amica15.f90 hard-stops if it parses other
            # keys before the data file. Dict insertion order preserves that.
            param = {"files": "./data.fdt", "outdir": "./amicaout/", **merged}
            (work / "input.param").write_text(_render_param(param))

            env = None
            if self.threads is not None:
                import os

                env = {**os.environ, "OMP_NUM_THREADS": str(self.threads)}

            proc = subprocess.run(
                [str(binary), "input.param"],
                cwd=work,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env=env,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"native AMICA failed (exit {proc.returncode}).\n"
                    f"stdout tail:\n{proc.stdout[-2000:]}\n"
                    f"stderr tail:\n{proc.stderr[-2000:]}"
                )
            if not (outdir / "W").exists():
                raise RuntimeError(
                    "native AMICA produced no output (no 'W' file); stdout tail:\n"
                    f"{proc.stdout[-2000:]}"
                )
            # A collapsed fit writes NaN weights (and zero model probabilities);
            # loadmodout's pinv(W@S) would then fail with an opaque SVD error, so
            # detect it here and report it as the degenerate fit it is (cf. the
            # #50 degenerate-fit contract for the Python backends). An empty W
            # (truncated write) is caught too -- an all-NaN check vacuously passes
            # on a zero-length array.
            w_raw = np.fromfile(outdir / "W")
            if w_raw.size == 0 or not np.all(np.isfinite(w_raw)):
                raise RuntimeError(
                    "native AMICA produced a degenerate fit (non-finite weights); "
                    "the run did not converge. Try more iterations, more data, or "
                    "fewer models."
                )
            self.output_ = loadmodout(outdir)
        return self

    def transform(self, X: np.ndarray, model_idx: int = 0) -> np.ndarray:
        """Source activations for ``X`` from the fitted model (delegates to
        ``AmicaOutput.sources``)."""
        if self.output_ is None:
            raise RuntimeError("call fit() before transform().")
        return self.output_.sources(X, model_idx)
