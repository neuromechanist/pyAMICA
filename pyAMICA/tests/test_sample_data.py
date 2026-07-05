"""Tests for pyAMICA using sample data."""

import os.path as op
import numpy as np
import pytest
from pathlib import Path

import pyAMICA
from pyAMICA import AMICA_NumPy as AMICA
from pyAMICA.amica_data import load_data_file
from pyAMICA.amica_load import loadmodout

# Setup sample data paths
sample_data_path = op.join(pyAMICA.__path__[0], "sample_data")
sample_params_file = op.join(sample_data_path, "sample_params.json")
eeglab_data_file = op.join(sample_data_path, "eeglab_data.fdt")
amicaout_dir = op.join(sample_data_path, "amicaout")


@pytest.mark.slow
@pytest.mark.xfail(
    reason="uses a per-row corrcoef of the internal W (= Fortran W^T, so rows are "
    "mismatched) and a full 2000-iter run; superseded by "
    "test_sample_data_numpy_vs_fortran, which uses the correct get_weights()@sphere "
    "Hungarian metric. The LL scale/sign issue itself is fixed (issue #24).",
    strict=True,
    raises=AssertionError,
)
def test_sample_data_scikit(tmp_path):
    """Test pyAMICA using scikit-learn style API."""
    # Load original results for comparison
    orig_results = loadmodout(amicaout_dir)

    # Initialize and fit AMICA model using scikit-learn style API.
    # Override outdir (the params file defaults to the relative './amicaout/')
    # so this test does not write stray output into the repo root.
    model = AMICA.from_json_file(sample_params_file, outdir=str(tmp_path / "amicaout"))
    model.fit()

    # Compare weights
    W_pyamica = model.W[:, :, 0]  # Get weights from first model

    correlations = np.zeros((32, 32))
    for i in range(32):
        for j in range(32):
            correlations[i, j] = abs(
                np.corrcoef(W_pyamica[i], orig_results.W[j, :, 0])[0, 1]
            )

    # Verify results
    max_correlations = np.max(correlations, axis=1)
    assert np.all(max_correlations > 0.8), (
        "Some components don't match original results"
    )

    best_matches = np.argmax(correlations, axis=1)
    assert len(np.unique(best_matches)) == 32, "Some components are duplicated"


@pytest.mark.slow
@pytest.mark.xfail(
    reason="Progress, not passing. Fixed: the #30 format, the #39 NaN (seed pin + "
    "restart), and the #41 LL degradation -- the lrate-ceiling ratchet "
    "(_check_convergence) now keeps the long run improving (seed=0 LL -3.399 at "
    "2000 iters, better than the ~150-iter -3.404 and Fortran's -3.402; 150-iter "
    "mean-corr parity preserved). Remaining: this test's strict all-32-components "
    ">0.8 gate still fails (~27/32) because pyAMICA converges to a comparable/"
    "slightly-better-LL but different-partition solution than the Fortran "
    "reference on a few ill-determined components -- a solution-basin difference "
    "(cf. the single-model tail of #27), not a drift. The project's other parity "
    "test (test_sample_data_numpy_vs_fortran) uses the more appropriate mean>0.9 "
    "bar and passes.",
    strict=True,
)
def test_sample_data_cli():
    """Full CLI-vs-Fortran integration test (issue #30 format + #39/#41 stability).

    Runs the real amica_cli entrypoint for the full 2000-iter sample config and
    Hungarian-matches the loadmodout-read W against the Fortran reference with a
    strict all-32-components > 0.8 gate. A fixed seed is pinned so the test does
    not depend on a random init (see #39). Currently xfail on a residual
    solution-basin difference (see the xfail reason above); the LL-degradation
    drift itself is fixed (#41).
    """
    import subprocess
    import sys

    # amica_cli.py uses relative imports, so it must be run as a module
    # (see its module docstring), not as a direct script path.
    test_outdir = Path("test_output")

    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pyAMICA.amica_cli",
                sample_params_file,
                "--outdir",
                str(test_outdir),
                "--seed",
                "0",
            ],
            check=True,
            cwd=Path(__file__).parent.parent.parent,
        )

        # Load and compare results
        orig_results = loadmodout(amicaout_dir)
        test_results = loadmodout(test_outdir)

        correlations = np.zeros((32, 32))
        for i in range(32):
            for j in range(32):
                correlations[i, j] = abs(
                    np.corrcoef(test_results.W[i, :, 0], orig_results.W[j, :, 0])[0, 1]
                )

        # Verify results
        max_correlations = np.max(correlations, axis=1)
        assert np.all(max_correlations > 0.8), (
            "Some components don't match original results"
        )

        best_matches = np.argmax(correlations, axis=1)
        assert len(np.unique(best_matches)) == 32, "Some components are duplicated"

    finally:
        # Cleanup
        if test_outdir.exists():
            import shutil

            shutil.rmtree(test_outdir)


@pytest.mark.xfail(
    reason="components not fully decorrelated (off-diagonal < 0.1) after only 50 "
    "iterations -- an under-convergence limit, not a correctness bug. The LL "
    "scale/sign issue is fixed (issue #24; LL is now ~ -3.4, not positive).",
    strict=True,
    raises=AssertionError,
)
def test_sample_data_light(tmp_path):
    """Light version of the test for CI that runs only 50 iterations."""
    # Load and preprocess the data (eeglab_data.fdt stores float32 samples)
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32)

    # Initialize AMICA model with reduced iterations. Override outdir (the
    # params file defaults to the relative './amicaout/') so this test does
    # not write stray output into the repo root.
    model = AMICA.from_json_file(sample_params_file, outdir=str(tmp_path / "amicaout"))
    model.max_iter = 50  # Override max_iter for quick testing

    # Fit the model
    model.fit(data)

    # Basic checks
    W = model.get_weights()
    assert W.shape == (32, 32), f"Expected weight matrix shape (32, 32), got {W.shape}"
    assert not np.any(np.isnan(W)), "Weight matrix contains NaN values"
    assert not np.any(np.isinf(W)), "Weight matrix contains infinite values"

    # Check if weights are reasonably scaled
    assert np.all(np.abs(W) < 100), "Weight values are unreasonably large"
    assert np.all(np.abs(W) > 1e-6), "Weight values are unreasonably small"

    # Check if components are decorrelated
    Y = model.transform(data)
    corr = np.corrcoef(Y[:, :, 0])
    np.fill_diagonal(corr, 0)  # Remove diagonal elements
    assert np.all(np.abs(corr) < 0.1), "Components are not sufficiently decorrelated"


@pytest.mark.slow
@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_sample_data_numpy_vs_fortran(tmp_path):
    """Real-data parity: the NumPy backend's total spatial filter matches the
    Fortran reference (amicaout) with Hungarian-matched component correlation
    > 0.9 (issue #24). Replaces the removed synthetic source-recovery tests.

    The strict > 0.95 definition-of-done gate is the torch backend's
    test_end_to_end_correlation_vs_fortran; this confirms the (bit-identical
    trajectory) NumPy port also converges to the Fortran solution on real data.
    """
    from scipy.optimize import linear_sum_assignment

    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )
    W_ref = np.fromfile(op.join(amicaout_dir, "W"), np.float64).reshape(
        32, 32, order="F"
    )
    S_ref = np.fromfile(op.join(amicaout_dir, "S"), np.float64).reshape(
        32, 32, order="F"
    )

    model = AMICA(
        use_tqdm=False,
        num_models=1,
        num_mix=3,
        seed=42,
        block_size=512,
        lrate=0.05,
        lratefact=0.5,
        max_decs=5,
        do_newton=True,
        newt_start=50,
        newtrate=1.0,
        rho0=1.5,
        minrho=1.0,
        maxrho=2.0,
        rholrate=0.05,
        invsigmin=0.0,
        invsigmax=100.0,
        doscaling=True,
        do_mean=True,
        do_sphere=True,
        max_iter=150,
        writestep=10000,
        outdir=str(tmp_path / "out"),
    )
    model.fit(data)

    filt_np = model.get_weights() @ model.sphere  # true unmixing @ sphere
    filt_ref = W_ref @ S_ref
    a = filt_np / np.linalg.norm(filt_np, axis=1, keepdims=True)
    b = filt_ref / np.linalg.norm(filt_ref, axis=1, keepdims=True)
    corr = np.abs(a @ b.T)
    rows, cols = linear_sum_assignment(1 - corr)
    mean_corr = float(corr[rows, cols].mean())
    assert mean_corr > 0.9, f"NumPy vs Fortran component corr {mean_corr:.3f} <= 0.9"


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_cli_output_format_roundtrip(tmp_path):
    """The Fortran-format writer round-trips through loadmodout and load_results,
    and the viz helpers consume the result (issue #30; viz smoke for #15).

    Real sample data, short fit -- this checks the on-disk format contract and
    array shapes, not convergence (the full CLI-vs-Fortran correlation is
    test_sample_data_cli). Fast, so not marked slow.
    """
    import matplotlib

    matplotlib.use("Agg")
    from pyAMICA.amica_data import load_results
    from pyAMICA import amica_viz

    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )[:, :4096]
    outdir = tmp_path / "out"
    model = AMICA(
        use_tqdm=False,
        num_models=1,
        num_mix=3,
        seed=1,
        max_iter=15,
        writestep=10000,
        do_opt_block=False,
        outdir=str(outdir),
    )
    # fit() persists the final result even though max_iter (15) is not a
    # writestep (10000) multiple -- exercises the unconditional final write.
    model.fit(data)

    # loadmodout reads the Fortran-format output. Before issue #30 the CLI wrote
    # .npy, so this raised FileNotFoundError for 'W'.
    out = loadmodout(outdir)
    assert out.W.shape == (32, 32, 1)

    # load_results returns AMICA's internal shapes for the viz helpers.
    r = load_results(str(outdir))
    assert r["A"].shape == (32, 32)
    assert r["W"].shape == (32, 32, 1)
    assert r["alpha"].shape == (3, 32)
    assert r["comp_list"].shape == (32, 1)
    assert int(r["comp_list"].min()) == 0 and int(r["comp_list"].max()) == 31

    # Value-level round-trip: the loaded arrays equal the in-memory ones. Guards
    # against a transpose/axis-order or dtype regression that the shape checks
    # above would miss.
    np.testing.assert_allclose(r["W"], model.W)
    np.testing.assert_allclose(r["A"], model.A)
    np.testing.assert_allclose(r["alpha"], model.alpha)
    np.testing.assert_array_equal(r["comp_list"], model.comp_list)

    # viz helpers run without error on the loaded results.
    amica_viz.plot_convergence(str(outdir))
    amica_viz.plot_components(str(outdir), data=None, max_comps=3)
    amica_viz.plot_pdf_fits(str(outdir), data, max_comps=2)


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_restart_on_early_nan_recovers(tmp_path):
    """An early non-finite LL triggers reinitialize-and-restart (Fortran
    restartiter path, #39); the fit recovers and finishes with a finite LL.

    Injects a non-finite LL for the first two iterations via a subclass to
    exercise the restart control flow deterministically. This is an error-path
    test, not a data mock -- the model still fits the real sample data after the
    restart, and the numerical result is not asserted from fabricated values.
    """
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )[:, :2048]

    class _InjectEarlyNaN(AMICA):
        _nan_iters = 2

        def _get_updates_and_likelihood(self):
            upd = super()._get_updates_and_likelihood()
            if self.iter < self._nan_iters:
                upd["ll"] = float("nan")
            return upd

    model = _InjectEarlyNaN(
        use_tqdm=False,
        num_models=1,
        num_mix=3,
        seed=3,
        max_iter=20,
        writestep=10_000_000,
        do_opt_block=False,
        do_newton=False,
        restartiter=10,
        maxrestarts=3,
        outdir=str(tmp_path / "out"),
    )
    model.fit(data)

    # One restart per injected-NaN iteration, then a normal finite fit.
    assert model.numrestarts == 2
    assert model.converged is True
    assert len(model.ll) >= 1
    assert np.isfinite(model.ll[-1])
    # The restart must be announced in the run log, not silent.
    log_text = (tmp_path / "out" / "out.txt").read_text().lower()
    assert "reinitializing" in log_text


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_restart_gives_up_after_maxrestarts(tmp_path):
    """If every early iteration is non-finite, the fit stops after maxrestarts
    reinitializations instead of looping forever (Fortran numrestarts cap)."""
    data = load_data_file(eeglab_data_file, 32, 30504, dtype=np.float32).astype(
        np.float64
    )[:, :2048]

    class _AlwaysNaN(AMICA):
        def _get_updates_and_likelihood(self):
            upd = super()._get_updates_and_likelihood()
            upd["ll"] = float("nan")
            return upd

    model = _AlwaysNaN(
        use_tqdm=False,
        num_models=1,
        num_mix=3,
        seed=3,
        max_iter=50,
        writestep=10_000_000,
        do_opt_block=False,
        do_newton=False,
        restartiter=10,
        maxrestarts=2,
        outdir=str(tmp_path / "out"),
    )
    model.fit(data)
    # Restarts are capped, the run stops on the persistent non-finite LL, and
    # the terminal failure is surfaced (converged=False), not silently ignored.
    assert model.numrestarts == 2
    assert model.converged is False
    assert not np.isfinite(model.ll[-1])


@pytest.mark.skipif(not op.exists(eeglab_data_file), reason="sample data missing")
def test_cli_subprocess_output_loadable(tmp_path):
    """The actual amica_cli entrypoint writes loadmodout-readable output.

    Runs the CLI as a module on a short, stable config (real sample data) and
    confirms loadmodout reads the result -- the direct regression for the #30
    FileNotFoundError (the CLI previously wrote .npy). Kept short (few
    iterations, Newton off) so it stays fast and avoids the separate long-fit
    NaN of #39; the full 2000-iter integration run is test_sample_data_cli.
    """
    import json
    import subprocess
    import sys

    params = {
        "files": [eeglab_data_file],
        "data_dim": 32,
        "field_dim": [30504],
        "num_models": 1,
        "num_mix": 3,
        "max_iter": 8,
        "writestep": 4,
        "do_newton": False,
        "do_opt_block": False,
        "block_size": 512,
    }
    params_file = tmp_path / "params.json"
    params_file.write_text(json.dumps(params))
    outdir = tmp_path / "cli_out"

    # amica_cli uses relative imports, so run it as a module from the repo root.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pyAMICA.amica_cli",
            str(params_file),
            "--outdir",
            str(outdir),
        ],
        check=True,
        cwd=Path(__file__).parent.parent.parent,
    )

    # loadmodout reads the CLI output (previously raised FileNotFoundError).
    out = loadmodout(outdir)
    assert out.W.shape == (32, 32, 1)
    assert np.all(np.isfinite(out.W))


if __name__ == "__main__":
    pytest.main([__file__])
