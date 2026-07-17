"""Tests for the native Fortran run engine (epic #165, phase 3).

Resolver logic is unit-tested directly; the engine is tested end-to-end against a
REAL native binary (no mocks) on the bundled sample EEG. The binary path comes
from ``PAMICA_NATIVE_BINARY`` (set in CI after building via native/build.sh, or
locally); the E2E tests skip cleanly when it is absent.
"""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from pyAMICA import AMICANative
from pyAMICA.native import resolver
from pyAMICA.numpy_impl.data import load_data_file

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data"
DATA = SAMPLE / "eeglab_data.fdt"
_BINARY = os.environ.get("PAMICA_NATIVE_BINARY")


# --------------------------------------------------------------------------
# Resolver (no binary needed)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "system,machine,expected",
    [
        ("Darwin", "arm64", "macos-arm64"),
        ("Linux", "x86_64", "linux-x64"),
        ("Linux", "aarch64", "linux-arm64"),
        ("Windows", "AMD64", "windows-x64"),
        ("Windows", "ARM64", "windows-arm64"),
    ],
)
def test_platform_tag(monkeypatch, system, machine, expected):
    monkeypatch.setattr("platform.system", lambda: system)
    monkeypatch.setattr("platform.machine", lambda: machine)
    assert resolver.platform_tag() == expected


def test_platform_tag_unsupported_raises(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")  # Intel mac: not built
    with pytest.raises(RuntimeError, match="macos"):
        resolver.platform_tag()


def test_asset_name_alias_and_extension():
    # windows-arm64 has no native binary -> resolves to the x64 asset (#173).
    assert resolver.asset_name("windows-arm64") == "amica15-windows-x64.exe"
    assert resolver.asset_name("windows-x64") == "amica15-windows-x64.exe"
    assert resolver.asset_name("linux-arm64") == "amica15-linux-arm64"
    assert resolver.asset_name("macos-arm64") == "amica15-macos-arm64"


def test_verify_checksum(tmp_path):
    binary = tmp_path / "amica15-linux-x64"
    binary.write_bytes(b"not really a binary, but real bytes")
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    sha = tmp_path / "amica15-linux-x64.sha256"
    sha.write_text(f"{digest}  amica15-linux-x64\n")
    resolver.verify_checksum(binary, sha)  # passes

    binary.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolver.verify_checksum(binary, sha)


def test_resolve_env_override(monkeypatch, tmp_path):
    fake = tmp_path / "amica"
    fake.write_bytes(b"x")
    monkeypatch.setenv("PAMICA_NATIVE_BINARY", str(fake))
    assert resolver.resolve() == fake
    monkeypatch.setenv("PAMICA_NATIVE_BINARY", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        resolver.resolve()


def test_resolve_no_download_without_cache(monkeypatch, tmp_path):
    monkeypatch.delenv("PAMICA_NATIVE_BINARY", raising=False)
    monkeypatch.setenv("PAMICA_NATIVE_CACHE", str(tmp_path))
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    with pytest.raises(FileNotFoundError, match="download=False"):
        resolver.resolve(download=False)


def _fake_source(monkeypatch, tmp_path, *, content: bytes, good_sha: bool):
    """Redirect the resolver's downloader to a local 'release' directory, so the
    staging/verify/atomic-rename path is exercised with real files and no network."""
    src = tmp_path / "release"
    src.mkdir(exist_ok=True)
    (src / "amica15-linux-x64").write_bytes(content)
    digest = hashlib.sha256(content if good_sha else b"other").hexdigest()
    (src / "amica15-linux-x64.sha256").write_text(f"{digest}  amica15-linux-x64\n")

    def fake_download(url, dest):
        dest.write_bytes((src / Path(url).name).read_bytes())

    monkeypatch.setenv("PAMICA_NATIVE_CACHE", str(tmp_path / "cache"))
    monkeypatch.delenv("PAMICA_NATIVE_BINARY", raising=False)
    monkeypatch.setattr("platform.system", lambda: "Linux")
    monkeypatch.setattr("platform.machine", lambda: "x86_64")
    monkeypatch.setattr(resolver, "_download", fake_download)


def test_resolve_download_verifies_and_caches(monkeypatch, tmp_path):
    _fake_source(monkeypatch, tmp_path, content=b"binary-bytes", good_sha=True)
    path = resolver.resolve("v1")
    assert path.exists() and path.read_bytes() == b"binary-bytes"
    assert os.access(path, os.X_OK)  # marked executable only after verifying
    # second call hits the cache (same path, no re-download needed)
    assert resolver.resolve("v1") == path


def test_resolve_bad_checksum_does_not_poison_cache(monkeypatch, tmp_path):
    _fake_source(monkeypatch, tmp_path, content=b"tampered", good_sha=False)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        resolver.resolve("v1")
    # The unverified binary must NOT have landed at the cache path (else future
    # calls would return it unchecked -- the security bug this guards).
    cached = tmp_path / "cache" / "v1" / "amica15-linux-x64"
    assert not cached.exists()
    # A subsequent good download still succeeds (cache slot not poisoned).
    _fake_source(monkeypatch, tmp_path, content=b"good", good_sha=True)
    assert resolver.resolve("v1").read_bytes() == b"good"


# --------------------------------------------------------------------------
# Engine end-to-end (real binary required)
# --------------------------------------------------------------------------
requires_binary = pytest.mark.skipif(
    not _BINARY or not Path(_BINARY).exists(),
    reason="set PAMICA_NATIVE_BINARY to a built native binary for E2E tests",
)


@pytest.fixture(scope="module")
def real_data():
    if not DATA.exists():
        pytest.skip("sample data missing")
    return load_data_file(str(DATA), 32, 30504, dtype=np.float32).astype(np.float64)


@requires_binary
def test_native_engine_runs_and_produces_amica_output(real_data):
    eng = AMICANative(binary=_BINARY, n_models=1, n_mix=3, max_iter=8, threads=2)
    eng.fit(real_data)
    out = eng.output_
    assert out is not None
    assert out.num_models == 1
    assert out.W.shape == (32, 32, 1)
    assert out.A.shape == (32, 32, 1)
    assert np.all(np.isfinite(out.W))
    # A working decomposition converges to a finite, negative per-sample LL.
    assert np.isfinite(out.LL[-1]) and out.LL[-1] < 0

    sources = eng.transform(real_data)
    assert sources.shape == (32, real_data.shape[1])
    assert np.all(np.isfinite(sources))


@requires_binary
def test_native_engine_param_aliases_and_multimodel(real_data):
    # n_models/n_mix aliases reach the Fortran num_models/num_mix_comps. Full data
    # + enough iterations so the 2-model fit converges (a too-short multi-model run
    # collapses -- exercised separately below).
    eng = AMICANative(binary=_BINARY, n_models=2, n_mix=3, max_iter=15, threads=2)
    eng.fit(real_data)
    assert eng.output_ is not None
    assert eng.output_.num_models == 2
    assert eng.output_.W.shape == (32, 32, 2)
    assert np.all(np.isfinite(eng.output_.W))


@requires_binary
def test_native_engine_degenerate_fit_raises_clearly(real_data):
    # A too-short multi-model fit collapses to non-finite weights; the engine must
    # report that as a degenerate fit, not let loadmodout's pinv raise an opaque
    # SVD error (cf. the #50 degenerate-fit contract).
    eng = AMICANative(binary=_BINARY, n_models=2, max_iter=3, threads=2)
    with pytest.raises(RuntimeError, match="degenerate"):
        eng.fit(real_data[:, :2048])


@requires_binary
def test_native_engine_rejects_non_2d(real_data):
    with pytest.raises(ValueError, match="2-D"):
        AMICANative(binary=_BINARY).fit(real_data[0])


def test_native_engine_missing_binary_errors(tmp_path):
    eng = AMICANative(binary=tmp_path / "nope")
    with pytest.raises(FileNotFoundError):
        eng.fit(np.zeros((4, 100)))
