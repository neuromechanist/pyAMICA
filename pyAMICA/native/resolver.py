"""Resolve the native AMICA binary for the host platform (epic #165, phase 3).

The dependency-free binaries built by ``.github/workflows/release-binaries.yml``
are attached to each GitHub release. This module maps the host to the right asset
and returns a runnable path, downloading and caching (with a SHA-256 check) on
first use. Set ``PAMICA_NATIVE_BINARY`` to a local path to bypass all of this
(used by the tests and for a hand-built binary).
"""

from __future__ import annotations

import hashlib
import os
import platform
import stat
import urllib.request
from pathlib import Path

_REPO = os.environ.get("PAMICA_NATIVE_REPO", "sccn/pyAMICA")
_ENV_BINARY = "PAMICA_NATIVE_BINARY"
_ENV_CACHE = "PAMICA_NATIVE_CACHE"

# windows-arm64 has no native binary yet (issue #173); Windows 11 ARM runs the
# x64 binary via emulation, so it maps there until a native one exists.
_TAG_ALIAS = {"windows-arm64": "windows-x64"}


def platform_tag() -> str:
    """The release-asset tag for the host, e.g. ``linux-x64``.

    Raises ``RuntimeError`` for a platform no binary is built for (rather than
    silently returning a tag whose asset does not exist)."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin":
        if machine in ("arm64", "aarch64"):
            return "macos-arm64"
        raise RuntimeError(
            f"No native AMICA binary for macOS/{machine} (only Apple Silicon "
            "macos-arm64 is built). Build one with native/build.sh, or set "
            f"{_ENV_BINARY}."
        )
    if system == "Linux":
        if machine in ("x86_64", "amd64"):
            return "linux-x64"
        if machine in ("aarch64", "arm64"):
            return "linux-arm64"
    if system == "Windows":
        if machine in ("amd64", "x86_64"):
            return "windows-x64"
        if machine in ("arm64", "aarch64"):
            return "windows-arm64"
    raise RuntimeError(
        f"No native AMICA binary for {system}/{machine}. Set {_ENV_BINARY} to a "
        "locally built binary."
    )


def asset_name(tag: str) -> str:
    """Release-asset filename for a platform tag (applying the arm64->x64
    Windows alias and the .exe suffix)."""
    resolved = _TAG_ALIAS.get(tag, tag)
    ext = ".exe" if resolved.startswith("windows") else ""
    return f"amica15-{resolved}{ext}"


def _default_cache() -> Path:
    if env := os.environ.get(_ENV_CACHE):
        return Path(env)
    if os.environ.get("XDG_CACHE_HOME"):
        return Path(os.environ["XDG_CACHE_HOME"]) / "pamica" / "bin"
    return Path.home() / ".cache" / "pamica" / "bin"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_checksum(binary: Path, sha256_file: Path) -> None:
    """Verify ``binary`` against a ``sha256sum``-format file (``<hex>  <name>``).
    Raises ``ValueError`` on mismatch so a corrupt/tampered download never runs."""
    expected = sha256_file.read_text().split()[0].lower()
    actual = _sha256(binary)
    if actual != expected:
        raise ValueError(
            f"SHA-256 mismatch for {binary.name}: expected {expected}, got "
            f"{actual}. Refusing to use a binary that does not match its checksum."
        )


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url) as r:  # noqa: S310 (fixed github.com host)
        tmp.write_bytes(r.read())
    tmp.replace(dest)


def resolve(version: str = "latest", *, download: bool = True) -> Path:
    """Return a runnable path to the native AMICA binary for this host.

    Resolution order: ``PAMICA_NATIVE_BINARY`` env override -> cached download ->
    fresh download from the ``version`` release (verified against its ``.sha256``
    asset, then marked executable). ``download=False`` restricts to the override
    and cache (no network), raising if neither is present.
    """
    if override := os.environ.get(_ENV_BINARY):
        path = Path(override)
        if not path.exists():
            raise FileNotFoundError(f"{_ENV_BINARY}={override} does not exist.")
        return path

    tag = platform_tag()
    asset = asset_name(tag)
    cached = _default_cache() / version / asset
    if cached.exists():
        return cached

    if not download:
        raise FileNotFoundError(
            f"No cached native binary at {cached} and download=False. Cut a "
            f"release with the binary attached, or set {_ENV_BINARY}."
        )

    base = (
        f"https://github.com/{_REPO}/releases/latest/download"
        if version == "latest"
        else f"https://github.com/{_REPO}/releases/download/{version}"
    )
    _download(f"{base}/{asset}", cached)
    sha_asset = f"amica15-{_TAG_ALIAS.get(tag, tag)}.sha256"
    sha_path = cached.with_name(sha_asset)
    _download(f"{base}/{sha_asset}", sha_path)
    verify_checksum(cached, sha_path)
    cached.chmod(cached.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
    return cached
