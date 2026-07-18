"""Keep the release version consistent across the three metadata files.

A pamica release version lives in three places that must agree, because each is
consumed by a different tool at release time:

- ``pyproject.toml``  -> the wheel/sdist that ``publish.yml`` uploads to PyPI.
- ``CITATION.cff``    -> the citation record (``version`` + ``date-released``).
- ``.zenodo.json``    -> the Zenodo archive metadata (``version`` +
  ``publication_date``); Zenodo reads it from the *tag's* tree when the GitHub
  release is published, so it must be correct **before** tagging, not committed
  afterward.

Two modes:

    python scripts/sync_version.py sync 0.1.3            # writes all three
    python scripts/sync_version.py sync 0.1.3 --date 2026-08-01
    python scripts/sync_version.py check 0.1.3           # verifies, exit 1 on drift

``sync`` is the release-prep step (run it, commit "Bump version to X.Y.Z", tag).
``check`` is the release gate that ``publish.yml`` runs against the release tag,
so a mistagged or half-bumped release fails before anything reaches PyPI.

Only string-level substitutions are used so the files' formatting, key order,
and comments are preserved untouched.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
CITATION = ROOT / "CITATION.cff"
ZENODO = ROOT / ".zenodo.json"

_SEMVER = re.compile(r"^\d+\.\d+\.\d+([.\-+][0-9A-Za-z.\-+]+)?$")
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def read_versions() -> dict[str, str]:
    """Return the version string currently declared in each metadata file."""
    pyproject = tomllib.loads(PYPROJECT.read_text())["project"]["version"]

    m = re.search(r"(?m)^version:\s*(.+?)\s*$", CITATION.read_text())
    if m is None:
        raise ValueError("CITATION.cff has no top-level 'version:' key")
    citation = m.group(1).strip().strip('"')

    zenodo = json.loads(ZENODO.read_text()).get("version")
    if zenodo is None:
        raise ValueError(".zenodo.json has no 'version' key")

    return {
        "pyproject.toml": pyproject,
        "CITATION.cff": citation,
        ".zenodo.json": zenodo,
    }


def sync(version: str, released: str) -> None:
    pyproject = re.sub(
        r'(?m)^version = "[^"]*"',
        f'version = "{version}"',
        PYPROJECT.read_text(),
        count=1,
    )
    PYPROJECT.write_text(pyproject)

    citation = CITATION.read_text()
    citation = re.sub(r"(?m)^version:.*$", f"version: {version}", citation, count=1)
    citation = re.sub(
        r"(?m)^date-released:.*$", f'date-released: "{released}"', citation, count=1
    )
    CITATION.write_text(citation)

    zenodo = ZENODO.read_text()
    zenodo = re.sub(r'"version": "[^"]*"', f'"version": "{version}"', zenodo, count=1)
    zenodo = re.sub(
        r'"publication_date": "[^"]*"',
        f'"publication_date": "{released}"',
        zenodo,
        count=1,
    )
    ZENODO.write_text(zenodo)

    print(f"Set version {version} (released {released}) in all three metadata files.")


def check(version: str) -> int:
    versions = read_versions()
    drift = {name: got for name, got in versions.items() if got != version}
    if drift:
        print(f"Version mismatch (expected {version}):", file=sys.stderr)
        for name, got in versions.items():
            mark = "!=" if name in drift else "=="
            print(f"  {name}: {got} {mark} {version}", file=sys.stderr)
        return 1
    print(f"All metadata files agree on version {version}.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)

    p_sync = sub.add_parser("sync", help="write the version into all three files")
    p_sync.add_argument("version", help="release version, e.g. 0.1.3")
    p_sync.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="release date (ISO YYYY-MM-DD); defaults to today",
    )

    p_check = sub.add_parser("check", help="verify all three files match the version")
    p_check.add_argument("version", help="expected version, e.g. 0.1.3")

    args = parser.parse_args()
    version = args.version.removeprefix("v")

    if not _SEMVER.match(version):
        parser.error(f"'{version}' is not a MAJOR.MINOR.PATCH version")

    if args.mode == "sync":
        if not _ISO_DATE.match(args.date):
            parser.error(f"--date '{args.date}' is not ISO YYYY-MM-DD")
        sync(version, args.date)
        return 0
    return check(version)


if __name__ == "__main__":
    raise SystemExit(main())
