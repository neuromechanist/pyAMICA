"""Install the native AMICA binary for this system (epic #165).

    python -m pamica.native                 # download+cache the latest release binary
    python -m pamica.native --version v0.2.0 # a specific release
    python -m pamica.native --print         # just print where it would resolve to

Downloads the release asset matching the host platform, verifies its SHA-256, and
caches it; prints the resolved path. Idempotent (a cached binary is reused).
"""

from __future__ import annotations

import argparse
import sys

from . import resolver


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m pamica.native")
    parser.add_argument(
        "--version", default="latest", help="release tag (default: latest)"
    )
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_only",
        help="resolve from override/cache only; do not download",
    )
    args = parser.parse_args(argv)
    try:
        path = resolver.resolve(args.version, download=not args.print_only)
    except Exception as exc:  # surface a clear message, not a traceback
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
