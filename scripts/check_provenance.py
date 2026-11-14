#!/usr/bin/env python3
"""Check that the frozen files recorded in a result bundle are byte-identical to the files in this checkout.

Each campaign froze its plan, configuration and analysis code by SHA-256 before its first run, and its merged report
records the hash of every frozen file at analysis time (``freeze*`` entries of ``merged_report/*_summary.json``).
This script recomputes the SHA-256 of the same paths in the current checkout and compares them.

    python scripts/check_provenance.py <crossleak_v..._all-<experiment>.zip> [more bundles ...]

Exit code 0 if every recorded file matches, 1 otherwise. Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import sys
import zipfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def recorded_hashes(bundle: Path) -> dict[str, str]:
    """path -> sha256 recorded by the bundle's merged report (all freeze entries of all summaries)."""
    out: dict[str, str] = {}
    with zipfile.ZipFile(bundle) as z:
        for name in z.namelist():
            if "/merged_report/" not in name or not name.endswith("_summary.json"):
                continue
            summary = json.loads(z.read(name))
            for key, val in summary.items():
                if not key.startswith("freeze") or not isinstance(val, dict):
                    continue
                for path, rec in (val.get("files") or {}).items():
                    sha = rec.get("actual") or rec.get("expected")
                    if sha:
                        out[path] = sha
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(bundle: Path, repo: Path = REPO) -> bool:
    rec = recorded_hashes(bundle)
    if not rec:
        print(f"{bundle.name}: no frozen-file record found")
        return False
    ok = True
    for path, want in sorted(rec.items()):
        f = repo / path
        got = sha256(f) if f.is_file() else None
        status = "OK" if got == want else ("MISSING" if got is None else "DIFFERENT")
        ok &= status == "OK"
        print(f"  {status:9} {path}")
    print(f"{bundle.name}: {len(rec)} frozen files, {'all match' if ok else 'MISMATCH'}")
    return ok


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    results = [check(Path(a)) for a in argv]
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
