"""Bundle everything needed to analyse a run into one zip.

Name: crossleak_<version>_<YYYYmmdd-HHMMSS>_<host>[_<suffix>].zip  (in ~/crossleak-artifacts).
Contents: runs/, logs/, reports/, configs/, prereg/, the environment snapshot, pip freeze,
git state, and the data manifests (never the raw data).
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from crossleak.config import TAG, artifacts_dir, data_dir, hostname, logs_dir, reports_dir, repo_root, runs_dir

log = logging.getLogger("crossleak.artifact")


def _cmd(args: list[str]) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=60, cwd=repo_root()).stdout
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}"


def snapshot() -> dict:
    from crossleak.doctor import environment

    return {
        "environment": environment(),
        "git_commit": _cmd(["git", "rev-parse", "HEAD"]).strip(),
        "git_describe": _cmd(["git", "describe", "--tags", "--always", "--dirty"]).strip(),
        "git_status": _cmd(["git", "status", "--porcelain"]),
        "release_commit": (repo_root() / "RELEASE_COMMIT").read_text().strip() if (repo_root() / "RELEASE_COMMIT").exists() else None,
        "argv": sys.argv,
    }


def make_artifact(suffix: str | None = None) -> Path:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    sfx = ""
    if suffix:
        sfx = "_" + re.sub(r"[^A-Za-z0-9.-]+", "-", suffix).strip("-")
    name = f"crossleak_{TAG}_{stamp}_{hostname()}{sfx}"
    out_dir = artifacts_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.zip"
    root = repo_root()
    sources: list[tuple[Path, str]] = []
    for base, arc in ((runs_dir(), "runs"), (logs_dir(), "logs"), (reports_dir(), "reports"), (root / "configs", "configs"), (root / "prereg", "prereg")):
        if base.exists():
            for f in sorted(base.rglob("*")):
                if f.is_file():
                    sources.append((f, f"{name}/{arc}/{f.relative_to(base)}"))
    for f in sorted((data_dir() / "raw").glob("MANIFEST.json")) + sorted((data_dir() / "processed").glob("*.report.json")):
        sources.append((f, f"{name}/data_manifest/{f.name}"))
    files_index = []
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for f, arc in sources:
            zf.write(f, arc)
            files_index.append({"path": arc, "bytes": f.stat().st_size})
        zf.writestr(f"{name}/env/snapshot.json", json.dumps(snapshot(), indent=2))
        zf.writestr(f"{name}/env/pip_freeze.txt", _cmd([sys.executable, "-m", "pip", "freeze"]))
        zf.writestr(f"{name}/INDEX.json", json.dumps({"name": name, "version": TAG, "host": hostname(), "created": stamp, "files": files_index}, indent=2))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    log.info("artifact: %s (%.1f MB, sha256 %s, %d files)", path, path.stat().st_size / 1e6, digest, len(files_index))
    print(f"ARTIFACT {path} {path.stat().st_size / 1e6:.1f}MB sha256={digest}")
    return path
