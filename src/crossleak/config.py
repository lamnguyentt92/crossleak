"""Paths, versions and configuration loading."""

from __future__ import annotations

import hashlib
import os
import socket
import zlib
from pathlib import Path
from typing import Any

import yaml

from crossleak import __version__

VERSION = __version__
TAG = f"v{VERSION}"


def repo_root() -> Path:
    """Root of the source checkout (the package is installed in editable mode)."""
    return Path(__file__).resolve().parents[2]


def data_dir() -> Path:
    """Shared raw/processed data directory (immutable inputs, safe to share across versions)."""
    return Path(os.environ.get("CROSSLEAK_DATA", Path.home() / "crossleak-data")).expanduser()


def runs_dir() -> Path:
    return Path(os.environ.get("CROSSLEAK_RUNS", repo_root() / "runs")).expanduser()


def logs_dir() -> Path:
    return Path(os.environ.get("CROSSLEAK_LOGS", repo_root() / "logs")).expanduser()


def reports_dir() -> Path:
    return Path(os.environ.get("CROSSLEAK_REPORTS", repo_root() / "reports")).expanduser()


def artifacts_dir() -> Path:
    return Path(os.environ.get("CROSSLEAK_ARTIFACTS", Path.home() / "crossleak-artifacts")).expanduser()


def hostname() -> str:
    return socket.gethostname().split(".")[0]


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute() and not path.exists():
        path = repo_root() / path
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_path"] = str(path)
    cfg["_sha256"] = sha256_file(path)
    return cfg


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts: Any, base: int = 0) -> int:
    """Deterministic 31-bit seed from arbitrary parts (independent of PYTHONHASHSEED)."""
    text = "|".join(str(p) for p in parts)
    return (zlib.crc32(text.encode("utf-8")) ^ int(base)) & 0x7FFFFFFF
