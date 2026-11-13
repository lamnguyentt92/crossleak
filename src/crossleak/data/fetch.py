"""Download public datasets with checksum verification and a manifest."""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.request
from pathlib import Path

from crossleak.config import data_dir

log = logging.getLogger("crossleak.fetch")

SOURCES: dict[str, dict] = {
    "eld": {
        "filename": "eld.zip",
        "urls": [
            "https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip",
        ],
        "sha256": "f6c4d0e0df12ecdb9ea008dd6eef3518adb52c559d04a9bac2e1b81dcfc8d4e1",
        "md5": None,
        "licence": "CC BY 4.0",
        "citation": "Trindade A. (2015) ElectricityLoadDiagrams20112014. UCI ML Repository. doi:10.24432/C58C86",
    },
    "lifesnaps": {
        "filename": "lifesnaps.zip",
        "urls": [
            "https://zenodo.org/api/records/7229547/files/rais_anonymized.zip/content",
            "https://zenodo.org/records/7229547/files/rais_anonymized.zip?download=1",
        ],
        "sha256": None,
        "md5": "726afe263ab4b900a721eac19b2ca13a",
        "licence": "CC BY 4.0",
        "citation": "Yfantidou S. et al. LifeSnaps (v4). Zenodo. doi:10.5281/zenodo.7229547",
    },
}


def _hash(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def raw_path(name: str) -> Path:
    return data_dir() / "raw" / SOURCES[name]["filename"]


def verify(name: str, path: Path) -> dict:
    src = SOURCES[name]
    info = {"file": str(path), "bytes": path.stat().st_size, "sha256": _hash(path, "sha256")}
    ok = True
    if src["sha256"]:
        ok &= info["sha256"] == src["sha256"]
    if src["md5"]:
        info["md5"] = _hash(path, "md5")
        ok &= info["md5"] == src["md5"]
    info["checksum_ok"] = bool(ok)
    return info


def fetch(name: str, retries: int = 3) -> dict:
    """Download one dataset if missing; verify checksum; update manifest. Raises on failure."""
    if name not in SOURCES:
        raise ValueError(f"unknown dataset {name}")
    dest = raw_path(name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        info = verify(name, dest)
        if info["checksum_ok"]:
            log.info("%s already present and verified (%d bytes)", name, info["bytes"])
            _write_manifest(name, info)
            return info
        log.warning("%s present but checksum mismatch; re-downloading", name)
    last_err: Exception | None = None
    for url in SOURCES[name]["urls"]:
        for attempt in range(1, retries + 1):
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                log.info("downloading %s from %s (attempt %d)", name, url, attempt)
                t0 = time.time()
                req = urllib.request.Request(url, headers={"User-Agent": "crossleak-research/0.1"})
                with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
                    while True:
                        block = resp.read(1 << 20)
                        if not block:
                            break
                        out.write(block)
                tmp.replace(dest)
                info = verify(name, dest)
                info["url"] = url
                info["download_sec"] = round(time.time() - t0, 1)
                if not info["checksum_ok"]:
                    raise RuntimeError(f"checksum mismatch for {name}: {info}")
                log.info("%s downloaded and verified in %.1fs", name, info["download_sec"])
                _write_manifest(name, info)
                return info
            except Exception as exc:  # noqa: BLE001 - retry any network error
                last_err = exc
                log.warning("download failed: %s", exc)
                if tmp.exists():
                    tmp.unlink()
                time.sleep(5 * attempt)
    raise RuntimeError(f"could not download {name}: {last_err}")


def _write_manifest(name: str, info: dict) -> None:
    path = data_dir() / "raw" / "MANIFEST.json"
    manifest = json.loads(path.read_text()) if path.exists() else {}
    src = SOURCES[name]
    manifest[name] = {**info, "licence": src["licence"], "citation": src["citation"]}
    path.write_text(json.dumps(manifest, indent=2))
