"""Logging: English messages, version and host in every log file name and line."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

from crossleak.config import TAG, hostname, logs_dir

_FMT = f"%(asctime)s %(levelname)s [{TAG}] %(name)s: %(message)s"


def setup_logging(command: str, level: int = logging.INFO) -> Path:
    """Log to console and to logs/crossleak_<version>_<command>_<host>_<timestamp>.log."""
    ldir = logs_dir()
    ldir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    path = ldir / f"crossleak_{TAG}_{command}_{hostname()}_{stamp}.log"
    root = logging.getLogger()
    root.setLevel(level)
    for h in list(root.handlers):
        root.removeHandler(h)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(logging.Formatter(_FMT))
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(logging.Formatter(_FMT))
    root.addHandler(fh)
    root.addHandler(sh)
    for noisy in ("urllib3", "filelock", "huggingface_hub", "transformers", "httpx", "httpx2", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    logging.getLogger("crossleak").info("log file: %s", path)
    return path
