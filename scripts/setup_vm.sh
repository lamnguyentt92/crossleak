#!/usr/bin/env bash
# Create this version's own virtualenv and verify it. Safe to re-run; skips if this version is already set up.
# Usage (inside the checkout, e.g. ~/crossleak-v0.1.1):  bash scripts/setup_vm.sh [--force]
set -euo pipefail
cd "$(dirname "$0")/.."
VERSION="$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)"
FORCE=0; [ "${1:-}" = "--force" ] && FORCE=1
if [ "${FORCE}" = 0 ] && [ -f .setup_ok ] && [ "$(cat .setup_ok)" = "${VERSION}" ] && [ -x .venv/bin/crossleak ]; then
  echo "[setup] v${VERSION} already set up in $(pwd) (use --force to redo)"
  echo "[setup] PASS: v${VERSION} ready."
  exit 0
fi
echo "[setup] CrossLeak v${VERSION} in $(pwd) on $(hostname)"
python3 -c 'import sys; assert sys.version_info >= (3, 10), "Python >= 3.10 required"; print("[setup] python", sys.version.split()[0])'
if [ ! -x .venv/bin/python ]; then
  rm -rf .venv
  if ! python3 -m venv .venv 2>/dev/null; then
    rm -rf .venv
    if [ "$(id -u)" = 0 ]; then
      PYV="$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
      echo "[setup] installing python3-venv (apt)"
      DEBIAN_FRONTEND=noninteractive apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv "python${PYV}-venv" >/dev/null \
        || DEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv >/dev/null
      python3 -m venv .venv
    else
      echo "[setup] FAIL: python3-venv missing; run: sudo apt install python3-venv"; exit 1
    fi
  fi
fi
# shellcheck disable=SC1091
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -c 'import torch' 2>/dev/null || python -m pip install --quiet torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install --quiet -e ".[test]"
echo "[setup] unit tests"
python -m pytest -q -p no:cacheprovider
echo "[setup] doctor (downloads both Chronos-2 models on first run)"
crossleak doctor
echo "${VERSION}" > .setup_ok
echo "[setup] PASS: v${VERSION} ready."
