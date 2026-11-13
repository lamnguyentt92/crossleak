#!/usr/bin/env bash
# Progress and errors of this machine's E0 shard.  Usage: bash scripts/status.sh SHARD NSHARDS
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. .venv/bin/activate
crossleak e0 status --shard "${1:-0}" --nshards "${2:-1}" 2>/dev/null | grep -v " INFO "
LAST="$(ls -t logs/crossleak_*_e0-run-* 2>/dev/null | head -1)"
if [ -n "${LAST}" ]; then
  echo "--- last log: ${LAST}"
  grep -E "progress|cell done|FAILED|ERROR|budget|campaign finished" "${LAST}" | tail -n 8
  echo "--- errors (if any):"
  grep -nE "ERROR|Traceback" "${LAST}" | tail -n 20 || true
fi
