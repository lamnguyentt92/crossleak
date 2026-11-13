#!/usr/bin/env bash
# Progress and errors of this machine's shard.  Usage: bash scripts/status.sh SHARD NSHARDS [PHASE]
#   config: CROSSLEAK_CONFIG (default configs/main.yaml)
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. .venv/bin/activate
CONFIG="${CROSSLEAK_CONFIG:-configs/main.yaml}"
DRIVER="$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1])).get('driver', 'e0'))" "${CONFIG}")"
if [ "${DRIVER}" = "main" ]; then
  crossleak main status --config "${CONFIG}" --shard "${1:-0}" --nshards "${2:-1}" --phase "${3:-main}" 2>/dev/null | grep -v " INFO "
else
  crossleak e0 status --config "${CONFIG}" --shard "${1:-0}" --nshards "${2:-1}" 2>/dev/null | grep -v " INFO "
fi
LAST="$(ls -t logs/crossleak_*-run-shard* 2>/dev/null | head -1)"
if [ -n "${LAST}" ]; then
  echo "--- last log: ${LAST}"
  grep -E "progress|cell done|FAILED|ERROR|budget|campaign finished" "${LAST}" | tail -n 8
  echo "--- errors (if any):"
  grep -nE "ERROR|Traceback" "${LAST}" | tail -n 20 || true
fi
