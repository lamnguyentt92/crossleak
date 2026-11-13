#!/usr/bin/env bash
# Live dashboard of a running cluster campaign, refreshed every 30 s (read-only; Ctrl-C to quit).
# Stops by itself when the controller has written the result zip and prints its path.
#   bash scripts/watch.sh                 # watch the run of this version
#   bash scripts/watch.sh 0.1.2           # watch the run of another installed version (e.g. one already running)
#   bash scripts/watch.sh 0.1.2 --once    # print once
#   bash scripts/watch.sh 0.1.2 --smoke   # watch a smoke run
HERE="$(cd "$(dirname "$0")" && pwd)"
V="$(grep -m1 '^version' "${HERE}/../pyproject.toml" | cut -d'"' -f2)"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then V="$1"; shift; fi
PY="${HOME}/crossleak-v${V}/.venv/bin/python"
if [ ! -x "${PY}" ]; then echo "[watch] ${PY} not found: is v${V} installed in ~/crossleak-v${V}?"; exit 1; fi
cd "${HOME}/crossleak-v${V}" || exit 1
exec "${PY}" "${HERE}/watch.py" --version "${V}" "$@"
