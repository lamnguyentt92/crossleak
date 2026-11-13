#!/usr/bin/env bash
# Bundle runs, logs, reports, configs, prereg and environment into one zip in ~/crossleak-artifacts.
# Usage: bash scripts/make_artifact.sh [suffix]
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. .venv/bin/activate
if [ -n "${1:-}" ]; then crossleak artifact --suffix "$1"; else crossleak artifact; fi
