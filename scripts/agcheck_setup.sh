#!/usr/bin/env bash
# Create a SEPARATE virtualenv with autogluon.timeseries for the real-system check (docs/AGCHECK.md).
# The main CrossLeak venv is not touched. Idempotent: skips if the marker matches. Usage: bash scripts/agcheck_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."
VENV=".venv-ag"
MARK="${VENV}/.ag_ok"
AG_VERSION="1.6.3"
if [ -f "${MARK}" ] && grep -q "${AG_VERSION}" "${MARK}" 2>/dev/null; then
  echo "[agcheck-setup] ${VENV} already has autogluon ${AG_VERSION}; nothing to do"
  exit 0
fi
PY="$(command -v python3.11 || command -v python3.10 || command -v python3)"
echo "[agcheck-setup] creating ${VENV} with ${PY}"
"${PY}" -m venv "${VENV}"
"${VENV}/bin/pip" install -q --upgrade pip
# CPU torch in the range autogluon 1.6.3 needs, then autogluon.timeseries and pyarrow (for reading the dumps)
"${VENV}/bin/pip" install -q "torch>=2.10,<2.14" --index-url https://download.pytorch.org/whl/cpu
"${VENV}/bin/pip" install -q "autogluon.timeseries==${AG_VERSION}" pyarrow scikit-learn
"${VENV}/bin/python" -c "import autogluon.timeseries, torch; print('autogluon', autogluon.version.__version__, 'torch', torch.__version__)"
echo "autogluon ${AG_VERSION} $(date -Iseconds)" > "${MARK}"
echo "[agcheck-setup] PASS: ${VENV} ready"
