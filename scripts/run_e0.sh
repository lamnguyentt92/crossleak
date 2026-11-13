#!/usr/bin/env bash
# E0 operator script for ONE machine: data -> campaign (resumable) -> local report -> artifact zip.
# Usage: bash scripts/run_e0.sh SHARD NSHARDS [BUDGET_MIN]
#   one machine:   bash scripts/run_e0.sh 0 1
#   five machines: normally started by `bash scripts/cluster.sh run` from the controller; by hand:
#                  on machine i (i = 0..4) run  bash scripts/run_e0.sh i 5
# Re-running the same command resumes and skips finished cells. At the end it writes
# runs/<experiment>/RUN_EXIT.json (exit code, time, artifact path) for the cluster controller.
set -uo pipefail
cd "$(dirname "$0")/.."
SHARD="${1:?shard index}"
NSHARDS="${2:?number of shards}"
BUDGET="${3:-}"
CONFIG="${CROSSLEAK_CONFIG:-configs/e0.yaml}"   # smoke test: CROSSLEAK_CONFIG=configs/e0_smoke.yaml
# shellcheck disable=SC1091
. .venv/bin/activate
VERSION="$(python -c 'import crossleak; print(crossleak.__version__)')"
EXP="$(python -c "import sys, yaml; print(yaml.safe_load(open(sys.argv[1]))['experiment'])" "${CONFIG}")"
RUNS="${CROSSLEAK_RUNS:-runs}"
mkdir -p "${RUNS}/${EXP}"
exec 9>"${RUNS}/${EXP}/.run.lock"
if ! flock -n 9; then echo "[run_e0] another run_e0.sh is already running ${EXP} in $(pwd); not starting a second one"; exit 4; fi
SFX="e0"; case "${CONFIG}" in *smoke*) SFX="smoke";; esac
echo "[run_e0] v${VERSION} config ${CONFIG} shard ${SHARD}/${NSHARDS} on $(hostname) at $(date -Iseconds)"
finish() {  # finish <rc> <artifact path>
  printf '{"rc": %d, "finished": "%s", "artifact": "%s", "shard": %s, "nshards": %s, "host": "%s", "version": "v%s"}\n' \
    "$1" "$(date -Iseconds)" "$2" "${SHARD}" "${NSHARDS}" "$(hostname)" "${VERSION}" > "${RUNS}/${EXP}/RUN_EXIT.json"
}
if ! crossleak data all; then
  echo "[run_e0] FAIL: data step"
  ART="$(crossleak artifact --suffix "${SFX}-s${SHARD}of${NSHARDS}-datafail" | awk '/^ARTIFACT/ {print $2}')"
  finish 1 "${ART}"; exit 1
fi
EXTRA=()
if [ -n "${BUDGET}" ]; then EXTRA=(--budget-min "${BUDGET}"); fi
crossleak e0 run --config "${CONFIG}" --shard "${SHARD}" --nshards "${NSHARDS}" "${EXTRA[@]}"
RC=$?
crossleak e0 status --config "${CONFIG}" --shard "${SHARD}" --nshards "${NSHARDS}"
crossleak e0 report --config "${CONFIG}" > /dev/null || echo "[run_e0] WARN: local report failed (artifact still made)"
ART="$(crossleak artifact --suffix "${SFX}-s${SHARD}of${NSHARDS}" | awk '/^ARTIFACT/ {print $2}')"
echo "[run_e0] artifact ${ART}"
finish "${RC}" "${ART}"
echo "[run_e0] finished with campaign exit code ${RC} at $(date -Iseconds)"
exit "${RC}"
