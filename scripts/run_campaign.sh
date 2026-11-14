#!/usr/bin/env bash
# Operator script for ONE machine: data -> campaign (resumable) -> local report -> artifact zip.
# Usage: bash scripts/run_campaign.sh SHARD NSHARDS [BUDGET_MIN] [PHASE]
#   config:   CROSSLEAK_CONFIG (default configs/main.yaml; E0: configs/e0.yaml; smoke: configs/main_smoke.yaml)
#   phase:    main (default) | cost (edge-cost cells; run on one machine while the others are idle)
#   one machine:   bash scripts/run_campaign.sh 0 1            then   bash scripts/run_campaign.sh 0 1 "" cost
#   five machines: normally started by `bash scripts/cluster.sh run` from the controller.
# Re-running the same command resumes and skips finished cells. At the end it writes
# runs/<experiment>/RUN_EXIT.json (exit code, time, artifact path) for the cluster controller.
set -uo pipefail
cd "$(dirname "$0")/.."
SHARD="${1:?shard index}"
NSHARDS="${2:?number of shards}"
BUDGET="${3:-}"
PHASE="${4:-main}"
CONFIG="${CROSSLEAK_CONFIG:-configs/main.yaml}"
# shellcheck disable=SC1091
. .venv/bin/activate
VERSION="$(python -c 'import crossleak; print(crossleak.__version__)')"
read -r EXP DRIVER < <(python -c "import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); print(c['experiment'], c.get('driver', 'e0'))" "${CONFIG}")
RUNS="${CROSSLEAK_RUNS:-runs}"
mkdir -p "${RUNS}/${EXP}"
exec 9>"${RUNS}/${EXP}/.run.lock"
if ! flock -n 9; then echo "[run] another campaign is already running ${EXP} in $(pwd); not starting a second one"; exit 4; fi
SFX="${EXP}"; case "${EXP}" in *smoke*) SFX="smoke";; esac
echo "[run] v${VERSION} config ${CONFIG} (${DRIVER}) phase ${PHASE} shard ${SHARD}/${NSHARDS} on $(hostname) at $(date -Iseconds)"
finish() {  # finish <rc> <artifact path>
  printf '{"rc": %d, "finished": "%s", "artifact": "%s", "shard": %s, "nshards": %s, "phase": "%s", "host": "%s", "version": "v%s"}\n' \
    "$1" "$(date -Iseconds)" "$2" "${SHARD}" "${NSHARDS}" "${PHASE}" "$(hostname)" "${VERSION}" > "${RUNS}/${EXP}/RUN_EXIT.json"
}
if ! crossleak data all; then
  echo "[run] FAIL: data step"
  ART="$(crossleak artifact --suffix "${SFX}-s${SHARD}of${NSHARDS}-datafail" | awk '/^ARTIFACT/ {print $2}')"
  finish 1 "${ART}"; exit 1
fi
EXTRA=()
if [ -n "${BUDGET}" ]; then EXTRA=(--budget-min "${BUDGET}"); fi
if [ "${DRIVER}" = "main" ] || [ "${DRIVER}" = "v3" ]; then
  crossleak main run --config "${CONFIG}" --shard "${SHARD}" --nshards "${NSHARDS}" --phase "${PHASE}" "${EXTRA[@]}"
  RC=$?
  crossleak main status --config "${CONFIG}" --shard "${SHARD}" --nshards "${NSHARDS}" --phase "${PHASE}"
  crossleak main report --config "${CONFIG}" --decoders none > /dev/null || echo "[run] WARN: local report failed (artifact still made)"
else
  crossleak e0 run --config "${CONFIG}" --shard "${SHARD}" --nshards "${NSHARDS}" "${EXTRA[@]}"
  RC=$?
  crossleak e0 status --config "${CONFIG}" --shard "${SHARD}" --nshards "${NSHARDS}"
  crossleak e0 report --config "${CONFIG}" > /dev/null || echo "[run] WARN: local report failed (artifact still made)"
fi
ART="$(crossleak artifact --suffix "${SFX}-s${SHARD}of${NSHARDS}" | awk '/^ARTIFACT/ {print $2}')"
echo "[run] artifact ${ART}"
finish "${RC}" "${ART}"
echo "[run] finished phase ${PHASE} with campaign exit code ${RC} at $(date -Iseconds)"
exit "${RC}"
