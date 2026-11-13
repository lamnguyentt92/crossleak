#!/usr/bin/env bash
# Run CrossLeak on all five testbed machines from ONE of them (the controller, normally VM-C, as root).
# Same conventions as EviCon: addresses in a testbed file (imported from EviCon's testbed.json), root SSH with the
# controller's key to the other four machines.
#
#   bash scripts/cluster.sh init                  # import EviCon's testbed.json (or: init --vm-a IP --vm-b IP --vm-d IP --vm-e IP)
#   bash scripts/cluster.sh check                 # SSH, OS, CPU, RAM, disk, python, Internet of every machine
#   bash scripts/cluster.sh run --smoke           # whole pipeline on tiny sizes
#   nohup bash scripts/cluster.sh run > ~/crossleak-v<version>-cluster.out 2>&1 < /dev/null &    # main study, one command
#   (E0 pilot: add --config configs/e0.yaml)
#   bash scripts/cluster.sh status                # progress on every machine
#   bash scripts/cluster.sh stop                  # stop the runs this tool started
#   bash scripts/cluster.sh bundle                # one zip with everything that exists now (artifacts, logs, outputs)
# `run` = check -> setup every machine -> copy data -> one shard per machine -> monitor -> collect -> ONE zip.
# The zip is written whether the run succeeds or fails:
#   ~/crossleak-artifacts/crossleak_v<version>_<datetime>_all-<main|smoke>[-error].zip
# Run the same command again after an interruption: finished work is skipped.
set -uo pipefail
cd "$(dirname "$0")/.."
VERSION="$(grep -m1 '^version' pyproject.toml | cut -d'"' -f2)"
if [ ! -f .setup_ok ] || [ "$(cat .setup_ok)" != "${VERSION}" ] || [ ! -x .venv/bin/crossleak ]; then
  echo "[cluster] setting up this machine first (v${VERSION})"
  mkdir -p logs
  STAMP="$(date +%Y%m%d-%H%M%S)"
  LOG="logs/crossleak_v${VERSION}_setup-local_$(hostname)_${STAMP}.log"
  if ! bash scripts/setup_vm.sh 2>&1 | tee "${LOG}"; then
    ART="${CROSSLEAK_ARTIFACTS:-$HOME/crossleak-artifacts}"
    mkdir -p "${ART}"
    ZIP="${ART}/crossleak_v${VERSION}_${STAMP}_$(hostname)_setup-error.zip"
    python3 -m zipfile -c "${ZIP}" logs
    echo "[cluster] FAIL: local setup. Everything for analysis: ${ZIP}"
    exit 1
  fi
fi
# shellcheck disable=SC1091
. .venv/bin/activate
exec crossleak cluster "$@"
