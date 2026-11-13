#!/usr/bin/env bash
# Put every machine's newest artifact into ONE zip with a merged report:
#   ~/crossleak-artifacts/crossleak_v<version>_<datetime>_all-<e0|smoke>.zip
# Normally done by `bash scripts/cluster.sh run`. By hand (after copying each machine's zip into
# ~/crossleak-artifacts on this machine):  bash scripts/bundle_artifacts.sh [--smoke]
cd "$(dirname "$0")/.."
# shellcheck disable=SC1091
. .venv/bin/activate
exec crossleak cluster bundle "$@"
