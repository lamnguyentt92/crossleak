#!/usr/bin/env bash
# E0 operator script (kept for the E0 pilot): same as run_campaign.sh with the E0 config.
# Usage: bash scripts/run_e0.sh SHARD NSHARDS [BUDGET_MIN]
cd "$(dirname "$0")/.."
export CROSSLEAK_CONFIG="${CROSSLEAK_CONFIG:-configs/e0.yaml}"
exec bash scripts/run_campaign.sh "$@"
