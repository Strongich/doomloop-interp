#!/usr/bin/env bash
# Wait for step 2 to finish, then run step 1. One GPU, so they cannot overlap.
set -uo pipefail
cd "$(dirname "$0")/.."
while pgrep -f "suppress_answer.py.*correct120_clean" >/dev/null 2>&1; do sleep 60; done
echo "[chain $(date -u +%H:%M:%S)] step 2 finished, starting step 1"
exec bash scripts/run_tier3_step1.sh
