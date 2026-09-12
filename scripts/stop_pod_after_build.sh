#!/usr/bin/env bash
# Wait for the v2 corpus build to finish, pause, then delete the pod.
#
#   bash scripts/stop_pod_after_build.sh            # 20 minute grace period
#   GRACE_SECS=0 bash scripts/stop_pod_after_build.sh
#
# Runs LOCALLY, not on the pod: a waiter living on the pod would be deleting the
# machine out from under itself, and would never get to report what happened.
#
# /workspace is a 1000Gi PVC (doomloops-interp, longhorn, RWO), NOT emptyDir, so
# deleting the pod keeps every parquet. `make start` re-attaches the same claim.
set -uo pipefail
cd "$(dirname "$0")/.."
export KUBECONFIG="${KUBECONFIG:-$PWD/kubeconfig.yaml}"

POD="${POD:-reasoning-interp}"
NAMESPACE="${NAMESPACE:-vitenko-thesis}"
BUILD_SESSION="${BUILD_SESSION:-rl2}"
RL_PARQUET="${RL_PARQUET:-/workspace/data/rl2/rl.parquet}"
GRACE_SECS="${GRACE_SECS:-1200}"

say () { echo "[stop $(date -u +%H:%M:%S)] $*"; }

say "waiting for build session '$BUILD_SESSION' on $POD"
while kubectl exec -n "$NAMESPACE" "$POD" -- tmux has-session -t "$BUILD_SESSION" \
        >/dev/null 2>&1; do
  sleep 60
done
say "build session ended"

# Report what the build actually produced BEFORE the pod goes away — afterwards
# the only way to check is to bring it back up.
say "final build state:"
kubectl exec -n "$NAMESPACE" "$POD" -- bash -c '
  grep -E "rows$|rows \(|ERROR|Traceback|RL_PARQUET=" /workspace/data/rl2_build.log \
    | tail -12
  echo "--- files ---"
  ls -la /workspace/data/rl2/ 2>/dev/null
' 2>&1 | sed "s/^/    /"

rows=$(kubectl exec -n "$NAMESPACE" "$POD" -- bash -c "
cd /workspace/doomloop-interp && .venv/bin/python -c '
import sys, pyarrow.parquet as pq
try:
    print(pq.read_metadata(sys.argv[1]).num_rows)
except Exception:
    print(0)
' $RL_PARQUET" 2>/dev/null | tr -d "\r")
say "rl.parquet rows: ${rows:-unknown}"
if [[ "${rows:-0}" -lt 1 ]]; then
  say "WARNING: no readable rl.parquet — the build did not finish cleanly."
  say "  Deleting the pod anyway, as asked. The PVC keeps whatever was written,"
  say "  so the build can resume after maintenance (finished stages are skipped)."
fi

say "grace period: ${GRACE_SECS}s before deleting the pod"
sleep "$GRACE_SECS"

say "deleting pod $POD in $NAMESPACE"
kubectl delete pod "$POD" -n "$NAMESPACE"
say "done — PVC doomloops-interp retains /workspace; 'make start' re-attaches it"
