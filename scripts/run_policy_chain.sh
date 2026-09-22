#!/usr/bin/env bash
# Unattended: stage 1 -> gates -> stage 2 -> reports. Survives the operator
# disconnecting, because everything runs inside the pod.
#
# The pre-registration makes stage 2 a HUMAN checkpoint, so that a degenerate
# screen cannot promote itself into the selection. Running unattended removes
# that reviewer, so `--shortlist-out` applies sanity gates in its place and this
# script STOPS rather than proceeds when they fail. The gates catch a broken run
# (grading collapsed, nothing injected, everything truncated), not a
# disappointing one -- a disappointing screen is a finding and still proceeds.
#
# Both stages resume: a completed shard is a no-op, so re-running after an
# interruption continues rather than regenerates.
#
#   tmux new-session -d -s chain "bash scripts/run_policy_chain.sh > /workspace/chain.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/.cache/uv}

SHARDS=${SHARDS:-4}
ROOT=${ROOT:-data/reasoning_policy_v1}
LOGDIR=${LOGDIR:-/workspace}
SL=$LOGDIR/shortlist.json

banner() { echo; echo "===== $* ====="; date "+%F %T"; }

banner "STAGE 1"
if ! SHARDS=$SHARDS STAGE=1 bash scripts/run_policy_sharded.sh; then
  echo "CHAIN ABORTED: stage 1 failed" >&2; exit 1
fi

banner "STAGE 1 REPORT + GATES"
uv run python scripts/report_reasoning_policy.py --run "$ROOT/stage1" \
  --shortlist --shortlist-out "$SL" | tee "$LOGDIR/stage1_report.txt"
uv run python scripts/audit_unboxed.py --run "$ROOT/stage1" \
  | tee "$LOGDIR/stage1_unboxed.txt"

ok=$(uv run python -c "import json;print(json.load(open('$SL'))['ok'])" 2>/dev/null || echo False)
if [ "$ok" != "True" ]; then
  echo "CHAIN STOPPED: stage-1 sanity gates failed. Stage 2 NOT started." >&2
  uv run python -c "
import json
for p in json.load(open('$SL'))['problems']: print('  -', p)" >&2
  exit 2
fi

SHORTLIST=$(uv run python -c "
import json;print(' '.join(json.load(open('$SL'))['shortlist']))")
banner "STAGE 2  [$SHORTLIST]"
if ! SHARDS=$SHARDS STAGE=2 SHORTLIST="$SHORTLIST" bash scripts/run_policy_sharded.sh; then
  echo "CHAIN ABORTED: stage 2 failed" >&2; exit 3
fi

banner "STAGE 2 REPORT"
uv run python scripts/report_reasoning_policy.py --run "$ROOT/stage2" \
  | tee "$LOGDIR/stage2_report.txt"
uv run python scripts/audit_unboxed.py --run "$ROOT/stage2" --out "$LOGDIR/rescued.json" \
  | tee "$LOGDIR/stage2_unboxed.txt"

banner "POWER (whole-question variance, per selected arm)"
for arm in $SHORTLIST; do
  uv run python scripts/power_analysis.py --run "$ROOT/stage2" --arm "$arm" --pool 997 \
    2>/dev/null | sed -n '1,4p;/true effect/,$p'
done | tee "$LOGDIR/stage2_power.txt"

banner "CHAIN COMPLETE"
echo "reports: $LOGDIR/stage{1,2}_report.txt $LOGDIR/stage2_unboxed.txt $LOGDIR/stage2_power.txt"
