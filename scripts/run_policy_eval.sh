#!/usr/bin/env bash
# Frozen-policy evaluation, unattended: MATH-500, then fresh GSM8K test.
#
# Everything that defines the measurement is fixed in
# data/policy/protocol_v1_locked.json, written before any treated rollout on
# either set. This script refuses to start if a cohort file no longer matches the
# hash recorded there -- a silently regenerated or edited cohort would make the
# "held-out" label false.
#
#   tmux new-session -d -s eval "bash scripts/run_policy_eval.sh > /workspace/eval.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/.cache/uv}

LOCK=data/policy/protocol_v1_locked.json
LOGROOT=${LOGROOT:-/workspace/eval_logs}
ARMS="N@a1.0d256 D@a0.5d512 R101@a1.0d256 R202@a1.0d256 R303@a1.0d256"
COSTS="data/reasoning_policy_v1/stage2 data/reasoning_policy_v1_controls/stage2"

banner() { echo; echo "===== $* ====="; date "+%F %T"; }

uv run python - "$LOCK" <<'PY' || { echo "EVAL REFUSED: cohort does not match the lock" >&2; exit 1; }
import hashlib, json, sys
lock = json.load(open(sys.argv[1]))
for e in lock["evaluations"]:
    got = hashlib.sha256(open(e["cohort"], "rb").read()).hexdigest()
    ok = got == e["sha256"]
    print(f"{e['set']}: {e['cohort']} {'matches lock' if ok else 'MISMATCH'}")
    if not ok:
        sys.exit(1)
PY

run_set() {  # name cohort seeds maxnew outroot
  local name=$1 cohort=$2 seeds=$3 maxnew=$4 outroot=$5
  banner "$name"
  mkdir -p "$LOGROOT/$name"
  if ! SHARDS=4 STAGE=2 COHORT="$cohort" SEEDS="$seeds" MAXNEW="$maxnew" \
       SHORTLIST="$ARMS" CONTROLS="brevityA" REPLICATE=0 PAIR=0 COSTS="$COSTS" \
       OUTROOT="$outroot" PLAN="data/policy/shards_eval_$name.json" \
       LOGDIR="$LOGROOT/$name" bash scripts/run_policy_sharded.sh; then
    echo "EVAL ABORTED: $name failed" >&2; return 1
  fi
  banner "$name REPORT"
  uv run python scripts/report_reasoning_policy.py --run "$outroot/stage2" \
    | tee "$LOGROOT/${name}_report.txt"
  uv run python scripts/audit_unboxed.py --run "$outroot/stage2" \
    | tee "$LOGROOT/${name}_unboxed.txt"
}

run_set math500 data/policy/math500.jsonl 4 32768 data/eval_math500 || exit 2
run_set gsm8k_fresh data/policy/fresh_gsm8k_test_997.jsonl 4 16384 data/eval_gsm8k_fresh || exit 3
banner "EVAL COMPLETE"
