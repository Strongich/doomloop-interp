#!/usr/bin/env bash
# Data-parallel policy sweep: one single-GPU vLLM engine per shard.
#
# Tensor parallelism is refused by the steering hook, so scale-out is N
# independent engines split by policy. Read scripts/plan_policy_shards.py for why
# the split keeps matched N/D pairs together and replicates the baseline.
#
# Each shard writes its OWN output directory. They are never merged by appending
# to a shared journal: the manifest pins the policy list per directory precisely
# so two configurations cannot silently land in one file.
#
#   SHARDS=4 bash scripts/run_policy_sharded.sh
#   STAGE=2 SHARDS=4 SHORTLIST="..." bash scripts/run_policy_sharded.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SHARDS=${SHARDS:-4}
STAGE=${STAGE:-1}
OUTROOT=${OUTROOT:-data/reasoning_policy_v1}
COHORT=${COHORT:-data/policy/dev200.jsonl}
SEEDS=${SEEDS:-1}
MAXNEW=${MAXNEW:-16384}
BATCH=${BATCH:-200}
CHECKPOINT=${CHECKPOINT:-200}
BATCHTOK=${BATCHTOK:-8192}
GPUFRAC=${GPUFRAC:-0.90}
PLAN=${PLAN:-data/policy/shards.json}
LOGDIR=${LOGDIR:-/workspace}

avail=$(nvidia-smi --list-gpus | wc -l)
if [ "$avail" -lt "$SHARDS" ]; then
  echo "Asked for $SHARDS shards but only $avail GPUs are visible" >&2; exit 1
fi
[ -f "$PLAN" ] || { echo "No shard plan at $PLAN; run plan_policy_shards.py" >&2; exit 1; }

echo "stage=$STAGE shards=$SHARDS cohort=$COHORT seeds=$SEEDS batch=$BATCH"
nvidia-smi --query-gpu=index,name,memory.used --format=csv

pids=()
for i in $(seq 0 $((SHARDS-1))); do
  policies=$(uv run python -c "
import json,sys
print(' '.join(json.load(open('$PLAN'))['shards'][$i]))")
  out=$OUTROOT/stage${STAGE}_shard$i
  echo "GPU $i -> $out"
  echo "  $policies"
  CUDA_VISIBLE_DEVICES=$i uv run python scripts/reasoning_policy_vllm.py \
    --cohort "$COHORT" --outdir "$out" --policies $policies \
    --seeds "$SEEDS" --max-new-tokens "$MAXNEW" \
    --batch "$BATCH" --checkpoint-every "$CHECKPOINT" \
    --gpu-memory-utilization "$GPUFRAC" --max-num-batched-tokens "$BATCHTOK" \
    > "$LOGDIR/stage${STAGE}_shard$i.log" 2>&1 &
  pids+=($!)
done

echo "launched ${#pids[@]} shards: ${pids[*]}"
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
if [ "$fail" -ne 0 ]; then
  echo "At least one shard failed; inspect $LOGDIR/stage${STAGE}_shard*.log" >&2; exit 1
fi
echo "all shards complete"
uv run python scripts/merge_policy_shards.py --root "$OUTROOT" --stage "$STAGE"
