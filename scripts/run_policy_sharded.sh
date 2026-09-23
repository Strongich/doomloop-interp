#!/usr/bin/env bash
# Data-parallel policy sweep: one single-GPU vLLM engine per shard.
#
# Tensor parallelism is refused by the steering hook, so scale-out is N
# independent engines split by policy. See scripts/plan_policy_shards.py for why
# matched N/D pairs stay together and which policies are replicated.
#
# Each shard writes its OWN directory; they are never appended to a shared
# journal, because the manifest pins the policy list per directory precisely so
# two configurations cannot silently land in one file.
#
#   SHARDS=4 STAGE=1 bash scripts/run_policy_sharded.sh
#   STAGE=2 SHORTLIST="N@a0.5d256 D@a1.0d0 ..." bash scripts/run_policy_sharded.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SHARDS=${SHARDS:-4}
STAGE=${STAGE:-1}
OUTROOT=${OUTROOT:-data/reasoning_policy_v1}
MAXNEW=${MAXNEW:-16384}
BATCH=${BATCH:-200}
CHECKPOINT=${CHECKPOINT:-200}
BATCHTOK=${BATCHTOK:-8192}
GPUFRAC=${GPUFRAC:-0.90}
LOGDIR=${LOGDIR:-/workspace}
# `-` not `:-`: an explicitly EMPTY value means "no prompt controls". With `:-`,
# CONTROLS= silently fell back to the default and added both brevity arms.
CONTROLS=${CONTROLS-"brevityA brevityB"}

# --- stage-specific settings -------------------------------------------------
# Stage 2 is a different experiment, not a different output directory: another
# cohort, another seed count, and only the shortlisted policies. Deriving those
# from STAGE rather than leaving them as defaults is what stops STAGE=2 from
# silently re-running the stage-1 grid under a stage-2 name.
case "$STAGE" in
  1)
    COHORT=${COHORT:-data/policy/dev200.jsonl}
    SEEDS=${SEEDS:-1}
    PLAN_ARGS=(--stage 1)
    ;;
  2)
    COHORT=${COHORT:-data/policy/dev400.jsonl}
    SEEDS=${SEEDS:-2}
    : "${SHORTLIST:?stage 2 requires SHORTLIST (the stage-1 nominations)}"
    PLAN_ARGS=(--stage 2 --policies "$SHORTLIST" --controls $CONTROLS)
    # REPLICATE=0 runs the baseline once instead of on every shard (evaluation).
    [ "${REPLICATE:-1}" = "0" ] && PLAN_ARGS+=(--no-replicate)
    # PAIR=0 packs N and D independently; COSTS= measured run dirs for packing.
    [ "${PAIR:-1}" = "0" ] && PLAN_ARGS+=(--no-pair)
    [ -n "${COSTS:-}" ] && PLAN_ARGS+=(--from-run $COSTS)
    ;;
  *)
    echo "STAGE must be 1 or 2 (got '$STAGE')" >&2; exit 1 ;;
esac
[ -f "$COHORT" ] || { echo "No cohort at $COHORT" >&2; exit 1; }

# --- device selection --------------------------------------------------------
# Respect an inherited allocation. Replacing CUDA_VISIBLE_DEVICES with 0..N-1
# would renumber against the mask: with an inherited "2,3,5,7" the children would
# be handed 0,1,2,3, which are either different cards or not allocated at all.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -r -a DEVICES <<< "$CUDA_VISIBLE_DEVICES"
  echo "inheriting allocated devices: ${DEVICES[*]}"
else
  mapfile -t DEVICES < <(nvidia-smi --query-gpu=index --format=csv,noheader | tr -d ' ')
  echo "no inherited mask; discovered devices: ${DEVICES[*]}"
fi
if [ "${#DEVICES[@]}" -lt "$SHARDS" ]; then
  echo "Asked for $SHARDS shards but only ${#DEVICES[@]} device(s) allocated: ${DEVICES[*]}" >&2
  exit 1
fi

PLAN=${PLAN:-data/policy/shards_stage$STAGE.json}
uv run python scripts/plan_policy_shards.py --shards "$SHARDS" "${PLAN_ARGS[@]}" \
  --out "$PLAN" --questions "$(wc -l < "$COHORT")"

planned=$(uv run python -c "import json;print(len(json.load(open('$PLAN'))['shards']))")
[ "$planned" -eq "$SHARDS" ] || { echo "Plan has $planned shards, asked for $SHARDS" >&2; exit 1; }

echo "stage=$STAGE shards=$SHARDS cohort=$COHORT seeds=$SEEDS batch=$BATCH"

pids=(); starts=()
for i in $(seq 0 $((SHARDS-1))); do
  dev=${DEVICES[$i]}
  policies=$(uv run python -c "
import json;print(' '.join(json.load(open('$PLAN'))['shards'][$i]))")
  out=$OUTROOT/stage${STAGE}_shard$i
  echo "GPU $dev -> $out"
  echo "  $policies"
  starts+=("$SECONDS")
  CUDA_VISIBLE_DEVICES=$dev uv run python scripts/reasoning_policy_vllm.py \
    --cohort "$COHORT" --outdir "$out" --policies $policies \
    --seeds "$SEEDS" --max-new-tokens "$MAXNEW" \
    --batch "$BATCH" --checkpoint-every "$CHECKPOINT" \
    --gpu-memory-utilization "$GPUFRAC" --max-num-batched-tokens "$BATCHTOK" \
    > "$LOGDIR/stage${STAGE}_shard$i.log" 2>&1 &
  pids+=($!)
done

echo "launched ${#pids[@]} shards: ${pids[*]}"
fail=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "shard $i finished after $((SECONDS-${starts[$i]}))s"
  else
    echo "shard $i FAILED after $((SECONDS-${starts[$i]}))s" >&2; fail=1
  fi
done
[ "$fail" -eq 0 ] || { echo "inspect $LOGDIR/stage${STAGE}_shard*.log" >&2; exit 1; }

echo "all shards complete; measured wall times above -- compare against the plan's estimate"
uv run python scripts/merge_policy_shards.py --root "$OUTROOT" --stage "$STAGE" --plan "$PLAN"
