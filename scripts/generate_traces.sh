#!/usr/bin/env bash
# Sample reasoning traces on BOTH GPUs, then merge into one file per dataset.
#
#   scripts/generate_traces.sh                          # all three datasets
#   DATASETS="aime2025 amc23" scripts/generate_traces.sh
#   GSM8K_LIMIT=300 scripts/generate_traces.sh
#
# One vLLM engine per GPU, each taking a stride of the questions. Rollouts per
# question come from DEFAULT_ROLLOUTS in the python script: gsm8k 4, aime2025 8,
# amc23 8.
#
# GSM8K is 8792 questions. At 4 rollouts of up to 32k tokens that is ~35k long
# generations, i.e. days — so GSM8K_LIMIT is applied by default. It is the
# *recovered* control here (per D33 it rarely derails), so a few hundred
# questions is ample; AIME/AMC are 30 and 40 questions and run in full.
set -uo pipefail
cd "$(dirname "$0")/.."

OUT="${OUT:-data/traces}"
DATASETS="${DATASETS:-aime2025 amc23 gsm8k}"
# `-` not `:-`: GSM8K_LIMIT= (explicitly empty) must mean "no limit", but `:-`
# treats empty as unset and would silently reapply the 300 default. That cost a
# run: the driver reported success having sampled 300 of 8792 questions.
GSM8K_LIMIT="${GSM8K_LIMIT-300}"
NUM_SHARDS="${NUM_SHARDS:-2}"
mkdir -p "$OUT"

log () { echo "[$(date +%H:%M:%S)] $*"; }

# Datasets are run in separate passes so GSM8K can carry a --limit the others
# must not: --limit is per-dataset in the script and would otherwise truncate
# AIME/AMC too.
for ds in $DATASETS; do
  limit_arg=()
  [[ "$ds" == "gsm8k" && -n "$GSM8K_LIMIT" ]] && limit_arg=(--limit "$GSM8K_LIMIT")

  log "=== $ds on $NUM_SHARDS GPU(s) ${limit_arg[*]} ==="
  pids=()
  for ((s = 0; s < NUM_SHARDS; s++)); do
    CUDA_VISIBLE_DEVICES="$s" uv run python scripts/generate_traces.py \
        --out "$OUT" --datasets "$ds" \
        --shard-index "$s" --num-shards "$NUM_SHARDS" \
        "${limit_arg[@]}" > "$OUT/${ds}.shard${s}.log" 2>&1 &
    pids+=($!)
    sleep 5   # stagger engine init so both do not claim memory at once
  done

  fail=0
  for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
      log "  $ds shard $i ok"
    else
      log "  $ds shard $i FAILED — see $OUT/${ds}.shard${i}.log"
      fail=1
    fi
  done
  # Merge anyway on partial failure: the shards that finished are still usable,
  # and re-running skips their completed questions.
  [[ $fail -eq 1 ]] && log "  WARNING: a shard failed; merging what exists"
done

log "=== merge ==="
# shellcheck disable=SC2086  # DATASETS is intentionally word-split
uv run python scripts/generate_traces.py --out "$OUT" --datasets $DATASETS --merge
sync
log "done: $OUT"
