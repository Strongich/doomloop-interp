#!/usr/bin/env bash
# PROTOCOL v3 top-up: 12 more base + 12 more N@a1.0d256 rollouts (seeds 4..15) per hard
# question, same engine/sampler/cap as v2 §L3. Shards start as GPUs become free (the v2
# evals may still hold some), one single-GPU engine each.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT=data/distill_v3/gen; LOGDIR=/workspace/distill_logs; mkdir -p "$OUT"
uv run python - <<'PY'
import json
rows=[json.loads(x) for x in open("data/distill_v3/hard.jsonl")]
rows.sort(key=lambda r:(r["dataset"], r.get("level",0), r["question_id"]))
for k in range(6):
    open(f"data/distill_v3/gen/shard{k}.jsonl","w").write("".join(json.dumps(r)+"\n" for r in rows[k::6]))
PY
claimed=""
for k in 0 1 2 3 4 5; do
  while :; do
    g=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | \
        awk -F', ' -v c=" $claimed " '$2 < 1000 && index(c, " "$1" ") == 0 {print $1; exit}')
    [ -n "$g" ] && break; sleep 60
  done
  claimed="$claimed $g"
  CUDA_VISIBLE_DEVICES=$g nohup uv run python scripts/reasoning_policy_vllm.py \
    --cohort "$OUT/shard$k.jsonl" --outdir "$OUT/shard$k" \
    --policies base N@a1.0d256 --seed-start 4 --seeds 16 --max-new-tokens 32768 \
    --batch 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
    --checkpoint-every 500 --stream > "$LOGDIR/v3gen_shard$k.log" 2>&1 &
  echo "$(date +%H:%M) v3 shard $k -> GPU $g pid $!"
  sleep 90
done
