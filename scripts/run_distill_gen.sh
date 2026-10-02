#!/usr/bin/env bash
# Teacher generation, LOCKED PROTOCOL v2 §L3: Qwen3-1.7B, 4 x base + 4 x N@a1.0d256 per
# question, 32,768-token cap, one single-GPU streaming engine per question shard (pp=1,
# so D70 does not apply). Shards are a round-robin split of data/distill/train.jsonl
# ordered by (dataset, level), so every shard carries the same difficulty mix.
#
#   SHARDS=6 bash scripts/run_distill_gen.sh
set -euo pipefail
cd "$(dirname "$0")/.."
SHARDS=${SHARDS:-6}
BATCH=${BATCH:-512}
OUT=${OUT:-data/distill/gen}
LOGDIR=${LOGDIR:-/workspace/distill_logs}
mkdir -p "$OUT" "$LOGDIR"
uv run python - "$SHARDS" "$OUT" <<'EOF'
import json, sys
n, out = int(sys.argv[1]), sys.argv[2]
rows = [json.loads(x) for x in open("data/distill/train.jsonl")]
rows.sort(key=lambda r: (r["dataset"], r.get("level", 0), r["question_id"]))
for k in range(n):
    p = f"{out}/shard{k}.jsonl"
    txt = "".join(json.dumps(r) + "\n" for r in rows[k::n])
    try:
        if open(p).read() != txt:
            raise SystemExit(f"{p} exists with different content")
    except FileNotFoundError:
        open(p, "w").write(txt)
    print(p, len(rows[k::n]))
EOF
for k in $(seq 0 $((SHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$k nohup uv run python scripts/reasoning_policy_vllm.py \
    --cohort "$OUT/shard$k.jsonl" --outdir "$OUT/shard$k" \
    --policies base N@a1.0d256 --seeds 4 --max-new-tokens 32768 \
    --batch "$BATCH" --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 \
    --checkpoint-every 500 --stream > "$LOGDIR/gen_shard$k.log" 2>&1 &
  echo "shard $k -> GPU $k pid $!"
done
