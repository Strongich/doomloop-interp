#!/usr/bin/env bash
# Build the Stage-2 RL prompt set, v2: web + open-r1 reasoning + our own rollouts.
#
#   scripts/build_rl_data_v2.sh                     # full build
#   N_WEB_DOCS=200 N_R1_DOCS=200 scripts/build_rl_data_v2.sh   # shakeout
#   DATA_DIR=/workspace/data/rl2 scripts/build_rl_data_v2.sh
#
# Replaces WildChat with reasoning text, because the NLA measured at 0.905 cosine
# on web and 0.791 at the block boundaries this study probes — two separate gaps,
# domain and position (scripts/fve_by_domain.py). WildChat closes neither.
#
#   web       100k Ultra-FineWeb docs x 5 RANDOM positions  = 500k
#   openr1    ~89k open-r1 traces      x 5 RANDOM positions  ~ 445k
#   traces    all of our own rollouts  x 5 BLOCK positions   ~ 178k
#
# The two reasoning sources are sampled differently on purpose. open-r1 at random
# positions covers reasoning text broadly; our own rollouts at block boundaries
# target the exact positions where reconstruction is worst and where every
# finding is measured. Sampling both uniformly would fix the domain gap and leave
# the positional one.
#
# Still no API spend: RL needs no summaries — the AV generates the explanation
# during rollout and the AR scores it.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data/rl2}"
N_WEB_DOCS="${N_WEB_DOCS:-100000}"
N_R1_DOCS="${N_R1_DOCS:-100000}"      # open-r1 has ~89k; take() stops early
TRACES="${TRACES:-data/traces_all.jsonl}"
# Web documents the SFT warm-start never saw. The warm-start used 0..99,999.
WEB_START="${WEB_START:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
SEED="${SEED:-44}"
# Reasoning traces run far longer than web snippets: open-r1's median is ~4.8k
# tokens and p90 ~14k, so the 4096 default would cut most of them and confine
# positions to the opening. 32768 matches the generation budget.
REASONING_CONTEXT="${REASONING_CONTEXT:-32768}"
# A 32k-token document is 8x the web default, so the per-forward batch has to
# shrink by about as much to keep activation memory flat.
REASONING_BATCH="${REASONING_BATCH:-2}"
mkdir -p "$DATA_DIR"

if [[ ! -f "$TRACES" ]]; then
  echo "ERROR: $TRACES not found."
  echo "Build it with:  uv run python scripts/build_trace_corpus.py --out $TRACES"
  exit 1
fi

echo "=== 1/5 web: $N_WEB_DOCS Ultra-FineWeb documents from $WEB_START (random positions) ==="
uv run python -m reasoning_attention.datagen.extract \
    --output "$DATA_DIR/web_base.parquet" \
    --n-documents "$N_WEB_DOCS" --corpus-start "$WEB_START" \
    --batch-size "$BATCH_SIZE" --chunk-size "$CHUNK_SIZE" --seed "$SEED" \
    --source-tag ultrafineweb

echo "=== 2/5 open-r1: $N_R1_DOCS reasoning traces (random positions) ==="
uv run python -m reasoning_attention.datagen.extract \
    --output "$DATA_DIR/openr1_base.parquet" \
    --n-documents "$N_R1_DOCS" --corpus-start 0 \
    --batch-size "$REASONING_BATCH" --chunk-size "$CHUNK_SIZE" --seed "$SEED" \
    --corpus open-r1/OpenThoughts-114k-math --corpus-config default \
    --corpus-split train --text-column conversations \
    --corpus-kind reasoning --source-tag openr1 \
    --max-context-tokens "$REASONING_CONTEXT"

echo "=== 3/5 our rollouts: $TRACES (BLOCK-boundary positions) ==="
uv run python -m reasoning_attention.datagen.extract \
    --output "$DATA_DIR/traces_base.parquet" \
    --n-documents 10000000 --corpus-file "$TRACES" \
    --batch-size "$REASONING_BATCH" --chunk-size "$CHUNK_SIZE" --seed "$SEED" \
    --corpus-kind trace --source-tag qwen3_rollouts \
    --position-mode block --max-context-tokens "$REASONING_CONTEXT"

echo "=== 4/5 merge + shuffle ==="
# Shuffled so training does not see one source then the next — with a constant-LR
# RL schedule that ordering is an unintended curriculum.
uv run python -m reasoning_attention.datagen.merge \
    --inputs "$DATA_DIR/web_base.parquet" "$DATA_DIR/openr1_base.parquet" \
             "$DATA_DIR/traces_base.parquet" \
    --output "$DATA_DIR/rl_base.parquet" --seed "$SEED"

echo "=== 5/5 build the RL parquet ==="
uv run python -m reasoning_attention.datagen.build \
    --input "$DATA_DIR/rl_base.parquet" --output "$DATA_DIR/rl.parquet" --stage rl

echo
uv run python - "$DATA_DIR/rl.parquet" <<'PY'
import sys, collections, pyarrow.parquet as pq
t = pq.read_table(sys.argv[1], columns=["source"])
c = collections.Counter(t.column("source").to_pylist())
print(f"rows: {t.num_rows:,}")
for k, v in c.most_common():
    print(f"  {k:<16} {v:>9,}  ({100 * v / t.num_rows:.1f}%)")
PY
echo "RL_PARQUET=$DATA_DIR/rl.parquet"
