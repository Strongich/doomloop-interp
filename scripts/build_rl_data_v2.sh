#!/usr/bin/env bash
# Build the Stage-2 RL prompt set, v2: web + open-r1 reasoning + our own rollouts.
#
#   scripts/build_rl_data_v2.sh                     # full build
#   N_WEB_DOCS=200 N_R1_DOCS=200 N_TRACE_DOCS=200 scripts/build_rl_data_v2.sh   # shakeout
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
set -uo pipefail   # NOT -e: see run_extract
cd "$(dirname "$0")/.."

# `datagen.extract` regularly dies with a PyGILState_Release fatal error at
# interpreter shutdown — the streaming dataset's aiohttp threads at teardown,
# AFTER the parquet and sidecar are written (noted in CLAUDE.md). Under `set -e`
# that non-zero exit aborts the whole build even though the output is complete.
# So: run it, ignore the exit code, and verify the parquet instead.
count_rows () {
  uv run python -c "
import sys, pyarrow.parquet as pq
try:
    print(pq.read_metadata(sys.argv[1]).num_rows)
except Exception:
    print(0)
" "$1" 2>/dev/null
}

run_extract () {
  local out="$1"; shift
  # Resume: a complete parquet (readable footer, >0 rows) is left alone. This is
  # an ~8h build over three sources, and redoing a finished stage after an
  # interruption is hours. A half-written file has no footer, so count_rows
  # returns 0 and it is rebuilt — the footer IS the completion marker.
  if [[ -f "$out" ]]; then
    local have
    have=$(count_rows "$out")
    if [[ "${have:-0}" -ge 1 ]]; then
      echo "  -> $out: $have rows (exists, skipping — rm it to rebuild)"
      return 0
    fi
  fi
  uv run python -m reasoning_attention.datagen.extract --output "$out" "$@"
  local rows
  rows=$(count_rows "$out")
  if [[ "${rows:-0}" -lt 1 ]]; then
    echo "ERROR: $out has no rows — extraction genuinely failed" >&2
    exit 1
  fi
  echo "  -> $out: $rows rows"
}

DATA_DIR="${DATA_DIR:-data/rl2}"
N_WEB_DOCS="${N_WEB_DOCS:-100000}"
N_R1_DOCS="${N_R1_DOCS:-100000}"      # open-r1 has ~89k; take() stops early
# Our rollout corpus is a local file with no natural cap, so the default is a
# ceiling above its 35.7k lines. Lower it to keep a shakeout short — otherwise
# N_WEB_DOCS/N_R1_DOCS shrink stages 1-2 while stage 3 still runs in full.
N_TRACE_DOCS="${N_TRACE_DOCS:-10000000}"
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
# shrink by about as much to keep activation memory flat. open-r1 keeps this:
# its p90 is ~14k, so its tail is genuinely long.
REASONING_BATCH="${REASONING_BATCH:-2}"

# OUR rollouts are a different distribution. Measured over all 35,728
# (chat-template tokens): p50 1,722,
# p90 4,608, p95 6,607, p99 12,937, max 32,768. Capping at 8,192 DROPS the 3.0%
# that run longer rather than truncating them — a truncated trace still forces
# the batch to be sized for its length, and its block positions would come from
# a prefix that is not the document the rest of the corpus represents.
#
# With the tail gone the worst case is 8,192 rather than 32,768, so the batch can
# come up 4x. It stays a constant rather than a token budget because a constant
# is what the OOM risk is actually bounded by here: 8 x 8,192 is the same token
# count the web stage already runs at batch 8 x 4,096 x2.
TRACE_MAX_TOKENS="${TRACE_MAX_TOKENS:-8192}"
TRACE_BATCH="${TRACE_BATCH:-8}"
mkdir -p "$DATA_DIR"

if [[ ! -f "$TRACES" ]]; then
  echo "ERROR: $TRACES not found."
  echo "Build it with:  uv run python scripts/build_trace_corpus.py --out $TRACES"
  exit 1
fi

echo "=== 1/5 web: $N_WEB_DOCS Ultra-FineWeb documents from $WEB_START (random positions) ==="
run_extract "$DATA_DIR/web_base.parquet" \
    --n-documents "$N_WEB_DOCS" --corpus-start "$WEB_START" \
    --batch-size "$BATCH_SIZE" --chunk-size "$CHUNK_SIZE" --seed "$SEED" \
    --source-tag ultrafineweb

echo "=== 2/5 open-r1: $N_R1_DOCS reasoning traces (random positions) ==="
run_extract "$DATA_DIR/openr1_base.parquet" \
    --n-documents "$N_R1_DOCS" --corpus-start 0 \
    --batch-size "$REASONING_BATCH" --chunk-size "$CHUNK_SIZE" --seed "$SEED" \
    --corpus open-r1/OpenThoughts-114k-math --corpus-config default \
    --corpus-split train --text-column conversations \
    --corpus-kind reasoning --source-tag openr1 \
    --max-context-tokens "$REASONING_CONTEXT"

echo "=== 3/5 our rollouts: $TRACES (BLOCK-boundary positions) ==="
run_extract "$DATA_DIR/traces_base.parquet" \
    --n-documents "$N_TRACE_DOCS" --corpus-file "$TRACES" \
    --batch-size "$TRACE_BATCH" --chunk-size "$CHUNK_SIZE" --seed "$SEED" \
    --corpus-kind trace --source-tag qwen3_rollouts \
    --position-mode block --max-context-tokens "$REASONING_CONTEXT" \
    --max-document-tokens "$TRACE_MAX_TOKENS"

set -e
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
