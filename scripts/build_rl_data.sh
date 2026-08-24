#!/usr/bin/env bash
# Build the Stage-2 RL prompt set: 200k web + 200k chat activations.
#
#   scripts/build_rl_data.sh                       # full 40k + 40k documents
#   N_DOCS=200 scripts/build_rl_data.sh            # quick shakeout
#   DATA_DIR=/mnt/data/rl scripts/build_rl_data.sh
#
# 40k documents x 5 random positions = 200k activations per source, 400k total —
# a quarter of the paper's 500k-per-source, deliberately, since this run is
# smaller. No API spend: RL needs no summaries, because the AV generates the
# explanation during rollout and the AR scores it.
#
# The web slice starts at document 100000 so RL activations come from documents
# the SFT warm-start never saw.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data/rl}"
N_DOCS="${N_DOCS:-40000}"
WEB_START="${WEB_START:-100000}"
CHAT_START="${CHAT_START:-0}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
SEED="${SEED:-43}"
mkdir -p "$DATA_DIR"

echo "=== 1/4 web half: $N_DOCS Ultra-FineWeb documents from $WEB_START ==="
uv run python -m reasoning_attention.datagen.extract \
    --output "$DATA_DIR/web_base.parquet" \
    --n-documents "$N_DOCS" --corpus-start "$WEB_START" \
    --batch-size "$BATCH_SIZE" --chunk-size "$CHUNK_SIZE" --seed "$SEED" \
    --source-tag ultrafineweb

echo "=== 2/4 chat half: $N_DOCS WildChat conversations from $CHAT_START ==="
# --corpus-kind chat renders each conversation through the target model's chat
# template before sampling positions, so the activations come from dialogue shaped
# the way the model actually sees it.
uv run python -m reasoning_attention.datagen.extract \
    --output "$DATA_DIR/chat_base.parquet" \
    --n-documents "$N_DOCS" --corpus-start "$CHAT_START" \
    --batch-size "$BATCH_SIZE" --chunk-size "$CHUNK_SIZE" --seed "$SEED" \
    --corpus allenai/WildChat-1M --corpus-config default --corpus-split train \
    --text-column conversation --corpus-kind chat --source-tag wildchat

echo "=== 3/4 merge + shuffle ==="
# Shuffled so training does not see 200k web rows and then 200k chat rows — with
# a constant-LR RL schedule that ordering is an unintended curriculum.
uv run python -m reasoning_attention.datagen.merge \
    --inputs "$DATA_DIR/web_base.parquet" "$DATA_DIR/chat_base.parquet" \
    --output "$DATA_DIR/rl_base.parquet" --seed "$SEED"

echo "=== 4/4 build the RL parquet ==="
uv run python -m reasoning_attention.datagen.build \
    --input "$DATA_DIR/rl_base.parquet" --output "$DATA_DIR/rl.parquet" --stage rl

echo
echo "RL_PARQUET=$DATA_DIR/rl.parquet"
