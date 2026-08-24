#!/usr/bin/env bash
# Build the Stage-1 (SFT warm-start) dataset, all four stages.
#
#   scripts/build_warmstart_data.sh                    # full 100k docs
#   N_DOCS=200 scripts/build_warmstart_data.sh         # quick shakeout
#   DATA_DIR=/workspace/data/warmstart scripts/build_warmstart_data.sh
#
# Everything lands under $DATA_DIR. On the cluster point that at the PVC
# (/workspace/...) — the container filesystem is ephemeral and these files are
# GBs that cost real API money to regenerate.
#
# Layout produced:
#   $DATA_DIR/base.parquet                 500k rows, ~2.1 GB   (+ .nla_meta.yaml)
#   $DATA_DIR/halves/av_half.parquet       250k rows, ~1.0 GB
#   $DATA_DIR/halves/ar_half.parquet       250k rows, ~1.0 GB
#   $DATA_DIR/av_explained.parquet         + .chunks/  <- resumable API output
#   $DATA_DIR/ar_explained.parquet         + .chunks/
#   $DATA_DIR/av_sft.parquet               250k rows, ~1.35 GB  <- trainer input
#   $DATA_DIR/ar_sft.parquet               250k rows, ~1.28 GB  <- trainer input
#   ~9 GB total with the intermediates; ~2.6 GB if you delete them afterwards.
#
# Stage 3 (explain) is the only one that costs money and the only one that is
# resumable: it writes per-chunk parquets and skips completed chunks on restart.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data/warmstart}"
N_DOCS="${N_DOCS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-256}"
EXPLAIN_CHUNK="${EXPLAIN_CHUNK:-512}"
SEED="${SEED:-42}"
mkdir -p "$DATA_DIR"

echo "=== 1/4 extract: $N_DOCS Ultra-FineWeb docs x 5 positions ==="
if [[ -f "$DATA_DIR/base.parquet" ]]; then
  echo "  exists, skipping (delete it to re-extract)"
else
  uv run python -m reasoning_attention.datagen.extract \
      --output "$DATA_DIR/base.parquet" \
      --n-documents "$N_DOCS" --batch-size "$BATCH_SIZE" \
      --chunk-size "$CHUNK_SIZE" --seed "$SEED"
fi

echo "=== 2/4 split: disjoint AV / AR halves, partitioned by document ==="
uv run python -m reasoning_attention.datagen.split \
    --base "$DATA_DIR/base.parquet" --output-dir "$DATA_DIR/halves" --seed "$SEED"

echo "=== 3/4 explain: label both halves (COSTS API CREDIT, resumable) ==="
for half in av ar; do
  uv run python -m reasoning_attention.datagen.explain \
      --input "$DATA_DIR/halves/${half}_half.parquet" \
      --output "$DATA_DIR/${half}_explained.parquet" \
      --chunk-size "$EXPLAIN_CHUNK"
done

echo "=== 4/4 build: trainer-ready parquets ==="
uv run python -m reasoning_attention.datagen.build \
    --input "$DATA_DIR/av_explained.parquet" --output "$DATA_DIR/av_sft.parquet" --stage av_sft
uv run python -m reasoning_attention.datagen.build \
    --input "$DATA_DIR/ar_explained.parquet" --output "$DATA_DIR/ar_sft.parquet" --stage ar_sft

echo
du -sh "$DATA_DIR"
echo "train with:  DATA_DIR=$DATA_DIR scripts/train_sft.sh"
