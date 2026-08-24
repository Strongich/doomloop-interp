#!/usr/bin/env bash
# Launch the Stage-1 SFT warm-start for one or both halves of the NLA.
#
#   scripts/train_sft.sh                      # both halves, default data dir
#   scripts/train_sft.sh av                   # AV only
#   scripts/train_sft.sh ar                   # AR only
#   DATA_DIR=/path/to/parquets scripts/train_sft.sh
#   LIMIT=64 EPOCHS=1 scripts/train_sft.sh av # quick shakeout on 64 rows
#
# Hyperparameters are copied from the reference repo's Qwen2.5-7B case study
# (natural_language_autoencoders/configs/TRAINING_NOTES.md) — EXCEPT batch size,
# which is ours: theirs is 256 on 2xH100-80GB. Their own caveat stands: "these are
# the settings we used, not settings we claim are optimal."
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_DIR="${DATA_DIR:-data/warmstart}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints}"
AV_DATA="${AV_DATA:-$DATA_DIR/av_sft.parquet}"
AR_DATA="${AR_DATA:-$DATA_DIR/ar_sft.parquet}"

# --- Their Qwen2.5-7B settings (configs/TRAINING_NOTES.md), except batch size.
# --- Their caveat applies: "settings we used, not settings we claim are optimal."
LR="${LR:-2e-5}"                    # theirs, both AV and AR
MIN_LR="${MIN_LR:-2e-6}"            # cosine floor = LR/10
WARMUP_RATIO="${WARMUP_RATIO:-0.05}"  # theirs: 50 warmup iters / 1000 rollouts
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
# Theirs: flash_attention_2 for the AV, sdpa for the AR. FA2 needs the flash-attn
# package; unset falls back to whatever transformers picks.
AV_ATTN="${AV_ATTN:-}"
AR_ATTN="${AR_ATTN:-sdpa}"
# Theirs was 150 for Qwen2.5-7B (d_model 3584). Ours is d_model 2048, where
# sqrt_d_model = 45.3, so 150 is their absolute value, not their ratio (~2.5x
# sqrt(d) would be ~113 here). Unset keeps NLAConfig's sqrt_d_model.
INJECTION_SCALE="${INJECTION_SCALE:-}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"       # effective batch = BATCH_SIZE * GRAD_ACCUM
EPOCHS="${EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-1024}"
LORA_R="${LORA_R:-16}"
LOG_EVERY="${LOG_EVERY:-10}"
LIMIT="${LIMIT:-}"                  # empty = all rows
EXTRA_ARGS="${EXTRA_ARGS:-}"        # e.g. --full-finetune --gradient-checkpointing

STAGES=("${@:-av ar}")
read -r -a STAGES <<< "${STAGES[*]}"

run_stage () {
  local stage="$1" data="$2"
  if [[ ! -f "$data" ]]; then
    echo "ERROR: $data not found." >&2
    echo "Build it first, e.g.:" >&2
    echo "  uv run python -m reasoning_attention.datagen.extract --output $DATA_DIR/base.parquet" >&2
    echo "  uv run python -m reasoning_attention.datagen.split   --base $DATA_DIR/base.parquet --output-dir $DATA_DIR/halves" >&2
    echo "  uv run python -m reasoning_attention.datagen.explain --input $DATA_DIR/halves/${stage}_half.parquet --output $DATA_DIR/${stage}_explained.parquet" >&2
    echo "  uv run python -m reasoning_attention.datagen.build   --input $DATA_DIR/${stage}_explained.parquet --output $data --stage ${stage}_sft" >&2
    exit 1
  fi

  echo "=============================================================="
  echo " ${stage^^} SFT  |  data=$data"
  echo " lr=$LR->$MIN_LR cosine batch=$BATCH_SIZE x $GRAD_ACCUM epochs=$EPOCHS lora_r=$LORA_R"
  echo "=============================================================="

  local limit_arg=()
  [[ -n "$LIMIT" ]] && limit_arg=(--limit "$LIMIT")

  local attn="$AV_ATTN"
  [[ "$stage" == "ar" ]] && attn="$AR_ATTN"
  local opt_args=()
  [[ -n "$attn" ]] && opt_args+=(--attn-implementation "$attn")
  [[ -n "$INJECTION_SCALE" ]] && opt_args+=(--injection-scale "$INJECTION_SCALE")

  # shellcheck disable=SC2086  # EXTRA_ARGS is intentionally word-split
  uv run python -m reasoning_attention.training.sft \
    --stage "$stage" \
    --data "$data" \
    --output-dir "$OUTPUT_DIR" \
    --learning-rate "$LR" \
    --min-lr "$MIN_LR" \
    --warmup-ratio "$WARMUP_RATIO" \
    --save-interval "$SAVE_INTERVAL" \
    "${opt_args[@]}" \
    --batch-size "$BATCH_SIZE" \
    --grad-accum "$GRAD_ACCUM" \
    --epochs "$EPOCHS" \
    --max-length "$MAX_LENGTH" \
    --lora-r "$LORA_R" \
    --log-every "$LOG_EVERY" \
    "${limit_arg[@]}" \
    $EXTRA_ARGS
}

for stage in "${STAGES[@]}"; do
  case "$stage" in
    av) run_stage av "$AV_DATA" ;;
    ar) run_stage ar "$AR_DATA" ;;
    *)  echo "ERROR: unknown stage '$stage' (want av or ar)" >&2; exit 1 ;;
  esac
done

echo "done. checkpoints under $OUTPUT_DIR/"
