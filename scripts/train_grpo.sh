#!/usr/bin/env bash
# Stage 2: joint RL — AV by GRPO, AR by supervised MSE, on 2xA100 with FSDP.
#
# Adapted from natural_language_autoencoders/configs/rl.sh. Every hyperparameter
# below is theirs except the GPU layout and the batch sizes, which have to shrink:
# their defaults assume 16 GPUs (8 actor + 4 critic + 4 rollout) and a 1024-sample
# global batch on H100-80GB.
#
#   RL_PARQUET=data/rl/rl.parquet \
#   ACTOR_SFT_CKPT=... CRITIC_SL_CKPT=... RUN_DIR=runs/grpo1 \
#   scripts/train_grpo.sh
#
# The AR is trained alongside the AV because it *is* the reward model:
# reward = -MSE(AR(explanation), gold_activation). A frozen AR would hand out
# stale rewards the AV learns to game.
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"
NLA_REPO="${NLA_REPO:-$PROJECT_ROOT/natural_language_autoencoders}"

: "${RL_PARQUET:?set RL_PARQUET (build it with scripts/build_rl_data.sh)}"
: "${ACTOR_SFT_CKPT:?set ACTOR_SFT_CKPT — the AV checkpoint from the SFT stage}"
: "${CRITIC_SL_CKPT:?set CRITIC_SL_CKPT — the AR checkpoint from the SFT stage}"
: "${RUN_DIR:?set RUN_DIR for outputs}"
INSTRUCT_MODEL="${INSTRUCT_MODEL:-Qwen/Qwen3-1.7B}"

# The RL stack lives in its own env — see scripts/setup_rl_stack.sh for why.
RL_PYTHON="${RL_PYTHON:-$PROJECT_ROOT/.venv-rl/bin/python}"
if [[ ! -x "$RL_PYTHON" ]]; then
  echo "ERROR: $RL_PYTHON not found. Run scripts/setup_rl_stack.sh first," >&2
  echo "or point RL_PYTHON at an env that has miles + sglang + nla." >&2
  exit 1
fi

# --- GPU layout: 2xA100. Theirs is 8+4+4; three roles must share two devices. ---
# Ray hangs on placement if the roles ask for more GPUs than exist, so colocation
# is not optional here.
ACTOR_NODES="${ACTOR_NODES:-1}"
ACTOR_GPUS="${ACTOR_GPUS:-2}"
CRITIC_NODES="${CRITIC_NODES:-1}"
CRITIC_GPUS="${CRITIC_GPUS:-2}"
ROLLOUT_GPUS="${ROLLOUT_GPUS:-2}"

# --- Batch: theirs was 128 prompts x 8 samples = 1024 on 16 GPUs. ---
# GRPO group size stays at 8 — it is the advantage baseline, not a throughput
# knob, and shrinking it raises advantage variance.
ROLLOUT_BATCH="${ROLLOUT_BATCH:-16}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-$((ROLLOUT_BATCH * SAMPLES_PER_PROMPT))}"
ACTOR_MICRO="${ACTOR_MICRO:-4}"

# --- Their hyperparameters, copied. ---
# Production parity LRs at 1.41e-5 = the 1e-5 scan winner scaled by sqrt(2) for
# the 512->1024 batch step. We are far below 1024, so sqrt-scale back down.
LR_SCALE="$($RL_PYTHON -c "import math;print(f'{math.sqrt($GLOBAL_BATCH/1024):.4f}')")"
ACTOR_LR="${ACTOR_LR:-$($RL_PYTHON -c "print(f'{1.41e-5 * $LR_SCALE:.3e}')")}"
CRITIC_LR="${CRITIC_LR:-$ACTOR_LR}"   # parity, as they ran for most of training
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
MAX_RESPONSE_LEN="${MAX_RESPONSE_LEN:-150}"
MAX_CONTEXT_LEN="${MAX_CONTEXT_LEN:-300}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"

# --kl-coef is a no-op for GRPO (get_grpo_returns discards the kl tensor);
# --use-kl-loss is the path that actually adds KL to the policy loss. It is
# store_true, so gate on the env var to allow turning it off entirely.
if "$RL_PYTHON" -c "import sys;sys.exit(0 if float('$KL_LOSS_COEF') != 0 else 1)"; then
  KL_FLAGS=(--use-kl-loss --kl-loss-coef "$KL_LOSS_COEF")
else
  KL_FLAGS=()
fi

# Per-step ~1 GB embedding dump. /tmp is overlayfs (disk, ~1.5s/step); /dev/shm is
# tmpfs. Needs >= 8g of shm.
export NLA_EMBED_DUMP_DIR="${NLA_EMBED_DUMP_DIR:-/dev/shm/nla}"
mkdir -p "$NLA_EMBED_DUMP_DIR"
SHM_KB="$(df -k /dev/shm | awk 'NR==2{print $2}')"
if (( SHM_KB < 8 * 1024 * 1024 )); then
  echo "WARNING: /dev/shm is $((SHM_KB / 1024)) MiB; the reference wants >= 8 GiB." >&2
  echo "  Re-run the container with --shm-size=8g, or point NLA_EMBED_DUMP_DIR at disk." >&2
fi

cat <<EOM
==============================================================
 Stage 2: GRPO (AV) + MSE (AR)
   data          $RL_PARQUET
   actor ckpt    $ACTOR_SFT_CKPT
   critic ckpt   $CRITIC_SL_CKPT
   gpus          actor $ACTOR_GPUS / critic $CRITIC_GPUS / rollout $ROLLOUT_GPUS
   batch         $ROLLOUT_BATCH prompts x $SAMPLES_PER_PROMPT samples = $GLOBAL_BATCH
   lr            actor $ACTOR_LR / critic $CRITIC_LR  (1.41e-5 x sqrt($GLOBAL_BATCH/1024))
   kl coef       $KL_LOSS_COEF
   response cap  $MAX_RESPONSE_LEN
==============================================================
EOM

cd "$NLA_REPO"
exec "$RL_PYTHON" train.py \
    --train-backend "${TRAIN_BACKEND:-fsdp}" \
    --custom-actor-cls-path "${ACTOR_CLS:-nla.train_actor.NLAFSDPActor}" \
    --loss-type policy_loss \
    --advantage-estimator grpo \
    --force-use-critic \
    --n-samples-per-prompt "$SAMPLES_PER_PROMPT" \
    --rollout-function-path miles.rollout.sglang_rollout.generate_rollout \
    --custom-generate-function-path nla.rollout.nla_generate.generate \
    --custom-rm-path nla.reward.nla_rm \
    --data-source-path nla.data_source.NLADataSource \
    --prompt-data "$RL_PARQUET" \
    --input-key prompt \
    --hf-checkpoint "$INSTRUCT_MODEL" \
    --ref-load "$ACTOR_SFT_CKPT" \
    --load "$ACTOR_SFT_CKPT" \
    --nla-sidecar-source "$ACTOR_SFT_CKPT" \
    --save "$RUN_DIR/actor" \
    --critic-load "$CRITIC_SL_CKPT" \
    --critic-save "$RUN_DIR/critic" \
    --critic-lr "$CRITIC_LR" \
    --actor-num-nodes "$ACTOR_NODES" \
    --actor-num-gpus-per-node "$ACTOR_GPUS" \
    --critic-num-nodes "$CRITIC_NODES" \
    --critic-num-gpus-per-node "$CRITIC_GPUS" \
    --rollout-num-gpus "$ROLLOUT_GPUS" \
    --rollout-max-response-len "$MAX_RESPONSE_LEN" \
    --rollout-max-context-len "$MAX_CONTEXT_LEN" \
    `# REQUIRED. The radix cache keys on token IDs, but we inject a different` \
    `# activation vector at the same marker token every time — a cache hit would` \
    `# silently return another activation's output. Do NOT remove to "optimize".` \
    --sglang-disable-radix-cache \
    --sglang-context-length "$MAX_CONTEXT_LEN" \
    --router-history-backend none \
    `# cache_aware routing builds a prefix tree holding request bodies; with` \
    `# ~6-12MB input_embeds per request that tree IS the memory leak.` \
    --router-policy round_robin \
    --router-disable-circuit-breaker \
    --router-retry-max-backoff-ms 500 --router-retry-max-retries 2 \
    --rollout-batch-size "$ROLLOUT_BATCH" \
    --global-batch-size "$GLOBAL_BATCH" \
    --micro-batch-size "$ACTOR_MICRO" \
    --lr "$ACTOR_LR" --lr-decay-style constant \
    --attn-implementation "${ATTN_IMPL:-flash_attention_2}" \
    `# NO --gradient-checkpointing: it deadlocks NCCL in update_weights() —` \
    `# FSDP's full-param gather changes, the broadcast hangs, watchdog SIGABRTs.` \
    "${KL_FLAGS[@]}" \
    --save-interval "$SAVE_INTERVAL" \
    --loss-mask-type "${LOSS_MASK_TYPE:-qwen}" \
    "$@"
