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

# --- Where the actor's INITIAL WEIGHTS come from. ---
# Their rl.sh passes --hf-checkpoint $INSTRUCT_MODEL (base) and relies on --load
# $ACTOR_SFT_CKPT to overlay the SFT weights, because their SFT ran under miles
# and emits DCP checkpoints ("DCP iter dir from actor_sft.sh, e.g. .../iter_0002000").
# OUR SFT is our own loop and writes a plain HF directory, and the two miles
# loaders are NOT interchangeable:
#   --ref-load  -> os.path.isdir() then from_pretrained()  = HF, works
#   --load      -> needs latest_checkpointed_iteration.txt + iter_NNNNNNN/{model,
#                  optimizer,lr_scheduler} DCP dirs; otherwise logs
#                  "[FSDP] No tracker file at ...; skipping load." and CONTINUES
# So --load silently did nothing and the actor trained from base Qwen3-1.7B while
# the ref model was our AV — backwards, and invisible except as a ~5.4 nat gap
# between rollout/log_probs and rollout/ref_log_probs at step 0.
#
# Since av_sft IS a complete HF Qwen3ForCausalLM checkpoint, point --hf-checkpoint
# at it: the actor initializes from it, and sglang loads it as model_path too
# (rather than base) so the pre-first-sync engine matches the policy.
# --load is then free to mean what miles intends: resume an interrupted RL run.
RESUME_FROM="${RESUME_FROM:-$RUN_DIR/actor}"

# The RL stack lives in its own env — see scripts/setup_rl_stack.sh for why.
RL_PYTHON="${RL_PYTHON:-$PROJECT_ROOT/.venv-rl/bin/python}"
if [[ ! -x "$RL_PYTHON" ]]; then
  echo "ERROR: $RL_PYTHON not found. Run scripts/setup_rl_stack.sh first," >&2
  echo "or point RL_PYTHON at an env that has miles + sglang + nla." >&2
  exit 1
fi

# Miles' entry point is train.py at its REPO root, not an installed console
# script and not a module — their configs/rl.sh runs `python train.py` with the
# miles checkout as cwd. setup_rl_stack.sh clones it to .rl-src/miles and
# installs it editable, so the source tree is the right place to find it.
MILES_SRC="${MILES_SRC:-$PROJECT_ROOT/.rl-src/miles}"
MILES_TRAIN="$MILES_SRC/train.py"
if [[ ! -f "$MILES_TRAIN" ]]; then
  echo "ERROR: $MILES_TRAIN not found. Run scripts/setup_rl_stack.sh, or point" >&2
  echo "MILES_SRC at your miles checkout." >&2
  exit 1
fi

# --- GPU layout: 2xA100. Theirs is 8 actor + 4 critic + 4 rollout = 16. ---
# miles sizes ONE placement group for every role up front
# (miles/ray/placement_group.py:create_placement_groups) and Ray then blocks
# forever if the total exceeds the cluster — no error, no timeout, just a live
# process at idle CPU with no worker actors and empty nvidia-smi. Observed with
# 2/2/2: the non-colocate branch asks for actor + rollout + critic = 6 GPUs.
#
# The two branches that matter here:
#   default:     actor + rollout + critic   -> 2+2+2 = 6  (hangs)
#   --colocate:  actor + critic             -> 2+0+2 = 4  (still hangs)
# --colocate folds the sglang engines onto the ACTOR's GPUs and ignores
# --rollout-num-gpus, but the critic always gets its own. So the only layout
# that fits two devices is 1 actor (sharing with rollout) + 1 critic.
COLOCATE="${COLOCATE:-1}"
ACTOR_NODES="${ACTOR_NODES:-1}"
ACTOR_GPUS="${ACTOR_GPUS:-1}"
CRITIC_NODES="${CRITIC_NODES:-1}"
CRITIC_GPUS="${CRITIC_GPUS:-1}"
# Ignored under --colocate (miles sets it to ACTOR_GPUS * ACTOR_NODES); kept so
# COLOCATE=0 on a bigger box still works.
ROLLOUT_GPUS="${ROLLOUT_GPUS:-1}"
# Defaults to 8 in miles. Its own help: "If you are going to use less than 8 gpus
# per node under colocate mode, you should set this number."
NUM_GPUS_PER_NODE="${NUM_GPUS_PER_NODE:-2}"
COLOCATE_FLAGS=()
if [[ "$COLOCATE" == "1" ]]; then
  # --colocate also forces --offload, so the actor is swapped to CPU while sglang
  # generates and back for the training pass. That is the cost of two devices.
  COLOCATE_FLAGS=(--colocate --num-gpus-per-node "$NUM_GPUS_PER_NODE")
fi

# The AV was SFT'd with enable_thinking=False, and nla_generate.py omits that
# kwarg, so Qwen3 would default it to True and the policy would run
# off-distribution (D43). natural_language_autoencoders/ is gitignored and cloned
# per machine, so this edit is not carried by our history — verify it is present.
NLA_GEN_PY="$NLA_REPO/nla/rollout/nla_generate.py"
if [[ -f "$NLA_GEN_PY" ]] && ! grep -q "enable_thinking=False" "$NLA_GEN_PY"; then
  echo "ERROR: $NLA_GEN_PY does not pass enable_thinking=False." >&2
  echo "Qwen3 would prefill its own <think> block and every rollout would fail." >&2
  echo "Apply it with:" >&2
  echo "  $RL_PYTHON scripts/patch_nla_nonthinking.py --nla-repo $NLA_REPO" >&2
  exit 1
fi

# --- CUDA forward-compat guard. ---
# miles injects a hardcoded LD_LIBRARY_PATH into every sglang engine actor's Ray
# runtime_env, putting /usr/local/cuda/compat FIRST ("so a forward-compat
# libcuda.so wins if present", miles/ray/rollout.py). Compat libs are for a
# driver OLDER than the toolkit. When the driver is NEWER, that libcuda shadows
# the real one and CUDA init fails inside the engine with
#   Error 803: system has unsupported display driver / cuda driver combination
# surfacing as sglang's "No accelerator (CUDA, XPU, HPU, NPU) is available."
# while nvidia-smi and the driver process are perfectly healthy. We cannot
# override it — runtime_env env_vars win over ours — so check it here.
COMPAT_DIR="$(readlink -f /usr/local/cuda 2>/dev/null || true)/compat"
if [[ -d "$COMPAT_DIR" ]]; then
  DRV="$(cat /sys/module/nvidia/version 2>/dev/null || echo unknown)"
  COMPAT_LIB="$(ls "$COMPAT_DIR"/libcuda.so.*.* 2>/dev/null | head -1)"
  COMPAT_VER="${COMPAT_LIB##*/libcuda.so.}"
  if [[ -n "$COMPAT_VER" && "$DRV" != "unknown" ]] &&
     [[ "$(printf '%s\n' "$COMPAT_VER" "$DRV" | sort -V | head -1)" == "$COMPAT_VER" ]] &&
     [[ "$COMPAT_VER" != "$DRV" ]]; then
    echo "ERROR: $COMPAT_DIR holds libcuda $COMPAT_VER but the driver is $DRV." >&2
    echo "The compat lib is OLDER than the driver, so it will break CUDA init in" >&2
    echo "the sglang engine actors (error 803). Disable it:" >&2
    echo "  mv $COMPAT_DIR $COMPAT_DIR.disabled-mismatched-driver" >&2
    exit 1
  fi
fi

# Guard the arithmetic above rather than rediscovering the hang. Mirrors
# create_placement_groups: colocate drops the rollout term, nothing drops critic.
VISIBLE_GPUS="$("$RL_PYTHON" -c "import torch;print(torch.cuda.device_count())")"
if [[ "$COLOCATE" == "1" ]]; then
  WANT_GPUS=$((ACTOR_NODES * ACTOR_GPUS + CRITIC_NODES * CRITIC_GPUS))
else
  WANT_GPUS=$((ACTOR_NODES * ACTOR_GPUS + ROLLOUT_GPUS + CRITIC_NODES * CRITIC_GPUS))
fi
if (( WANT_GPUS > VISIBLE_GPUS )); then
  echo "ERROR: this layout needs $WANT_GPUS GPUs but only $VISIBLE_GPUS are visible." >&2
  echo "Ray would wait on the placement group forever instead of failing." >&2
  echo "  actor  ${ACTOR_NODES}x${ACTOR_GPUS}" >&2
  echo "  critic ${CRITIC_NODES}x${CRITIC_GPUS}" >&2
  [[ "$COLOCATE" == "1" ]] && echo "  rollout colocated with actor" >&2 \
                          || echo "  rollout ${ROLLOUT_GPUS}" >&2
  exit 1
fi

# --- Batch. ---
# ROLLOUT_BATCH is NOT a memory knob. It is the number of prompts per RL step;
# GLOBAL_BATCH samples are then accumulated over ceil(GLOBAL_BATCH/(ACTOR_MICRO *
# n_gpus)) micro-steps, so raising it costs wall-clock per step, not VRAM. Their
# own runs prove the separation: rollout_batch 64 (the 2-GPU LR scan) and 128
# (the 2x8-GPU production run) both at micro-batch 16. They scaled it with GPU
# COUNT, not with memory, and rescaled LR by sqrt(batch) to match.
#
# 64 is therefore the right target for us: it is their 2-GPU config, and we have
# 2 GPUs. Keep GLOBAL_BATCH an exact multiple of ACTOR_MICRO * n_gpus —
# TRAINING_NOTES measured a non-integer grad_accum (5.33) at 479s/step vs ~9s.
# 512 / (16 * 2) = 16 exactly.
#
# GRPO group size stays at 8 — it is the advantage baseline, not a throughput
# knob, and shrinking it raises advantage variance.
ROLLOUT_BATCH="${ROLLOUT_BATCH:-64}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-8}"
GLOBAL_BATCH="${GLOBAL_BATCH:-$((ROLLOUT_BATCH * SAMPLES_PER_PROMPT))}"
# 16, not rl.sh's ${ACTOR_MICRO:-4} default — TRAINING_NOTES' RL section says
# "m16 is fine with resp_len capped at 150", and their config sweep measured m16
# as the fastest point (9.05s vs 12.83s at m64+ckpt): 8 microbatches of fwd+bwd
# beat 2 of fwd+recompute+bwd, because the extra FSDP gathers cost less than the
# recompute they save. Bigger is NOT automatically faster here. Their ceiling was
# a 7B at d_model 3584; our 1.7B at 2048 has headroom, but the FLOP-equivalence
# argument is about ratios, so start at their measured optimum.
ACTOR_MICRO="${ACTOR_MICRO:-16}"

# NLAFSDPActor refuses to start unless
#   rollout_batch_size * n_samples_per_prompt == global_batch_size
# (bypass: NLA_I_KNOW_WHAT_IM_DOING=1). Their header explains why it matters: the
# FSDP path forces ONE optimizer step per rollout, so a mismatch does not change
# the step count — it silently rescales gradients through the loss normalizer.
# GLOBAL_BATCH defaults to exactly that product; this catches an override.
if (( GLOBAL_BATCH != ROLLOUT_BATCH * SAMPLES_PER_PROMPT )); then
  echo "ERROR: GLOBAL_BATCH=$GLOBAL_BATCH but ROLLOUT_BATCH x SAMPLES_PER_PROMPT" >&2
  echo "= $((ROLLOUT_BATCH * SAMPLES_PER_PROMPT)). Their actor requires these to be equal:" >&2
  echo "one optimizer step per rollout keeps training on-policy." >&2
  exit 1
fi

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
# Their released checkpoint is rollout_id 4199, which at their measured ~47s/step
# is ~55h on 2 GPUs. Their own LR scan shows most of the gain arrives early — 30
# steps moved fve_nrm from a 0.375 warm-start to 0.377-0.483 — so default to a
# short run and extend if FVE is still climbing.
NUM_ROLLOUT="${NUM_ROLLOUT:-400}"
# Must match what the AV was trained with. 1000 for us (D29): the rule is "a round
# number just above the dataset's mean activation norm" (~900 here), NOT sqrt(d).
# The reference has no default for this — absent means train_actor asserts — and a
# mismatch is the failure where the AV free-associates off the placeholder instead
# of reading the vector. mse_scale is separate (D30) and comes from the sidecar,
# defaulting to sqrt_d_model = 45.25, which is what we want.
INJECTION_SCALE="${INJECTION_SCALE:-1000}"

# --kl-coef is a no-op for GRPO (get_grpo_returns discards the kl tensor);
# --use-kl-loss is the path that actually adds KL to the policy loss. It is
# store_true, so gate on the env var to allow turning it off entirely.
if "$RL_PYTHON" -c "import sys;sys.exit(0 if float('$KL_LOSS_COEF') != 0 else 1)"; then
  # k2, not miles' default k1. NLAFSDPActor asserts on k1 + --use-kl-loss: as a
  # direct loss term, k1 = (log p - log p_ref) has zero expected gradient under
  # the sampling distribution, so the penalty does nothing while looking active in
  # the logs. Their configs/rl.sh ships k2; k1 is only legal with --use-unbiased-kl.
  KL_FLAGS=(--use-kl-loss --kl-loss-coef "$KL_LOSS_COEF" --kl-loss-type "${KL_LOSS_TYPE:-k2}")
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
   gpus          actor $ACTOR_GPUS / critic $CRITIC_GPUS / rollout $([[ "$COLOCATE" == "1" ]] && echo "colocated with actor" || echo "$ROLLOUT_GPUS")  ($WANT_GPUS of $VISIBLE_GPUS visible)
   batch         $ROLLOUT_BATCH prompts x $SAMPLES_PER_PROMPT samples = $GLOBAL_BATCH
   lr            actor $ACTOR_LR / critic $CRITIC_LR  (1.41e-5 x sqrt($GLOBAL_BATCH/1024))
   kl coef       $KL_LOSS_COEF
   response cap  $MAX_RESPONSE_LEN
   rollouts      $NUM_ROLLOUT  (theirs: 4199 ~= 55h on 2 GPUs)
   inj scale     $INJECTION_SCALE  (mse_scale comes from the checkpoint sidecar)
==============================================================
EOM

# The actor init must be a readable HF checkpoint, since that is now how its
# weights arrive. A DCP-style dir here would load nothing and train from scratch.
if [[ ! -f "$ACTOR_SFT_CKPT/config.json" ]]; then
  echo "ERROR: $ACTOR_SFT_CKPT has no config.json, so --hf-checkpoint cannot" >&2
  echo "initialize the actor from it. If this is a miles DCP checkpoint dir" >&2
  echo "(latest_checkpointed_iteration.txt + iter_*/), export it to HF first." >&2
  exit 1
fi

for ckpt in "$ACTOR_SFT_CKPT" "$CRITIC_SL_CKPT"; do
  if [[ ! -f "$ckpt/nla_meta.yaml" ]]; then
    echo "ERROR: $ckpt/nla_meta.yaml missing." >&2
    echo "  Stage 2 reads its NLA settings from that sidecar; our SFT loop does not" >&2
    echo "  write one. Generate it first:" >&2
    echo "    uv run python scripts/make_nla_sidecar.py --checkpoint $ckpt" >&2
    exit 1
  fi
done

cd "$NLA_REPO"
# cd into the miles root to match their invocation exactly. Every path we pass
# below is absolute, so the move is safe.
cd "$MILES_SRC"
exec "$RL_PYTHON" "$MILES_TRAIN" \
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
    --hf-checkpoint "$ACTOR_SFT_CKPT" \
    --ref-load "$ACTOR_SFT_CKPT" \
    --load "$RESUME_FROM" \
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
    "${COLOCATE_FLAGS[@]}" \
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
    --num-rollout "$NUM_ROLLOUT" \
    --nla-injection-scale "$INJECTION_SCALE" \
    --lr "$ACTOR_LR" --lr-decay-style constant \
    --attn-implementation "${ATTN_IMPL:-flash_attention_2}" \
    `# NO --gradient-checkpointing: it deadlocks NCCL in update_weights() —` \
    `# FSDP's full-param gather changes, the broadcast hangs, watchdog SIGABRTs.` \
    "${KL_FLAGS[@]}" \
    --save-interval "$SAVE_INTERVAL" \
    --loss-mask-type "${LOSS_MASK_TYPE:-qwen}" \
    "$@"
