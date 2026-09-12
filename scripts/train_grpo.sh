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

# RESUMING AN RL RUN — the two halves resume by DIFFERENT mechanisms, and getting
# only one of them right is silent:
#
#   actor  --load $RUN_DIR/actor           miles-native DCP, restores weights +
#                                          optimizer + the dataset iterator
#                                          position (placement_group.py:176), so
#                                          the prompt stream continues and one
#                                          epoch really means no repeats.
#   critic --critic-load <HF dir>          FSDP resolves the critic's
#                                          hf_checkpoint FROM --critic-load
#                                          (arguments.py:284), so this must point
#                                          at the RL critic's saved HF dir —
#                                          $RUN_DIR/critic/iter_NNNNNNN/hf — and
#                                          NOT at the Stage-1 SFT critic.
#
# Leaving CRITIC_SL_CKPT at the SFT critic on a resume reverts the reward model to
# its warm-start while the actor carries on from a trained policy: the AV then
# optimizes against a WORSE critic and fve_nrm drops back to its step-0 value.
# That is the D42 failure mode with the roles swapped. `--num-rollout` is the
# TOTAL (range(start_rollout_id, num_rollout)), not an increment.
#
# The check that it worked: fve_nrm at the first resumed step should match the
# last window of the previous run, not its opening window.
if [[ -f "$RESUME_FROM/latest_checkpointed_iteration.txt" ]]; then
  _resume_iter="$(cat "$RESUME_FROM/latest_checkpointed_iteration.txt")"
  echo "NOTE: resuming actor from $RESUME_FROM at iteration $_resume_iter" >&2
  case "$CRITIC_SL_CKPT" in
    "$RUN_DIR"/critic/*) : ;;
    *)
      echo "WARNING: --load will resume the actor at iteration $_resume_iter, but" >&2
      echo "  CRITIC_SL_CKPT=$CRITIC_SL_CKPT is not under $RUN_DIR/critic/." >&2
      echo "  The critic would restart from its SFT init and the reward signal" >&2
      echo "  would regress. Set:" >&2
      echo "    CRITIC_SL_CKPT=$RUN_DIR/critic/iter_$(printf '%07d' "$_resume_iter")/hf" >&2
      echo "  Set ALLOW_CRITIC_MISMATCH=1 only if you intend to reset the critic." >&2
      [[ "${ALLOW_CRITIC_MISMATCH:-0}" == "1" ]] || exit 1
      ;;
  esac
fi

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
# Same class of check: under transformers 5.x, apply_chat_template(tokenize=True)
# returns a BatchEncoding, and compute_canonical_neighbors iterates it expecting
# list[int]. Every config load goes through that function.
NLA_SCHEMA_PY="$NLA_REPO/nla/schema.py"
if [[ -f "$NLA_SCHEMA_PY" ]] && ! grep -q 'hasattr(ids, "keys")' "$NLA_SCHEMA_PY"; then
  echo "ERROR: $NLA_SCHEMA_PY does not unwrap BatchEncoding." >&2
  echo "Neighbor verification would report the injection token 0x and abort." >&2
  echo "Apply it with:" >&2
  echo "  $RL_PYTHON scripts/patch_nla_batchencoding.py --nla-repo $NLA_REPO" >&2
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
# Captured before the default is applied, so the EPOCHS guard can tell an explicit
# NUM_ROLLOUT from the fallback.
_num_rollout_explicit="${NUM_ROLLOUT:-}"
NUM_ROLLOUT="${NUM_ROLLOUT:-400}"
# EPOCHS is the friendlier knob: one epoch is rows/ROLLOUT_BATCH rollouts, because
# miles draws ROLLOUT_BATCH prompts per step and the DCP resume restores the
# dataset iterator position, so a step count really does map to corpus coverage.
# Deriving it from the parquet beats hand-arithmetic — the v2 corpus row count is
# not round, and getting it wrong silently trains a different fraction than
# intended. An explicit NUM_ROLLOUT still wins, so old invocations are unchanged.
if [[ -n "${EPOCHS:-}" ]]; then
  if [[ -n "$_num_rollout_explicit" ]]; then
    echo "ERROR: set either EPOCHS or NUM_ROLLOUT, not both." >&2
    exit 1
  fi
  _rows="$("$RL_PYTHON" -c "
import pyarrow.parquet as pq, sys
print(pq.read_metadata(sys.argv[1]).num_rows)
" "$RL_PARQUET")"
  NUM_ROLLOUT="$("$RL_PYTHON" -c "
import math, sys
rows, epochs, batch = int(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
print(math.ceil(rows * epochs / batch))
" "$_rows" "$EPOCHS" "$ROLLOUT_BATCH")"
  echo "EPOCHS=$EPOCHS over $_rows rows at $ROLLOUT_BATCH prompts/step -> --num-rollout $NUM_ROLLOUT" >&2
fi
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

# --- rollout transport --------------------------------------------------
# Every request ships the prompt's whole embedding matrix. nla_generate picks
# the encoding at rollout/nla_generate.py:221:
#
#   bf16_safe = sglang_disable_radix_cache and scale < 1000
#   if _BF16_B64_EMBEDS or bf16_safe: <bf16-base64>  else: <fp32 JSON>
#
# Our injection_scale is EXACTLY 1000 and the gate is `< 1000`, so we take the
# fp32 branch: ~12 MB of JSON per request against ~2.8 MB for bf16-base64
# (their line 54). At 512 requests/rollout that is ~6 GB serialized on the
# single RolloutManager event loop — measured pinned at 100% of one core while
# sglang idled at 2.3 concurrent requests (server cap 4096, semaphore 512, KV
# usage 0.00) and step time sat at 158s against their documented ~47s at the
# same 64x8 batch.
#
# BF16_EMBEDS=1 sets the env override, taking the compact path without touching
# injection_scale. Their gate is labelled a heuristic and is aimed at gemma27b
# at scale 60000 (~4% KL spikes); we are 60x below that. It is still a numerics
# change, so it is OFF by default and TIS_METRICS is the check: `tis_k3` is the
# train<->rollout logprob mismatch, ~0.001 when the transport is right and ~0.20
# when it is wrong (TRAINING_NOTES.md:277).
#
# bf16 is a TWO-SIDED protocol and the server half is a separate patch. The
# client sends `input_embeds_b64_bf16` + `input_embeds_shape`;
# patches/nla_input_embeds_b64.patch teaches sglang's http_server.py to decode
# them. Without it FastAPI validates a base64 STRING against a numeric-array
# schema and answers 400 on every request — the run reaches step 0, generates
# nothing, and dies. That is exactly what happened on 2026-08-28: the patch
# applied on v0.5.8 (rl_setup2.log) but not on v0.5.15, where upstream had moved
# the file, and setup_rl_stack.sh SKIPped it without failing. Check for the
# decoder rather than trusting the flag.
# The client takes the bf16 branch on EITHER condition, and `bf16_safe` never
# checks whether the server can decode. We disable the radix cache always
# (required for input_embeds), so the ONLY thing keeping us off the broken path
# is injection_scale being exactly 1000 rather than < 1000. Lower the scale — a
# rounding choice, e.g. 900 for a mean norm of ~800 — and the fast path turns
# itself on with no flag set and the run dies at step 0. So gate on the EFFECTIVE
# transport, not on BF16_EMBEDS.
WANTS_BF16=0
[[ "${BF16_EMBEDS:-0}" == "1" ]] && WANTS_BF16=1
if awk "BEGIN{exit !($INJECTION_SCALE < 1000)}"; then WANTS_BF16=1; fi

if [[ "$WANTS_BF16" == "1" ]]; then
  WHY="BF16_EMBEDS=1"
  [[ "${BF16_EMBEDS:-0}" != "1" ]] && \
    WHY="injection_scale=$INJECTION_SCALE is < 1000, which turns on bf16 transport by itself"
  HTTP_SERVER="$("$RL_PYTHON" -c \
      "import sglang.srt.entrypoints.http_server as m; print(m.__file__)" 2>/dev/null)"
  if [[ -z "$HTTP_SERVER" || ! -f "$HTTP_SERVER" ]]; then
    echo "ERROR: $WHY, but sglang.srt.entrypoints.http_server is not importable" >&2
    echo "  from $RL_PYTHON. Fix the RL venv." >&2
    exit 1
  fi
  if ! grep -q "input_embeds_b64_bf16" "$HTTP_SERVER"; then
    echo "ERROR: $WHY, but the server-side bf16 decoder is missing from" >&2
    echo "  $HTTP_SERVER" >&2
    echo "  Every /generate would return 400 Bad Request and no rollout would" >&2
    echo "  complete. Apply patches/nla_input_embeds_b64.patch (it must be REBASED" >&2
    echo "  for sglang >= v0.5.15 — the upstream file moved), or run with" >&2
    echo "  BF16_EMBEDS=0 and an injection_scale >= 1000 to stay on fp32." >&2
    echo "  NOTE: --use-distributed-post is the numerics-free alternative for the" >&2
    echo "  same bottleneck — see D47." >&2
    exit 1
  fi
  [[ "${BF16_EMBEDS:-0}" == "1" ]] && export NLA_BF16_B64_EMBEDS=1
fi

# --- rollout POST fan-out (D47) ------------------------------------------
# The fp32 payload is ~12 MB/request and 512 requests per rollout are encoded on
# ONE asyncio loop: measured RolloutManager pinned at 100% of a single core on a
# 256-core box while sglang idled at 2.3 concurrent requests and step time sat at
# 158s vs their ~47s. --use-distributed-post moves the POST (and its json
# encoding) onto Ray actors, and hands the payload over via Ray's binary
# serialization instead of an inline json.dumps on the caller.
#
# miles creates `num_gpus_per_node` actors per node (http_utils.py:274) — the
# MILES_HTTP_POST_ACTORS_PER_NODE in its docstring is NOT read by the code. So
# this buys us 2 posters, not 256: expect roughly 2x on the serialization share,
# not a full close of the 3.4x gap. Numerics are untouched, which is why this is
# worth trying before rebasing nla_input_embeds_b64.patch.
if [[ "${DISTRIBUTED_POST:-0}" == "1" ]]; then
  DPOST_FLAGS=(--use-distributed-post)
else
  DPOST_FLAGS=()
fi

# metrics_only returns pg_loss unchanged — pure observation, so it is on by
# default. Without --get-mismatch-metrics the tis_* keys are never computed.
if [[ "${TIS_METRICS:-1}" == "1" ]]; then
  TIS_FLAGS=(--get-mismatch-metrics
             --custom-tis-function-path "${TIS_FUNCTION:-nla.tis_metrics.metrics_only}")
else
  TIS_FLAGS=()
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
   rollouts      $NUM_ROLLOUT${EPOCHS:+  (= $EPOCHS epoch)}  (theirs: 4199 ~= 55h on 2 GPUs)
   inj scale     $INJECTION_SCALE  (mse_scale comes from the checkpoint sidecar)
   embeds        $([[ "${BF16_EMBEDS:-0}" == "1" ]] && echo "bf16-base64 (~2.8MB/req, FORCED past the scale<1000 gate)" || echo "fp32 JSON (~12MB/req — the default at scale $INJECTION_SCALE)")
   post fan-out  $([[ "${DISTRIBUTED_POST:-0}" == "1" ]] && echo "on — $NUM_GPUS_PER_NODE Ray POST actors (watch perf/train_wait_time)" || echo "off — all 512 requests encoded on one event loop")
   tis metrics   $([[ "${TIS_METRICS:-1}" == "1" ]] && echo "on — watch tis_k3: ~0.001 good, ~0.20 means the transport is wrong" || echo "off")
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

# The tis function is imported by path inside the training process; a typo
# would surface as a crash tens of GPU-minutes in, after Ray and sglang start.
if [[ "${TIS_METRICS:-1}" == "1" ]]; then
  "$RL_PYTHON" -c "
import importlib, sys
mod, _, fn = sys.argv[1].rpartition('.')
getattr(importlib.import_module(mod), fn)
" "${TIS_FUNCTION:-nla.tis_metrics.metrics_only}" || {
    echo "ERROR: cannot import ${TIS_FUNCTION:-nla.tis_metrics.metrics_only}" >&2
    echo "  Set TIS_METRICS=0 to run without the mismatch diagnostic." >&2
    exit 1
  }
fi

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
    `# sglang >= 0.5.15 captures a "breakable" prefill CUDA graph, which raises` \
    `# NotImplementedError: Breakable CUDA graph is not compatible with memory` \
    `# saver mode. --colocate forces --offload, which needs the memory saver, so` \
    `# the prefill graph is what has to give. Decode graphs are unaffected.` \
    --sglang-disable-prefill-cuda-graph \
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
    "${TIS_FLAGS[@]}" \
    "${DPOST_FLAGS[@]}" \
    --save-interval "$SAVE_INTERVAL" \
    --loss-mask-type "${LOSS_MASK_TYPE:-qwen}" \
    "$@"
