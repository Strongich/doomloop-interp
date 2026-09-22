#!/usr/bin/env bash
# Wait for the v2 corpus build to finish, then start Stage-2 RL from the SFT init.
#
#   tmux new-session -d -s chain 'bash scripts/chain_rl_after_build.sh'
#
# Exists because the build is ~6h and the training launch behind it is otherwise
# a manual step someone has to be awake for. It refuses to launch on a build that
# did not actually finish: a truncated corpus would train for days on a fraction
# of the data and look completely normal in the logs.
set -uo pipefail
cd "$(dirname "$0")/.."

BUILD_SESSION="${BUILD_SESSION:-rl2}"
DATA_DIR="${DATA_DIR:-/workspace/data/rl2}"
RL_PARQUET="${RL_PARQUET:-$DATA_DIR/rl.parquet}"
RUN_DIR="${RUN_DIR:-/workspace/data/runs/grpo_v2}"
ACTOR_SFT_CKPT="${ACTOR_SFT_CKPT:-/workspace/data/checkpoints/av_sft}"
CRITIC_SL_CKPT="${CRITIC_SL_CKPT:-/workspace/data/checkpoints/ar_sft}"
TRAIN_LOG="${TRAIN_LOG:-/workspace/data/grpo_v2.log}"
# Below this the build did not produce the three sources we asked for: web alone
# is ~500k and the two reasoning sources add ~600k more.
MIN_ROWS="${MIN_ROWS:-900000}"
EPOCHS="${EPOCHS:-0.5}"

say () { echo "[chain $(date -u +%H:%M:%S)] $*"; }

say "waiting for build session '$BUILD_SESSION' to finish"
while tmux has-session -t "$BUILD_SESSION" 2>/dev/null; do sleep 60; done
say "build session ended"

if [[ ! -f "$RL_PARQUET" ]]; then
  say "ABORT: $RL_PARQUET was never written — the build failed. Not launching."
  exit 1
fi

rows=$(uv run python -c "
import sys, pyarrow.parquet as pq
try:
    print(pq.read_metadata(sys.argv[1]).num_rows)
except Exception:
    print(0)
" "$RL_PARQUET" 2>/dev/null)

if [[ "${rows:-0}" -lt "$MIN_ROWS" ]]; then
  say "ABORT: $RL_PARQUET has ${rows:-0} rows, expected >= $MIN_ROWS."
  say "  A short corpus means a stage failed. Check the build log before launching."
  exit 1
fi
say "corpus OK: $rows rows"

# The compat guard in train_grpo.sh would catch this too, but catching it here
# means the failure surfaces now rather than after the engine actors spin up.
COMPAT="$(readlink -f /usr/local/cuda 2>/dev/null)/compat"
if [[ -d "$COMPAT" ]]; then
  say "ABORT: $COMPAT still exists and its libcuda is older than the driver."
  say "  It would shadow the real libcuda inside the sglang engine actors and"
  say "  CUDA init would fail with error 803. Rename it, then re-run this script:"
  say "    mv $COMPAT $COMPAT.disabled-mismatched-driver"
  exit 1
fi

say "launching Stage-2 RL: EPOCHS=$EPOCHS BF16_EMBEDS=1 -> $TRAIN_LOG"
RL_PARQUET="$RL_PARQUET" \
ACTOR_SFT_CKPT="$ACTOR_SFT_CKPT" \
CRITIC_SL_CKPT="$CRITIC_SL_CKPT" \
RUN_DIR="$RUN_DIR" \
EPOCHS="$EPOCHS" \
BF16_EMBEDS=1 \
DISTRIBUTED_POST=1 \
TIS_METRICS=1 \
  bash scripts/train_grpo.sh > "$TRAIN_LOG" 2>&1 &
TRAIN_PID=$!
say "train_grpo.sh pid $TRAIN_PID"

# --- bf16 transport watchdog -------------------------------------------------
# BF16_EMBEDS=1 is a numerics change, not just an encoding one: our injection
# scale is exactly 1000, the boundary of the reference's own "bf16 is safe below
# 1000" gate. tis_k3 is the train<->rollout logprob mismatch and the direct read
# on whether the transport is right: ~0.001 when it is, ~0.20 when it is not.
#
# Worth automating because this run is ~4.5 days. A broken transport does not
# crash — it trains on a policy that disagrees with its own rollouts, and looks
# entirely normal in the loss curve until the FVE fails to move.
TIS_MAX="${TIS_MAX:-0.05}"
TIS_DEADLINE=$(( $(date +%s) + ${TIS_WAIT_SECS:-5400} ))
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  tis=$(grep -oE "tis_k3[\"']?[:= ]+[0-9.eE+-]+" "$TRAIN_LOG" 2>/dev/null \
        | grep -oE "[0-9.eE+-]+$" | head -1)
  if [[ -n "$tis" ]]; then
    if awk "BEGIN{exit !($tis > $TIS_MAX)}"; then
      say "ABORT: first tis_k3=$tis exceeds $TIS_MAX — the bf16 transport is wrong."
      say "  Killing pid $TRAIN_PID rather than training days on a bad rollout path."
      say "  Re-run with BF16_EMBEDS=0 (fp32, slower but known good)."
      kill -9 "$TRAIN_PID" 2>/dev/null
      exit 1
    fi
    say "tis_k3=$tis (<= $TIS_MAX) — transport OK, letting it run"
    break
  fi
  if (( $(date +%s) > TIS_DEADLINE )); then
    say "WARNING: no tis_k3 seen before the deadline; leaving the run alone."
    say "  Check $TRAIN_LOG by hand."
    break
  fi
  sleep 60
done

wait "$TRAIN_PID"
say "train_grpo.sh exited with $?"
