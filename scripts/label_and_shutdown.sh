#!/usr/bin/env bash
# Chain: wait for extract -> split -> serve a local model on both GPUs -> label
# both halves -> stop the server -> delete this pod.
#
#   tmux new -s pipeline
#   scripts/label_and_shutdown.sh
#   # Ctrl-b d
#
#   NO_SHUTDOWN=1 scripts/label_and_shutdown.sh     # keep the pod alive at the end
#   LIMIT=2000 scripts/label_and_shutdown.sh        # small trial run
#   N_DOCS=100000 scripts/label_and_shutdown.sh     # EXTEND an existing corpus
#
# Extending: set N_DOCS above what is already extracted and re-run. The script
# extracts only the missing documents into a second shard, re-splits with the
# existing doc->half assignment PINNED, and relabels nothing that already has a
# summary. The reference trained on 100k docs x 5 positions = 500k pairs split
# 250k/250k (configs/TRAINING_NOTES.md); our first pass did 40k docs = 200k
# pairs, and the AR warm-start plateaued by step ~290 of 1143 — data-limited,
# not undertrained (D46).
#
# Labeling runs against a locally served model, so it costs GPU time and no API
# credit. Every stage is resumable: extract/split skip if their output exists, and
# explain skips chunks it already wrote — so re-running after a crash resumes
# rather than re-paying.
#
# The pod deletes itself through its ServiceAccount, which was granted exactly
# "delete pods" in this namespace (k8s-selfdelete-rbac.yaml). No kubeconfig is
# stored on the PVC.
set -euo pipefail
cd "$(dirname "$0")/.."

D="${DATA_DIR:-/workspace/data/warmstart}"
# Output stem for the explained/sft parquets. Set this when changing the prompt:
# explain resumes from `{stem}_explained.parquet.chunks`, so reusing the stem
# after a prompt change would silently keep the OLD labels.
STEM="${STEM:-}"
MODEL="${LABEL_MODEL:-Qwen/Qwen3.8-27B}"
PORT="${PORT:-8000}"
TP="${TP:-2}"
CONC="${CONC:-64}"
LIMIT="${LIMIT:-}"
N_DOCS="${N_DOCS:-40000}"
# Python block-buffers stdout when it is redirected to a file, so per-chunk
# progress would sit in an 8KB buffer for tens of minutes and the run would look
# hung. The shell's own `log` lines are unaffected, which makes it look worse:
# you see the stage banners and then nothing.
export PYTHONUNBUFFERED=1
export HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/workspace/.cache/uv}"
mkdir -p "$D"

log () { echo "[$(date +%H:%M:%S)] $*"; }

# --- 1. extract -----------------------------------------------------------
# Shards, oldest first. split streams them IN THIS ORDER, which is what keeps
# each half a prefix of its previous self and lets explain resume by offset.
SHARDS=("$D/base.parquet")
HAVE_DOCS=0
if [[ -f "$D/base.parquet" ]]; then
  HAVE_DOCS=$(uv run python -c "
from reasoning_attention.datagen.sidecar import read_sidecar
import sys
m = read_sidecar(sys.argv[1])['extraction']
print(m['corpus_start'] + m['n_documents'])
" "$D/base.parquet")
fi

# Wait rather than start a second one: an extract may already be running in
# another tmux session, and two writers on one parquet path corrupt it.
if [[ "$HAVE_DOCS" -ge "$N_DOCS" ]]; then
  log "extract: base.parquet already covers $HAVE_DOCS >= $N_DOCS docs, skipping"
elif [[ "$HAVE_DOCS" -gt 0 ]]; then
  # Extend. Positions are drawn from an RNG keyed on (seed, doc_id), so a run
  # over a disjoint document range merges row-for-row with the first one.
  EXTRA="$D/base_${HAVE_DOCS}_${N_DOCS}.parquet"
  SHARDS+=("$EXTRA")
  if [[ -f "$EXTRA" ]]; then
    log "extract: $EXTRA exists, skipping"
  else
    log "extract: extending by $((N_DOCS - HAVE_DOCS)) docs from offset $HAVE_DOCS"
    uv run python -m reasoning_attention.datagen.extract \
        --output "$EXTRA" --corpus-start "$HAVE_DOCS" \
        --n-documents "$((N_DOCS - HAVE_DOCS))" --batch-size 16
  fi
elif pgrep -f "datagen.extract" >/dev/null; then
  log "extract: already running elsewhere — waiting for it"
  while pgrep -f "datagen.extract" >/dev/null; do sleep 60; done
  [[ -f "$D/base.parquet" ]] || { log "ERROR: extract exited without writing base.parquet"; exit 1; }
  log "extract: finished"
else
  log "extract: starting ($N_DOCS docs)"
  uv run python -m reasoning_attention.datagen.extract \
      --output "$D/base.parquet" --n-documents "$N_DOCS" --batch-size 16
fi

# --- 2. split -------------------------------------------------------------
# Always re-run: with one shard this is a no-op rebuild, with two it appends the
# new documents. --preserve-halves pins every doc already assigned, so the old
# rows keep both their half and their position and the chunk cache stays valid.
PRESERVE=()
[[ -f "$D/halves/av_half.parquet" ]] && PRESERVE=(--preserve-halves "$D/halves")
if [[ ${#SHARDS[@]} -eq 1 && -f "$D/halves/av_half.parquet" ]]; then
  log "split: halves exist and there is nothing to add, skipping"
else
  log "split: partitioning ${#SHARDS[@]} shard(s) into disjoint AV / AR halves"
  # The final chunk of a previous labelling run covers a PARTIAL slice; once the
  # half grows, that same offset denotes a full chunk. Chunks written before the
  # nla_input_rows key existed cannot report this, so drop the last one of each
  # and let it be relabelled — a few hundred rows, versus silently losing them.
  for half in av ar; do
    last=$(ls -1 "$D/${half}${STEM}_explained.parquet.chunks"/chunk_*.parquet 2>/dev/null | tail -1 || true)
    [[ -n "$last" ]] && { log "split: dropping partial tail chunk $(basename "$last")"; rm -f "$last"; }
  done
  uv run python -m reasoning_attention.datagen.split \
      --base "${SHARDS[@]}" --output-dir "$D/halves" "${PRESERVE[@]}"
fi

# --- 3. serve -------------------------------------------------------------
# Both GPUs are free only now that extract is done, which is why the server
# starts here and not at the top of the script.
# Window sizing: contexts run to 4096 tokens (WarmStartDataConfig.max_context_tokens),
# plus ~400 tokens of instruction, plus the reserved output cap. 8192 was too tight —
# the server 400s when context + reserved output crosses the window — so leave slack.
# An interrupted Xet download leaves shards truncated under their final names, so
# the cache looks complete and the failure only surfaces minutes later inside the
# engine. Parse the headers first — cheap, and it fails with an actionable message.
log "verify: checking cached shards of $MODEL"
uv run python scripts/verify_weights.py "$MODEL" "$HF_HOME" || exit 1

log "serve: $MODEL on $TP GPU(s), port $PORT"
uv run vllm serve "$MODEL" \
    --port "$PORT" --tensor-parallel-size "$TP" \
    --gpu-memory-utilization 0.90 --max-model-len "${MAX_MODEL_LEN:-16384}" \
    > "$D/vllm.log" 2>&1 &
VLLM_PID=$!
# Always take the server down, including on failure — otherwise a crashed run
# leaves 2 GPUs pinned by an orphan process.
cleanup () { log "stopping vLLM (pid $VLLM_PID)"; kill "$VLLM_PID" 2>/dev/null || true; wait "$VLLM_PID" 2>/dev/null || true; }
trap cleanup EXIT

log "serve: waiting for the server to come up (model load is minutes)"
for i in $(seq 1 120); do
  if curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    log "serve: ready after ~$((i * 15))s"; break
  fi
  kill -0 "$VLLM_PID" 2>/dev/null || { log "ERROR: vLLM died — see $D/vllm.log"; tail -20 "$D/vllm.log"; exit 1; }
  sleep 15
done
curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null || { log "ERROR: server never became ready"; exit 1; }

# --- 4. label -------------------------------------------------------------
LIMIT_ARG=()
[[ -n "$LIMIT" ]] && LIMIT_ARG=(--limit "$LIMIT")
# INSTRUCTION selects the explainer prompt. Default `reference` is byte-identical
# to theirs and INVITES inline textual examples, so --reject-verbatim is invalid
# with it (explain.py refuses the combination) — verbatim overlap is the signal
# there, not a leak. See D48.
INSTRUCTION="${INSTRUCTION:-reference}"
REJECT_ARG=()
[[ "$INSTRUCTION" == "noquote" && "${REJECT_VERBATIM:-1}" == "1" ]] && REJECT_ARG=(--reject-verbatim)
# Resumes chunk-by-chunk: everything already labelled is skipped, so extending
# costs GPU time only for the new rows.
for half in av ar; do
  log "label: $half half"
  uv run python -m reasoning_attention.datagen.explain \
      --input "$D/halves/${half}_half.parquet" \
      --output "$D/${half}${STEM}_explained.parquet" \
      --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" \
      --instruction "$INSTRUCTION" \
      --concurrency "$CONC" "${REJECT_ARG[@]}" "${LIMIT_ARG[@]}"
done

# --- 5. build -------------------------------------------------------------
for half in av ar; do
  log "build: ${half}_sft"
  uv run python -m reasoning_attention.datagen.build \
      --input "$D/${half}${STEM}_explained.parquet" --output "$D/${half}${STEM}_sft.parquet" --stage "${half}_sft"
done
log "done: $(ls -1 "$D"/*_sft.parquet | tr '\n' ' ')"

# --- 6. shut down ---------------------------------------------------------
if [[ -n "${NO_SHUTDOWN:-}" ]]; then
  log "NO_SHUTDOWN set — leaving the pod running"
  exit 0
fi
cleanup; trap - EXIT
SA=/var/run/secrets/kubernetes.io/serviceaccount
NS=$(cat "$SA/namespace")
POD="${POD_NAME:-$(hostname)}"
log "deleting pod $NS/$POD — PVC data is untouched"
curl -s --cacert "$SA/ca.crt" -H "Authorization: Bearer $(cat "$SA/token")" \
     -X DELETE "https://kubernetes.default.svc/api/v1/namespaces/$NS/pods/$POD" | head -c 200
echo
sleep 60   # give the API server time to terminate us
