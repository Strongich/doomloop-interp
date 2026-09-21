#!/usr/bin/env bash
# Fixed-prefix branching pilot -- see EXPERIMENT-fixed-prefix.md for the design.
#
#   bash scripts/run_prefix_pilot.sh            # full run, resumable
#   LIMIT=80 SEEDS=1 bash scripts/run_prefix_pilot.sh   # shakeout
#
# vLLM resumes from its fsync'd JSONL journal; CSVs are derived snapshots.
# BACKEND=transformers retains the historical CSV-based runner. Backend output
# folders are separate. An interrupted vLLM call loses at most --checkpoint-every
# requests, which is independent of the number of concurrently scheduled requests.
#
# Seed-major ordering: seed 0 finishes every arm before seed 1 starts, so an
# interrupted run is a complete experiment at smaller k rather than a ragged
# one. `prefix_report.py` reads the partial file directly and only scores
# questions that are complete across every arm and seed.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/prefix logs
say () { echo "[prefix $(date -u +%H:%M:%S)] $*"; }

LIMIT=${LIMIT:-0}
SEEDS=${SEEDS:-4}
BACKEND=${BACKEND:-vllm}
BATCH=${BATCH:-32}   # vLLM concurrent sequences; HF also applies --kv-budget
ALPHA=${ALPHA:-1.0}
MAX_NEW=${MAX_NEW:-12288}
EXIT_TOK=${EXIT_TOK:-4096}
ARMS=${ARMS:-"base N D exit"}
PREFIXES=${PREFIXES:-data/prefix/prefixes.jsonl}
OUTDIR=${OUTDIR:-data/prefix_${BACKEND}}
# The KV cache grows one token at a time and fragments the 16 GB card: batch 8
# OOM'd with 4.3 GB reserved but unallocated. Expandable segments fix that.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if [[ ! -s "$PREFIXES" ]]; then
  say "no prefixes -- running detection first"
  uv run python scripts/probe_candidates.py --limit 400 2>&1 | tee logs/probe_candidates.log
  uv run python scripts/build_prefix_set.py --rule run4 --out "$PREFIXES" 2>&1 | tee logs/build_prefix_set.log
fi

say "branching: backend=$BACKEND seeds=$SEEDS batch=$BATCH out=$OUTDIR"
# shellcheck disable=SC2086
uv run python scripts/branch_continue.py \
    --backend "$BACKEND" --seeds "$SEEDS" --batch "$BATCH" --alpha "$ALPHA" \
    --max-new-tokens "$MAX_NEW" --exit-tokens "$EXIT_TOK" --limit "$LIMIT" \
    --prefixes "$PREFIXES" --outdir "$OUTDIR" \
    --arms $ARMS 2>&1 | tee -a "logs/prefix_${BACKEND}_branch.log"

say "scoring"
uv run python scripts/prefix_report.py --csv "$OUTDIR/branches.csv" 2>&1 \
    | tee "logs/prefix_${BACKEND}_report.log"
say "PREFIX_PILOT_DONE"
