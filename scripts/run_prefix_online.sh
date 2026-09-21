#!/usr/bin/env bash
# The ONLINE fixed-prefix experiment -- the deployable version of Finding 10.
#
# Finding 10 froze at the FIRST probe of a four-agreement run, which needs the
# three later probes to know the run exists. That is a valid paired comparison
# at a prefix chosen with hindsight; it is not a policy. Here the freeze is the
# FOURTH agreeing probe, the first boundary at which the rule could actually
# fire, so nothing downstream of the cut is consulted to choose it (D57).
#
# Also re-probes with the corrected extractor: braces preserved, unclosed boxes
# discarded as unknown, 96-token probe budget, symbolic agreement. On this
# GSM8K-heavy set those change 0.15%/0.3% of probes, but the old extractor could
# not represent a symbolic answer at all, so the re-probe keeps one code path.
#
#   bash scripts/run_prefix_online.sh
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/prefix_online logs
say () { echo "[online $(date -u +%H:%M:%S)] $*"; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SEEDS=${SEEDS:-2}
BACKEND=${BACKEND:-vllm}
BATCH=${BATCH:-32}
LIMIT=${LIMIT:-0}
OUTDIR=${OUTDIR:-data/prefix_online_${BACKEND}}
# Keep the completed/stopped HF branches in data/prefix_online untouched.
mkdir -p "$OUTDIR"

if [[ ! -s data/prefix/candidates_v2.jsonl ]]; then
  say "re-probing with the corrected extractor (~10 min, vLLM)"
  uv run python scripts/probe_candidates.py --limit 400 \
      --out data/prefix/candidates_v2.jsonl 2>&1 | tee logs/probe_candidates_v2.log
fi

if [[ ! -s data/prefix_online/prefixes.jsonl ]]; then
  say "freezing at the fourth agreeing probe"
  uv run python scripts/build_prefix_set.py --rule run4 \
      --candidates data/prefix/candidates_v2.jsonl \
      --out data/prefix_online/prefixes.jsonl \
      --audit data/prefix_online/audit.md 2>&1 | tee logs/build_prefix_online.log
fi

say "branching: backend=$BACKEND seeds=$SEEDS out=$OUTDIR"
uv run python scripts/branch_continue.py \
    --backend "$BACKEND" --seeds "$SEEDS" --batch "$BATCH" --alpha 1.0 --max-new-tokens 12288 \
    --exit-tokens 4096 --limit "$LIMIT" \
    --prefixes data/prefix_online/prefixes.jsonl --outdir "$OUTDIR" \
    --arms base N D exit 2>&1 | tee -a "logs/prefix_online_${BACKEND}_branch.log"

say "scoring"
uv run python scripts/prefix_report.py --csv "$OUTDIR/branches.csv" 2>&1 \
    | tee "logs/prefix_online_${BACKEND}_report.log"
say "ONLINE_DONE"
