#!/usr/bin/env bash
# Is N's effect about steering AFTER the answer, or just about steering LATE?
#
# Finding 10 has an accidental two-point sweep -- the hindsight freeze at 321
# think tokens and the online one at 540 -- and they disagreed at k=1. This runs
# the freeze depth as a proper variable, plus the matched non-answer control that
# the pre-registration listed as stage 2 and never ran.
#
#   run3     freeze at the 3rd agreeing probe   median 393 tok, 394 questions
#   run4     the registered rule                median 540 tok, 387 questions
#   run6     freeze at the 6th agreeing probe   median 724 tok, 340 questions
#   shuf     run4 depth, boundary borrowed from ANOTHER question (median 523)
#
# `shuf` is the control that decides the mechanistic claim: it holds freeze depth
# fixed while breaking its coincidence with where the candidate settles. If N
# behaves the same there, "after the answer" is not doing the work -- "late in
# the trace" is.
#
# run4 is already done (data/prefix_online_vllm); only the three others run here.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/prefix_sweep logs
say () { echo "[sweep $(date -u +%H:%M:%S)] $*"; }
SEEDS=${SEEDS:-2}

for tag in run3 run6 shuf; do
  out="data/prefix_sweep/${tag}"
  if [[ -s "$out/branches.csv" ]]; then say "$tag done, skipping"; continue; fi
  say "=== $tag"
  mkdir -p "$out"
  uv run python scripts/branch_continue.py --backend vllm \
      --seeds "$SEEDS" --batch 32 --alpha 1.0 --max-new-tokens 12288 \
      --exit-tokens 4096 --limit 0 \
      --prefixes "data/prefix_sweep/prefixes_${tag}.jsonl" --outdir "$out" \
      --arms base N D exit 2>&1 | tee -a "logs/sweep_${tag}.log" \
      | grep -E "tok/s|resumed|Error" || true
  uv run python scripts/prefix_report.py --csv "$out/branches.csv" 2>&1 \
      | tee "logs/sweep_${tag}_report.log" | head -12
done
say "SWEEP_DONE"
