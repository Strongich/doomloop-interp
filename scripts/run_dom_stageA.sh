#!/usr/bin/env bash
# STAGE A of the NLA-necessity ablation. Do the cheap directions work?
#
#   bash scripts/run_dom_stageA.sh          # ~2h, resumable
#
# Cosines are already in (scripts/build_dom_directions.py): the two NLA
# directions agree with each other at 0.930, while diff-of-means sits at 0.43
# against them and the linear probe at 0.07. Finding 5d's scale says 0.754 is
# where "same meaning, different words" sits -- so these are genuinely DIFFERENT
# directions, and only behaviour can say whether they work.
#
# 150 questions, stratified 50 per band, disjoint from the 297 probe traces the
# directions were built from. Suppress-only: Finding 6's `none` column is the
# baseline, paired by question_id.
#
# Primary endpoint is DOUBT BLOCKS REMOVED, not length. Vector A removes 92%, the
# strongest system prompt removes 32% (Finding 7). Power to tell a cheap
# direction apart from A is >=0.95 at n=50 anywhere in that range. Accuracy at
# n=150 resolves only ~11.5pp and is reported as a crude bound, never a result.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/dom logs
say () { echo "[dom $(date -u +%H:%M:%S)] $*"; }

SET=${SET:-data/dom_testset150.jsonl}
LIMIT=${LIMIT:-150}
ALPHA=${ALPHA:-1.0}
MAX_NEW=${MAX_NEW:-12288}
BATCH=${BATCH:-4}

run () {
  local tag="$1" dir="$2"
  local out="data/dom/${tag}.csv"
  if [[ -s "$out" ]]; then say "$tag done, skipping"; return 0; fi
  say "arm $tag ($dir)"
  uv run python scripts/suppress_answer.py \
      --traces "$SET" --limit "$LIMIT" --alpha "$ALPHA" --batch "$BATCH" \
      --max-new-tokens "$MAX_NEW" --seed 0 --direction "$dir" \
      --conditions suppress --out "$out" --dump "data/dom/${tag}_texts.jsonl" \
      2>&1 | tee "logs/dom_${tag}.log" \
      | grep -E "correct|median|doubt|capped|resuming|ERROR" || true
}

# A_ref re-runs the NLA direction on THIS set, so the comparison is like-for-like
# rather than against Finding 6's different question mix.
run A_ref data/pool/dir_A_1trace.pt
run D_diffmeans data/dom/dir_D_diffmeans.pt
run P_probe     data/dom/dir_P_probe.pt

say "scoring"
uv run python scripts/dom_report.py 2>&1 | tee logs/dom_report.log
say "DOM_STAGEA_DONE"
