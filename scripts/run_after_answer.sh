#!/usr/bin/env bash
# Post-answer suppression. Inject only once the gold value has been written.
#
#   bash scripts/run_after_answer.sh          # ~6h, resumable at batch level
#
# Design and rationale: EXPERIMENT-after-answer.md
#
# 400 questions, 100 per band, including all_right for the first time -- that is
# where the answer is most reliably stated, so the trigger fires most often.
# Three arms: baseline, NLA direction, difference-of-means. The baseline arm runs
# through the same script so it records trigger position and post-answer share on
# identical questions.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/afteranswer logs
say () { echo "[aa $(date -u +%H:%M:%S)] $*"; }

SET=${SET:-data/after_answer_400.jsonl}
LIMIT=${LIMIT:-400}
BATCH=${BATCH:-4}
MAX_NEW=${MAX_NEW:-12288}

run () {
  local tag="$1" cond="$2" dir="$3"
  local out="data/afteranswer/${tag}.csv"
  if [[ -s "$out" ]]; then say "$tag done, skipping"; return 0; fi
  say "arm $tag"
  uv run python scripts/suppress_after_answer.py \
      --traces "$SET" --limit "$LIMIT" --batch "$BATCH" --max-new-tokens "$MAX_NEW" \
      --cond "$cond" --direction "$dir" \
      --out "$out" --dump "data/afteranswer/${tag}_texts.jsonl" \
      2>&1 | tee "logs/aa_${tag}.log" | grep -E "correct|triggered|resuming|wrote|ERROR" || true
}

run baseline none        data/pool/dir_A_1trace.pt
run N_after  after       data/pool/dir_A_1trace.pt
run D_after  after       data/dom/dir_D_diffmeans.pt

say "scoring"
uv run python scripts/after_answer_report.py 2>&1 | tee logs/aa_report.log
say "AA_DONE"
