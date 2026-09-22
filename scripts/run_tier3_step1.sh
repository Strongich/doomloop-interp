#!/usr/bin/env bash
# TIER 3 STEP 1 — the confirmatory run. Vector A on every untested hard question.
#
#   bash scripts/run_tier3_step1.sh        # ~17h, resumable at batch granularity
#
# WHY
# Tier 2 (n=235) measured +7.2pp, 95% CI [+0.8, +13.6], p=0.037 -- and it also
# chose the vector. A result that both selects and tests on the same data is a
# pilot, not a finding. This re-tests the SAME pre-specified vector on questions
# that took no part in choosing it.
#
# PRE-SPECIFIED (see TIER3-PREREG.md, written before launch)
#   vector     A -- 35 deltas from ONE correct rollout (aime2025:1 r1), all
#              depths, normalized mean. Chosen because it was the best arm on two
#              independent prior sets: 5b's easy set (96.7%, best of seven) and
#              Tier 2's hard set (+7.2pp, best of four).
#   primary    accuracy change across all 789 questions, Cochran-Mantel-Haenszel
#              stratified by band. Secondary: per-band McNemar, length sign test.
#   alpha 1.0, MAX_NEW 12288, batch 4, seed 0, conditions none + suppress.
#
# `random` is dropped: it came in at -0.9pp (p=0.90) in Tier 2 and costs a third
# of the compute. Its job -- ruling out "any perturbation helps" -- is done.
#
# SET  data/hard832.jsonl, 789 questions (292 all_wrong / 88 mostly_wrong /
#      409 mixed), mean historical accuracy 0.374. Excludes the 235 Tier 2
#      questions and the 43 questions any delta pool was derived from.
#
# POWER  0.997 if the true effect matches the pilot (+8.3pp); 0.95 at +6.3pp;
#      0.66 at +4.2pp. So: powered for the pilot effect, not a guarantee.
#      The `mixed` band is load-bearing -- without it, power falls to 0.90.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/tier3 logs
say () { echo "[t3s1 $(date -u +%H:%M:%S)] $*"; }

SET=${SET:-data/hard832.jsonl}
LIMIT=${LIMIT:-789}
ALPHA=${ALPHA:-1.0}
MAX_NEW=${MAX_NEW:-12288}
BATCH=${BATCH:-4}
OUT=data/tier3/step1_A.csv

if [[ -s "$OUT" ]]; then
  say "$OUT already complete"
else
  say "vector A on $LIMIT questions (resumes from ${OUT}.partial if present)"
  uv run python scripts/suppress_answer.py \
      --traces "$SET" --limit "$LIMIT" --alpha "$ALPHA" --batch "$BATCH" \
      --max-new-tokens "$MAX_NEW" --seed 0 \
      --direction data/pool/dir_A_1trace.pt \
      --conditions none suppress \
      --out "$OUT" --dump data/tier3/step1_A_texts.jsonl \
      2>&1 | tee logs/t3s1_A.log \
      | grep -E "correct|median|doubt|capped|resuming|ERROR" || true
fi

say "scoring"
uv run python scripts/tier3_step1_report.py 2>&1 | tee logs/t3s1_report.log
say "T3S1_DONE"
