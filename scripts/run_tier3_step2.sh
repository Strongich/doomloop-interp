#!/usr/bin/env bash
# TIER 3 STEP 2 — how much accuracy could suppression cost on questions the model
# already solves?  Three extra seeds on Finding 5's own 120 questions, so every
# question is measured at k=4 rollouts instead of 1.
#
#   bash scripts/run_tier3_step2.sh      # ~3.7h, resumable
#
# WHY
# Finding 5 reported 92.5% -> 93.3% with p = 1.00 and a CI of [-3.5, +5.2]pp. That
# is not evidence of no effect; it is an inability to see one. Only 7 of 120 pairs
# disagreed between conditions, because a single rollout per question is a coin
# flip on top of the intervention. Averaging 4 rollouts per question cuts that
# sampling noise and turns the bound from "no drop bigger than ~11pp" into "no
# drop bigger than ~4pp" (simulated, 80% power).
#
# WHAT IT CANNOT DO
# It cannot confirm a GAIN. ~88% of these questions are ones the model answers
# correctly every time, so accuracy has nowhere to go: power against +4pp is 0.04.
# 5b's +4.2pp on this set stays unconfirmed no matter how many seeds are added.
# Demonstrating a gain needs the `mixed` band, which is Tier 3 step 3.
#
# DESIGN
#   set        data/correct120_clean.jsonl -- the 120 held-out questions Finding
#              5b's arms were run on, NOT Finding 5's original 120. Finding 5's
#              set contains 11 questions the A vector was derived from; 5b's set
#              has zero overlap with the derivation pool, and already carries a
#              seed-0 run for BOTH vectors (data/pool/arms/{A_1trace,G_global297}
#              .csv, verified same row order and shared baseline).
#              Frozen to a file so --seed changes ONLY the sampling, never the
#              trace selection: --seed also drives the selection shuffle, so
#              passing it to the original invocation would silently have chosen a
#              different 120 questions.
#   seeds      1, 2, 3 on top of the existing seed 0, giving k=4 per question
#              per vector.
#   vectors    A  data/pool/dir_A_1trace.pt    35 deltas, one correct rollout
#              G  data/delta_suppress_mean.pt  297 deltas, Finding 5's own vector
#   budget     6144, matching Finding 5 (8 of 120 baseline traces capped there;
#              these traces are short, median 1228 think tokens).
#
# The A arm carries `none`; the G arm is suppress-only and reuses that baseline,
# which is valid because trace order is identical across arms and seeds (the file
# holds exactly 120 rows, so the shuffle is a no-op before the length sort).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/tier3 logs
say () { echo "[t3s2 $(date -u +%H:%M:%S)] $*"; }

SET=${SET:-data/correct120_clean.jsonl}
SEEDS=${SEEDS:-"1 2 3"}
ALPHA=${ALPHA:-1.0}
MAX_NEW=${MAX_NEW:-6144}
BATCH=${BATCH:-6}

run () {
  local tag="$1" dir="$2" seed="$3"; shift 3
  local out="data/tier3/step2_${tag}_seed${seed}.csv"
  if [[ -s "$out" ]]; then say "$tag seed $seed already done, skipping"; return 0; fi
  say "$tag seed $seed"
  # shellcheck disable=SC2086
  uv run python scripts/suppress_answer.py \
      --traces "$SET" --limit 120 --alpha "$ALPHA" --batch "$BATCH" \
      --max-new-tokens "$MAX_NEW" --seed "$seed" --direction "$dir" \
      --out "$out" --dump "data/tier3/step2_${tag}_seed${seed}_texts.jsonl" "$@" \
      2>&1 | tee "logs/t3s2_${tag}_seed${seed}.log" \
      | grep -E "correct|median|doubt|capped|resuming|ERROR" || true
}

for s in $SEEDS; do
  run A data/pool/dir_A_1trace.pt      "$s" --conditions none suppress
  run G data/delta_suppress_mean.pt    "$s" --conditions suppress
done

say "scoring"
uv run python scripts/tier3_step2_report.py 2>&1 | tee logs/t3s2_report.log
say "T3S2_DONE"
