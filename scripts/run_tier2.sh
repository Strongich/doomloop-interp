#!/usr/bin/env bash
# TIER 2 — does suppressing self-doubt still hold up on questions the model gets WRONG?
#
#   bash scripts/run_tier2.sh              # ~9-12h on one 16GB GPU (6 generation passes)
#   LIMIT=60 bash scripts/run_tier2.sh     # quick shakeout first
#
# Every stage is skipped if its output exists, so an interrupted run resumes.
# Nothing here needs the pod: it is one local GPU, no network, no API.
#
# ---------------------------------------------------------------------------
# WHY THIS EXISTS
#
# Findings 5 and 5a-5d are all measured on traces the model ALREADY answers
# correctly (92.5% accuracy, 93% GSM8K). They therefore cannot see the one case
# that would overturn them: a question where the model's first pass is wrong and
# the self-doubt is what RESCUES it. FINDINGS.md names this as the single biggest
# hole in Finding 5. This run fills it.
#
# The test set is deliberately built so accuracy can move in BOTH directions.
# Finding 5 sat at a 92.5% ceiling where a drop was almost the only detectable
# outcome; here the baseline is ~30%, so a rise is just as visible as a fall.
#
# THREE OUTCOMES, ALL WORTH HAVING
#   accuracy holds  -> the doubt is decorative everywhere, and distilling the
#                      steered behaviour into the weights is safe.
#   accuracy drops  -> the doubt is load-bearing exactly where problems are hard.
#                      That reverses the "overthinking" story, explains why the
#                      model doubts at all, and makes CONDITIONAL suppression the
#                      goal -- which a fixed direction cannot do and a policy can.
#   accuracy rises  -> the doubt actively hurts on hard problems. Strongest result
#                      of the three.
#
# ---------------------------------------------------------------------------
# DESIGN
#
# Test set   data/hard_sample_heldout.jsonl -- 235 questions.
#            Built from all 35,728 rollouts by grouping per question and keeping
#            those the model does NOT reliably solve, stratified into three bands
#            of ~80: all_wrong (0% of its rollouts correct), mostly_wrong (<50%),
#            mixed (50-99%). Mean historical accuracy 0.302. The representative
#            rollout is wrong in every one of them. 231 gsm8k / 2 amc23 / 2 aime.
#            5 questions were dropped because they overlap the traces the
#            direction was derived from -- steering a trace you derived from is
#            not a test.
#
# Vectors    A  data/pool/dir_A_1trace.pt   35 deltas from ONE rollout
#               (aime2025:1 rollout 1), normalized mean. Finding 5b showed this
#               matches the 297-probe vector on easy questions. Note it is an
#               AIME-derived vector steering a 98%-GSM8K test set: that is a
#               cross-domain test on purpose, and 5a found cross-domain transfer
#               works at least as well as same-domain.
#            G  data/delta_suppress_mean.pt  the 297-probe global vector from
#               Finding 5, for continuity -- if A and G disagree HERE but agreed
#               on easy questions, that is itself a finding.
#            W  data/pool_wrong/dir_W_wrong1trace.pt   ~35 deltas from ONE rollout
#               the model got WRONG. Built exactly like A, so the only difference
#               is the outcome of the trace it was read off.
#            WP data/pool_wrong/dir_WP_wrong30traces.pt  all deltas from 30 wrong
#               rollouts, built exactly like Finding 5b's arm E. W on its own
#               cannot separate "derived from a failure" from "derived from that
#               one rollout"; WP is what makes the comparison interpretable.
#
#            A and G are read off traces the model got RIGHT -- as was every
#            direction in the study so far. If self-doubt represents something
#            different when the model is genuinely lost, a direction derived from
#            failures could point somewhere else, and nothing so far would have
#            caught it. The derivation stage prints the cosines between the two
#            families before any generation runs: that number alone answers the
#            cheap half of the question.
#
# Conditions none / suppress / random, all three. The baseline cannot be reused
#            from Finding 5 because this is a different test set, and sampling is
#            stochastic. `random` is a matched-norm meaningless push at the same
#            positions: it controls for "editing the hidden state at 36 places
#            damages the model", which is the deflationary reading of Finding 5.
#
# alpha      1.0 -- the best setting in Finding 5 (shortest reasoning at equal
#            accuracy) and the one used throughout 5b.
#
# Budget     MAX_NEW=12288, NOT Finding 5's 6144. This is the one parameter that
#            would quietly rig the result. Wrong rollouts are long: median ~4.5k
#            tokens, p90 ~9.2k. At a 6144 cap roughly a third of the BASELINE
#            would be truncated -- while suppression, which halves trace length,
#            would rarely cap at all. Suppression would then score better on
#            `answered` and on accuracy purely because it fits the budget.
#            12288 covers p90 at about 2x the cost. Cap rates are reported per
#            condition so the residual effect stays visible.
#
# READOUTS   accuracy overall AND per band; median thinking tokens; doubt blocks
#            per answer; answered rate; capped rate. Paired McNemar plus a 95% CI
#            for each suppress-vs-none comparison, repeated within each band.
#
#            Per-band is the point. If doubt rescues only genuinely hard
#            questions, it shows up in all_wrong and is washed out in the average.
#
# EXPECTED POWER  At a ~30% baseline, far more traces disagree between conditions
#            than at Finding 5's 92.5% ceiling (where only 7 of 120 did, giving a
#            useless +-4.6pp CI). Expect 40-70 discordant pairs and a CI nearer
#            +-5pp on 235 questions -- and, unlike Finding 5, a genuine ability to
#            detect a drop.
#
# WHAT THIS STILL WILL NOT ANSWER
#   - Whether suppression helps if applied only at SOME boundaries. It is all or
#     nothing at every break, as in Finding 5.
#   - Anything about a prompt baseline, which remains unrun.
# ---------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/tier2 logs
say () { echo "[tier2 $(date -u +%H:%M:%S)] $*"; }

TRACES=${TRACES:-data/hard_sample_heldout.jsonl}
LIMIT=${LIMIT:-235}
ALPHA=${ALPHA:-1.0}
MAX_NEW=${MAX_NEW:-12288}
BATCH=${BATCH:-4}          # lower than Finding 5's 6: traces here run much longer

if [[ ! -f "$TRACES" ]]; then
  echo "ERROR: $TRACES missing. Rebuild it with the snippet in FINDINGS 'Reproducing'." >&2
  exit 1
fi

COMMON="--traces $TRACES --limit $LIMIT --alpha $ALPHA --batch $BATCH \
        --max-new-tokens $MAX_NEW --seed 0"

run_arm () {
  local tag="$1" dir="$2"; shift 2
  local out="data/tier2/${tag}.csv"
  if [[ -s "$out" ]]; then say "$tag already done, skipping"; return 0; fi
  say "arm $tag  ($dir)"
  # shellcheck disable=SC2086
  uv run python scripts/suppress_answer.py $COMMON --direction "$dir" \
      --out "$out" --dump "data/tier2/${tag}_texts.jsonl" "$@" \
      2>&1 | tee "logs/tier2_${tag}.log" | grep -E "correct|median|doubt|capped|ERROR" || true
}

# --- derive the wrong-trace directions (GPU, ~1h, skipped if already built) ---
if [[ ! -s data/pool_wrong/dir_W_wrong1trace.pt ]]; then
  say "deriving directions from WRONG rollouts"
  [[ -s data/wrong_sample.jsonl ]] || \
    uv run python scripts/build_wrong_sample.py 2>&1 | tee logs/wrong_sample.log
  [[ -s data/pool_wrong/raw.pt ]] || \
    uv run python scripts/derive_pool.py --traces data/wrong_sample.jsonl \
        --out data/pool_wrong/raw.pt --n-per-dataset 14 2>&1 | tee logs/derive_pool_wrong.log
  uv run python scripts/build_wrong_variants.py 2>&1 | tee logs/build_wrong_variants.log
fi

# Arm A carries none + random; they are identical across arms given the same test
# set and seed, so every other arm runs suppress-only. Saves roughly two hours
# per arm.
run_arm A_1trace     data/pool/dir_A_1trace.pt              --conditions none suppress random
run_arm G_global297  data/delta_suppress_mean.pt            --conditions suppress
run_arm W_wrong1     data/pool_wrong/dir_W_wrong1trace.pt   --conditions suppress
run_arm WP_wrong30   data/pool_wrong/dir_WP_wrong30traces.pt --conditions suppress

say "scoring"
uv run python scripts/tier2_report.py 2>&1 | tee logs/tier2_report.log
say "TIER2_DONE"
