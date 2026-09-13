#!/usr/bin/env bash
# Tier 0 + Tier 1: close the vector-construction gaps on correct-only rollouts.
#
#   bash scripts/run_tier01.sh
#
# Every stage is skipped if its output already exists, so an interrupted run
# resumes rather than repeating hours of AV verbalization.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/pool/arms logs
say () { echo "[tier $(date -u +%H:%M:%S)] $*"; }

# --- Tier 0b: is Delta lexical or semantic? (AR only, no generation) ---------
if [[ ! -f data/paraphrase_delta.pt ]]; then
  say "paraphrase test"
  uv run python scripts/paraphrase_delta.py 2>&1 | tee logs/paraphrase.log
fi

# --- Tier 1a: one derivation over 42 traces, raw deltas kept -----------------
if [[ ! -f data/pool/raw.pt ]]; then
  say "deriving raw deltas (the long one, ~1-1.5h)"
  uv run python scripts/derive_pool.py --n-per-dataset 14 --out data/pool/raw.pt \
    2>&1 | tee logs/derive_pool.log
fi

# --- Tier 1b: build the factorial (free) ------------------------------------
say "building variants"
uv run python scripts/build_variants.py --raw data/pool/raw.pt --out-dir data/pool \
  2>&1 | tee logs/variants.log

# --- Tier 1c: held-out test set --------------------------------------------
# The traces the directions were derived from must not be scored on.
if [[ ! -f data/pool/testset.jsonl ]]; then
  uv run python - <<'PY'
import json, torch
raw = torch.load("data/pool/raw.pt", map_location="cpu", weights_only=False)
src = {(m["question_id"], str(m["rollout_index"])) for m in raw["meta"]}
keep = [json.loads(l) for l in open("data/correct_sample.jsonl")]
keep = [t for t in keep if (t["question_id"], str(t["rollout_index"])) not in src]
with open("data/pool/testset.jsonl", "w") as f:
    for t in keep:
        f.write(json.dumps(t) + "\n")
print(f"test set: {len(keep)} traces ({len(src)} source traces held out)")
PY
fi

# --- Tier 1d: generation arms ----------------------------------------------
# Arm A carries none/random; they are identical across arms given the same test
# set and seed, so the rest run suppress-only. That is ~3h instead of ~7h.
ALPHA=${ALPHA:-1.0}
LIMIT=${LIMIT:-120}
COMMON="--traces data/pool/testset.jsonl --limit $LIMIT --alpha $ALPHA --batch 6 \
        --max-new-tokens 6144 --seed 0"

run_arm () {
  local tag="$1" dir="$2"; shift 2
  local out="data/pool/arms/${tag}.csv"
  if [[ -s "$out" ]]; then say "$tag already done"; return 0; fi
  say "arm $tag  ($dir)"
  # shellcheck disable=SC2086
  uv run python scripts/suppress_answer.py $COMMON --direction "$dir" \
      --out "$out" --dump "data/pool/arms/${tag}_texts.jsonl" "$@" \
      2>&1 | tee "logs/arm_${tag}.log" | grep -E "correct|median|doubt|ERROR" || true
}

run_arm A_1trace          data/pool/dir_A_1trace.pt          --conditions none suppress random
run_arm B_spread          data/pool/dir_B_spread.pt          --conditions suppress
run_arm C_firstonly       data/pool/dir_C_firstonly.pt       --conditions suppress
run_arm D_10traces        data/pool/dir_D_10traces.pt        --conditions suppress
run_arm E_30traces        data/pool/dir_E_30traces.pt        --conditions suppress
run_arm F_30traces_simple data/pool/dir_F_30traces_simple.pt --conditions suppress
run_arm G_global297       data/delta_suppress_mean.pt        --conditions suppress

say "TIER01_DONE"
