#!/usr/bin/env bash
# Two-stage delay x alpha sweep. See EXPERIMENT-reasoning-policy.md (kept local).
#
#   STAGE=1  screen every configuration on 200 questions x 1 seed
#   STAGE=2  evaluate the shortlist on 400 questions x 2 seeds
#
# Stage 2 needs SHORTLIST set to the policies stage 1 nominated, because the
# shortlist rule is applied by report_reasoning_policy.py and reviewed by a human
# before any stage-2 token is generated. It is deliberately NOT automatic: an
# unattended pipeline that screens and then commits to its own shortlist removes
# the one checkpoint where a degenerate screen can be caught.
#
#   bash scripts/run_reasoning_policy_sweep.sh
#   STAGE=2 SHORTLIST="N@a0.5d256 N@a1.0d512 D@a0.5d0 D@a1.0d256" \
#       bash scripts/run_reasoning_policy_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE=${STAGE:-1}
OUTROOT=${OUTROOT:-data/reasoning_policy_v1}
BATCH=${BATCH:-64}
CHECKPOINT=${CHECKPOINT:-256}
MAXNEW=${MAXNEW:-16384}
GPUFRAC=${GPUFRAC:-0.90}
BATCHTOK=${BATCHTOK:-4096}

if [ "$STAGE" = "1" ]; then
  COHORT=${COHORT:-data/policy/dev200.jsonl}
  OUTDIR=${OUTDIR:-$OUTROOT/stage1}
  SEEDS=${SEEDS:-1}
  POLICIES="base"
  for d in 0 256 512 1024; do
    for a in 0.1 0.25 0.5 1.0; do
      POLICIES="$POLICIES N@a${a}d${d} D@a${a}d${d}"
    done
  done
elif [ "$STAGE" = "2" ]; then
  COHORT=${COHORT:-data/policy/dev400.jsonl}
  OUTDIR=${OUTDIR:-$OUTROOT/stage2}
  SEEDS=${SEEDS:-2}
  : "${SHORTLIST:?set SHORTLIST to the stage-1 nominated policies}"
  # Brevity prompts are prompt controls and ride along with the shortlist.
  POLICIES="base $SHORTLIST brevityA brevityB"
else
  echo "STAGE must be 1 or 2" >&2; exit 1
fi

echo "stage=$STAGE cohort=$COHORT seeds=$SEEDS out=$OUTDIR"
echo "policies: $POLICIES"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv

exec uv run python scripts/reasoning_policy_vllm.py \
  --cohort "$COHORT" --outdir "$OUTDIR" --policies $POLICIES \
  --seeds "$SEEDS" --max-new-tokens "$MAXNEW" \
  --batch "$BATCH" --checkpoint-every "$CHECKPOINT" \
  --gpu-memory-utilization "$GPUFRAC" --max-num-batched-tokens "$BATCHTOK"
