#!/usr/bin/env python3
"""EXPERIMENT-vers-doubt.md §3: pick α for P on dev400 (base re-run in the same engine).

The LARGEST α whose dev arm has
  1. doubt paragraphs per rollout above base (question-clustered 95% CI excludes 0)
  2. Δacc vs base ≥ −1pp
  3. capped rate ≤ max(2 × base, base + 1pp)
Writes data/vers/selected.txt with the policy name, or NONE (reading M0).
D69 grading from rollouts.jsonl; bootstrap 2,000 reps over questions, seed 20261005.

    python scripts/vers_select.py --dev data/vers/dev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_attention.grading import grade  # noqa: E402

REPS, SEED = 2000, 20261005
GRID = ["P@a0.25d256", "P@a0.5d256", "P@a1.0d256"]


def load(path: Path) -> pd.DataFrame:
    rows = []
    for line in path.open():
        r = json.loads(line)
        rows.append({"question_id": r["question_id"], "policy": r["policy"],
                     "seed": int(r["seed"]), "correct": int(grade(r["text"], r["gold"]).is_correct),
                     "capped": int(r["capped"]), "doubt_blocks": int(r["doubt_blocks"]),
                     "total_tokens": int(r["total_tokens"])})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, default=Path("data/vers/dev"))
    ap.add_argument("--out", type=Path, default=Path("data/vers/selected.txt"))
    args = ap.parse_args()
    df = load(args.dev / "rollouts.jsonl")
    base = df[df.policy == "base"]
    qs = sorted(base.question_id.unique())
    idx = np.random.default_rng(SEED).integers(0, len(qs), (REPS, len(qs)))
    report, selected = {}, "NONE"
    for pol in GRID:
        arm = df[df.policy == pol]
        m = arm.merge(base, on=["question_id", "seed"], suffixes=("", "_b"))
        assert len(m) == len(base) == len(arm), (pol, len(m), len(base), len(arm))
        qm = m.groupby("question_id").mean(numeric_only=True).reindex(qs)
        dd = (qm.doubt_blocks - qm.doubt_blocks_b).to_numpy()
        dboot = dd[idx].mean(1)
        dacc = (qm.correct - qm.correct_b).to_numpy() * 100
        cap, capb = m.capped.mean() * 100, m.capped_b.mean() * 100
        r = {
            "doubt": m.doubt_blocks.mean(), "base_doubt": m.doubt_blocks_b.mean(),
            "ddoubt": dd.mean(), "ddoubt_ci": [float(np.percentile(dboot, 2.5)),
                                               float(np.percentile(dboot, 97.5))],
            "acc": m.correct.mean() * 100, "base_acc": m.correct_b.mean() * 100,
            "dacc": dacc.mean(), "dacc_ci": [float(np.percentile(dacc[idx].mean(1), q))
                                             for q in (2.5, 97.5)],
            "capped": cap, "base_capped": capb,
            "length": m.total_tokens.sum() / m.total_tokens_b.sum() - 1,
        }
        r["c1"] = r["ddoubt_ci"][0] > 0
        r["c2"] = r["dacc"] >= -1.0
        r["c3"] = cap <= max(2 * capb, capb + 1.0)
        r["pass"] = r["c1"] and r["c2"] and r["c3"]
        if r["pass"]:
            selected = pol  # GRID is ascending in α, so the last pass is the largest
        report[pol] = r
        print(f"{pol:<13} doubt {r['doubt']:.2f} vs {r['base_doubt']:.2f} "
              f"Δ {r['ddoubt']:+.2f} [{r['ddoubt_ci'][0]:+.2f},{r['ddoubt_ci'][1]:+.2f}] | "
              f"acc {r['acc']:.1f} vs {r['base_acc']:.1f} Δ {r['dacc']:+.2f}pp | "
              f"capped {cap:.2f}% vs {capb:.2f}% | len {r['length']:+.1%} | "
              f"c1 {r['c1']} c2 {r['c2']} c3 {r['c3']}")
    print(f"SELECTED {selected}")
    (args.out.parent / "select.json").write_text(json.dumps(report, indent=2, default=float))
    args.out.write_text(selected + "\n")


if __name__ == "__main__":
    main()
