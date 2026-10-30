"""Reproduce paper policy tables from saved rollouts using the current grader.

Run from the repository root:
    uv run python paper/scripts/regrade_policy_tables.py

Original data and research documents are never changed. F12 preserves the
baseline rows selected by the original merge, updating their scores from the
existing D69 shard CSVs. F13 is regraded directly from its journals. Bootstrap
calculations reuse scripts/alpha_report.py (10,000 question resamples, seed 0).
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import alpha_report as ar
from reasoning_attention.grading import grade

KEY = ["question_id", "policy", "seed"]


def score(row):
    result = grade(row["text"], row["gold"])
    return row["question_id"], row["policy"], row["seed"], int(result.is_correct)


def replace_scores(original, fixes):
    fixes = fixes.drop_duplicates(KEY)
    result = original.drop(columns="correct").merge(
        fixes[KEY + ["correct"]], on=KEY, validate="one_to_one"
    )
    assert len(result) == len(original)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "paper/figures/source_data")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    levels = {r["question_id"]: r["level"] for r in map(
        json.loads, (ROOT / "data/policy/math500.jsonl").open())}
    original = pd.read_csv(ROOT / "data/eval_math500/stage2/rollouts.csv")
    fixes = pd.concat([pd.read_csv(p) for p in sorted(
        (ROOT / "data/regraded/data/eval_math500").glob("stage2_shard*/rollouts.csv"))])
    old = replace_scores(original, fixes)
    old["level"] = old.question_id.map(levels)
    base = old[old.policy == "base"]
    ar.RNG = np.random.default_rng(0)
    output = {"grading": "existing D69 shard scores; original merge baseline retained",
              "bootstrap_replicates": ar.B, "bootstrap_seed": 0,
              "accuracy_scores_changed": int((original.correct.to_numpy() !=
                                               old.correct.to_numpy()).sum()),
              "overall": {}, "levels": {}}
    for policy, arm in old.groupby("policy"):
        output["overall"][policy] = ar.paired(base, arm)
        output["overall"][policy]["mean_tokens"] = float(arm.total_tokens.mean())
    for level in range(1, 6):
        b = base[base.level == level]
        output["levels"][str(level)] = {}
        for policy in ("N@a1.0d256", "D@a0.5d512"):
            arm = old[(old.policy == policy) & (old.level == level)]
            output["levels"][str(level)][policy] = ar.paired(b, arm)
    (args.output / "policy_math500_regraded.json").write_text(
        json.dumps(output, indent=2, default=float) + "\n")
    rows = []
    for path in sorted((ROOT / "data/eval_math500_alpha").glob("stage2_shard*/rollouts.jsonl")):
        rows.extend(map(json.loads, path.open()))
    with Pool(args.workers) as pool:
        fixes = pd.DataFrame(pool.map(score, rows, chunksize=8), columns=KEY + ["correct"])
    original_alpha = pd.read_csv(ROOT / "data/eval_math500_alpha/stage2/rollouts.csv")
    new = replace_scores(original_alpha, fixes)
    print("F13 changed accuracy scores:", int((original_alpha.correct.to_numpy() !=
                                                new.correct.to_numpy()).sum()))
    with tempfile.TemporaryDirectory(prefix="vers-policy-") as temp:
        old_path, new_path = Path(temp) / "old.csv", Path(temp) / "new.csv"
        old.to_csv(old_path, index=False)
        new.to_csv(new_path, index=False)
        subprocess.run([sys.executable, str(ROOT / "scripts/alpha_report.py"),
                        "--old", str(old_path), "--new", str(new_path),
                        "--cohort", str(ROOT / "data/policy/math500_L45.jsonl"),
                        "--out", str(args.output / "policy_alpha_regraded.json")], check=True)


if __name__ == "__main__":
    main()
