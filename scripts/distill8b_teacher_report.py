#!/usr/bin/env python3
r"""PROTOCOL v4 §M3 report: 8B teacher accuracy, tokens and capped rate on MATH-train, base vs
NU26@a0.75d256, overall and by level. Flags (for §M11) a steered accuracy more than 1pp below
base. Output: data/distill_8b/teacher_report.json

    uv run python scripts/distill8b_teacher_report.py
"""

from __future__ import annotations

import collections
import csv
import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/distill_8b"


def main() -> None:
    lv = {json.loads(x)["question_id"]: json.loads(x)["level"] for x in open(ROOT / "data/distill/math_train.jsonl")}
    rows = [r for f in sorted(glob.glob(str(D / "gen/shard*/rollouts.csv"))) for r in csv.DictReader(open(f))]
    agg: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        for key in ("all", f"L{lv[r['question_id']]}"):
            a = agg[key][r["policy"]]
            a.append((int(r["correct"]), int(r["total_tokens"]), int(r["capped"])))
    out = {"rollouts": len(rows), "questions": len({r["question_id"] for r in rows})}
    for key in sorted(agg):
        out[key] = {p: {"n": len(v), "acc": round(100 * sum(x[0] for x in v) / len(v), 2),
                        "tokens": round(sum(x[1] for x in v) / len(v), 1),
                        "capped": round(100 * sum(x[2] for x in v) / len(v), 3)}
                    for p, v in sorted(agg[key].items())}
    a = out["all"]
    gap = a["NU26@a0.75d256"]["acc"] - a["base"]["acc"]
    out["steered_minus_base_pp"] = round(gap, 2)
    out["flag_more_than_1pp_below"] = gap < -1
    (D / "teacher_report.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
