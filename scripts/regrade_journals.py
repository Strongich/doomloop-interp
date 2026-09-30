#!/usr/bin/env python3
"""Re-grade saved rollout journals with the current grader (D69: gold canonicalization).

For every rollouts.jsonl matched by the globs: re-grade every row, report flips per
(run, policy) in both directions, and write data/regraded/<run dir>/rollouts.csv, the
original CSV with `correct`, `has_answer` and `status` replaced. Originals are never
modified. Runs are compared with the regraded CSVs by pointing xfer_compare at them.

    uv run python scripts/regrade_journals.py 'data/xfer8b/*confirm*' 'data/eval_math500/**'
"""

from __future__ import annotations

import glob
import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _grade(args: tuple[str, str]) -> tuple[int, int, str]:
    from reasoning_attention.grading import grade

    g = grade(*args)
    return int(g.is_correct), int(g.has_answer), g.status


def main() -> None:
    runs = sorted({str(Path(p).parent) for pat in sys.argv[1:]
                   for p in glob.glob(pat.rstrip("/") + "/rollouts.jsonl", recursive=True)
                   + glob.glob(pat, recursive=True) if p.endswith("rollouts.jsonl")})
    total = Counter()
    with Pool(64) as pool:
        for run in runs:
            rows = [json.loads(line) for line in open(Path(run) / "rollouts.jsonl")]
            new = pool.map(_grade, [(r["text"], r["gold"]) for r in rows], chunksize=16)
            flips = Counter()
            fix = {}
            for r, (c, h, s) in zip(rows, new, strict=True):
                key = (r["question_id"], r["policy"], r["seed"])
                fix[key] = (c, h, s)
                if c != r["correct"]:
                    flips[(r["policy"], f"{r['correct']}->{c}")] += 1
            total.update({k[1]: v for k, v in flips.items()})
            csv = Path(run) / "rollouts.csv"
            if csv.exists():
                df = pd.read_csv(csv)
                keys = list(zip(df.question_id, df.policy, df.seed, strict=True))
                df["correct"] = [fix[k][0] for k in keys]
                df["has_answer"] = [fix[k][1] for k in keys]
                df["status"] = [fix[k][2] for k in keys]
                out = ROOT / "data/regraded" / Path(run).resolve().relative_to(ROOT) / "rollouts.csv"
                out.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(out, index=False)
            print(f"{run}: {len(rows)} rows; flips {dict(flips) or 'none'}", flush=True)
    print("TOTAL flips:", dict(total))


if __name__ == "__main__":
    main()
