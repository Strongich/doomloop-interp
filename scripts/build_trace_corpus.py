#!/usr/bin/env python3
"""Flatten our sampled rollouts into one jsonl for RL activation extraction.

The Stage-2 corpus needs text, not labels — the AV generates the explanation
during rollout and the AR scores it — so this only has to emit question +
response per rollout.

    uv run python scripts/build_trace_corpus.py --out data/traces_all.jsonl

Two decisions worth knowing:

**All rollouts, not one per question.** More positions from the same problem are
fine here: unlike Stage-1, RL has no disjoint AV/AR split to protect (both halves
see the same rollout — the AV verbalizes `h_l`, the AR reconstructs it), so
near-duplicate rollouts cannot leak across a boundary that does not exist. Pass
`--one-per-question` to take a single rollout each anyway.

**Failures are kept, not filtered.** 92.1% of our rollouts answer correctly, but
the study's whole contrast is recovered-versus-failed, and an NLA that has never
seen a derailed activation will confabulate hardest exactly where it is most
needed. `--drop-incorrect` exists but defaults off.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/traces"))
    p.add_argument("--datasets", nargs="+", default=["aime2025", "amc23", "gsm8k"])
    p.add_argument("--out", type=Path, default=Path("data/traces_all.jsonl"))
    p.add_argument("--one-per-question", action="store_true")
    p.add_argument("--drop-incorrect", action="store_true")
    p.add_argument(
        "--min-chars",
        type=int,
        default=400,
        help="skip very short responses: too little context for an activation to mean much",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seen_questions: set[str] = set()
    stats: collections.Counter[str] = collections.Counter()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    with args.out.open("w") as out:
        for name in args.datasets:
            path = args.traces / f"{name}.jsonl"
            if not path.exists():
                print(f"  skip {name}: {path} missing")
                continue
            with path.open() as fh:
                for line in fh:
                    row: dict[str, Any] = json.loads(line)
                    stats["total"] += 1
                    if args.drop_incorrect and not row["is_correct"]:
                        stats["dropped_incorrect"] += 1
                        continue
                    if len(row["response"]) < args.min_chars:
                        stats["dropped_short"] += 1
                        continue
                    if args.one_per_question:
                        if row["question_id"] in seen_questions:
                            continue
                        seen_questions.add(row["question_id"])
                    out.write(
                        json.dumps(
                            {
                                "question": row["question"],
                                "response": row["response"],
                                "question_id": row["question_id"],
                                "rollout_index": row["rollout_index"],
                                "dataset": row["dataset"],
                                "outcome": row["outcome"],
                                "is_correct": row["is_correct"],
                            }
                        )
                        + "\n"
                    )
                    stats[name] += 1
                    stats["kept"] += 1

    print(f"wrote {args.out}")
    print(f"  {stats['kept']:,} rollouts of {stats['total']:,}")
    for name in args.datasets:
        if stats[name]:
            print(f"    {name:<10} {stats[name]:>7,}")
    for key in ("dropped_incorrect", "dropped_short"):
        if stats[key]:
            print(f"  {key}: {stats[key]:,}")


if __name__ == "__main__":
    main()
