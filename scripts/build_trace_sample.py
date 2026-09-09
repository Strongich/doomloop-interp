#!/usr/bin/env python3
"""Pull a balanced, question-matched subset of traces into one small jsonl.

Analyses that need a forward pass per trace should not rescan a 568 MB
gsm8k.jsonl; and they should not compare a random handful of correct traces
against a random handful of wrong ones, because problem difficulty then differs
between the arms. This writes matched pairs: for every question where some
rollouts answered correctly and some did not, one rollout of each.

    uv run python scripts/build_trace_sample.py --out data/anchor_sample.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

KEEP = (
    "question_id",
    "rollout_index",
    "dataset",
    "question",
    "gold",
    "response",
    "outcome",
    "is_correct",
    "exited_think",
    "stop_reason",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/traces"))
    p.add_argument("--datasets", nargs="+", default=["aime2025", "amc23", "gsm8k"])
    p.add_argument("--out", type=Path, default=Path("data/anchor_sample.jsonl"))
    p.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="cap the number of questions, spread evenly over the datasets",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--only-correct",
        action="store_true",
        help="drop the pairing and emit correct rollouts only, sampled over all "
        "questions rather than only the mixed-outcome ones",
    )
    return p.parse_args()


def _write_correct_only(
    args: argparse.Namespace,
    rng: random.Random,
    by_q: dict[str, dict[bool, list[dict[str, Any]]]],
) -> None:
    """One correct rollout per question, round-robin across datasets."""
    per_dataset: dict[str, list[str]] = defaultdict(list)
    for q, sides in by_q.items():
        if sides.get(True):
            per_dataset[q.split(":", 1)[0]].append(q)
    for pool in per_dataset.values():
        rng.shuffle(pool)

    buckets = [per_dataset[d] for d in args.datasets if per_dataset.get(d)]
    cap = args.max_pairs or sum(len(b) for b in buckets)
    chosen: list[str] = []
    while len(chosen) < cap and any(buckets):
        for bucket in buckets:
            if bucket and len(chosen) < cap:
                chosen.append(bucket.pop())
        buckets = [b for b in buckets if b]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for q in chosen:
            fh.write(json.dumps(rng.choice(by_q[q][True])) + "\n")
    counts: dict[str, int] = defaultdict(int)
    for q in chosen:
        counts[q.split(":", 1)[0]] += 1
    print(f"correct-only: {len(chosen)} traces, per dataset {dict(counts)}")
    print(f"wrote {args.out}")


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    by_q: dict[str, dict[bool, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))

    for name in args.datasets:
        path = args.traces / f"{name}.jsonl"
        if not path.exists():
            print(f"  skip {name}: missing")
            continue
        with path.open() as fh:
            for line in fh:
                row = json.loads(line)
                by_q[row["question_id"]][bool(row["is_correct"])].append({k: row[k] for k in KEEP})

    if args.only_correct:
        _write_correct_only(args, rng, by_q)
        return

    mixed = [q for q, sides in by_q.items() if sides.get(True) and sides.get(False)]
    per_dataset: dict[str, list[str]] = defaultdict(list)
    for q in mixed:
        per_dataset[q.split(":", 1)[0]].append(q)
    for pool in per_dataset.values():
        rng.shuffle(pool)

    # Round-robin so the small datasets are represented at all: GSM8K holds 97%
    # of the mixed-outcome questions.
    chosen: list[str] = []
    buckets = [per_dataset[d] for d in args.datasets if per_dataset.get(d)]
    cap = args.max_pairs or sum(len(b) for b in buckets)
    while len(chosen) < cap and any(buckets):
        for bucket in buckets:
            if bucket and len(chosen) < cap:
                chosen.append(bucket.pop())
        buckets = [b for b in buckets if b]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for q in chosen:
            for label in (True, False):
                fh.write(json.dumps(rng.choice(by_q[q][label])) + "\n")

    counts: dict[str, int] = defaultdict(int)
    for q in chosen:
        counts[q.split(":", 1)[0]] += 1
    print(f"{len(mixed)} mixed-outcome questions; kept {len(chosen)} pairs")
    print(f"  per dataset: {dict(counts)}")
    print(f"wrote {args.out} ({2 * len(chosen)} traces)")


if __name__ == "__main__":
    main()
