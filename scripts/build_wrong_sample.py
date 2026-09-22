#!/usr/bin/env python3
"""Collect WRONG rollouts to derive a suppression direction from.

Every direction the study has used so far — the 297-probe global vector, arm A,
every variant in Finding 5b — was derived from traces the model got RIGHT. So
"suppression works" has only ever been demonstrated with a vector read off
successful reasoning. If doubt means something different when the model is
actually lost, a direction derived from failures could behave differently, and
nothing so far would have noticed.

This builds the derivation set for that arm: one rollout per question, wrong
answer, and **disjoint from the Tier 2 test set** — steering a trace you derived
from is not a test.

    uv run python scripts/build_wrong_sample.py
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/traces_all.jsonl"))
    p.add_argument("--exclude", type=Path, default=Path("data/hard_sample_heldout.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/wrong_sample.jsonl"))
    p.add_argument("--per-dataset", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    held = {json.loads(line)["question_id"] for line in args.exclude.open()}
    print(f"excluding {len(held)} held-out questions")

    seen: set[str] = set()
    pool: dict[str, list[dict]] = collections.defaultdict(list)
    for line in args.traces.open():
        r = json.loads(line)
        if r["is_correct"] or r["question_id"] in held or r["question_id"] in seen:
            continue
        seen.add(r["question_id"])          # one rollout per question
        pool[r["dataset"]].append(r)

    rng = random.Random(args.seed)
    out = []
    for ds in sorted(pool):
        rows = pool[ds]
        rng.shuffle(rows)
        out.extend(rows[: args.per_dataset])
        print(f"  {ds:<9} {len(rows[: args.per_dataset]):>4} of {len(rows)} wrong questions")

    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} wrong traces -> {args.out}")


if __name__ == "__main__":
    main()
