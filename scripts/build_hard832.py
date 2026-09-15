#!/usr/bin/env python3
"""Build Tier 3 step 1's confirmatory set: every untested hard question.

A question is "hard" if the model did not solve all 4 of its recorded rollouts.
Bands: all_wrong (0/4), mostly_wrong (<50%), mixed (50-99%).

Excluded:
  * the 235 questions already measured in Tier 2 -- they selected the vector, so
    testing on them again would not be confirmation
  * every question any suppression direction was derived from (both delta pools),
    since steering a trace you read the direction off is not a test

One row per question: the representative rollout (first wrong one if any, else
the first), used for the question text and for length-sorted batching. The
response is regenerated at eval time, never read off this file.

    uv run python scripts/build_hard832.py
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/traces_all.jsonl"))
    p.add_argument("--tested", type=Path, default=Path("data/hard_sample_heldout.jsonl"))
    p.add_argument("--pools", type=Path, nargs="+",
                   default=[Path("data/pool/raw.pt"), Path("data/pool_wrong/raw.pt")])
    p.add_argument("--out", type=Path, default=Path("data/hard832.jsonl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tested = {json.loads(line)["question_id"] for line in args.tested.open()}
    derived: set[str] = set()
    for p in args.pools:
        if p.exists():
            meta = torch.load(p, map_location="cpu", weights_only=False)["meta"]
            derived |= {m["question_id"] for m in meta}
    print(f"excluding {len(tested)} already-tested and {len(derived)} derivation questions")

    # traces_all.jsonl carries no gold answer; recover it from the dataset by the
    # index encoded in question_id ("gsm8k:282"). Verified against the 235 golds
    # in the Tier 2 set: 200/200 sampled matched exactly.
    from reasoning_attention.data.math_datasets import load_all

    ds = load_all()

    def gold_of(qid: str) -> str:
        src, idx = qid.split(":")
        return str(ds[src][int(idx)]["answer"])

    rolls: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for line in args.traces.open():
        r = json.loads(line)
        rolls[r["question_id"]].append(r)

    out: list[dict[str, Any]] = []
    skipped = collections.Counter()
    for q, rs in rolls.items():
        n = len(rs)
        c = sum(r["is_correct"] for r in rs)
        acc = c / n
        if acc == 1.0 or n < 2:
            continue
        if q in tested:
            skipped["already tested"] += 1
            continue
        if q in derived:
            skipped["derivation source"] += 1
            continue
        band = "all_wrong" if acc == 0 else "mostly_wrong" if acc < 0.5 else "mixed"
        rep = next((r for r in rs if not r["is_correct"]), rs[0])
        out.append(
            {
                "question_id": q,
                "dataset": rep["dataset"],
                "question": rep["question"],
                "gold": gold_of(q),
                "response": rep["response"],
                "rollout_index": rep["rollout_index"],
                "is_correct": rep["is_correct"],
                "band": band,
                "hist_correct": c,
                "hist_total": n,
                "hist_acc": acc,
            }
        )
    for k, v in skipped.items():
        print(f"  skipped {v} ({k})")
    out.sort(key=lambda r: (r["band"], r["question_id"]))
    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"\nwrote {len(out)} questions -> {args.out}")
    for b, n in sorted(collections.Counter(r["band"] for r in out).items()):
        ds = collections.Counter(r["dataset"] for r in out if r["band"] == b)
        print(f"  {b:<14}{n:>5}   {dict(ds)}")
    print(f"  mean historical accuracy {sum(r['hist_acc'] for r in out)/len(out):.3f}")


if __name__ == "__main__":
    main()
