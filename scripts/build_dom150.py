#!/usr/bin/env python3
"""Build Finding 8's dom150 test set: 50 hard questions per band.

Starts from hard832 and drops every question whose trace is one of the 297 probe
traces that N, D and P were built from. Each band is shuffled with
random.Random(0) and its first 50 questions are kept.

    uv run python scripts/build_dom150.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import random
from pathlib import Path

BANDS = ("all_wrong", "mostly_wrong", "mixed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--hard", type=Path, default=Path("data/hard832.jsonl"))
    p.add_argument("--probes", type=Path, default=Path("data/block_explanations_3way.csv"))
    p.add_argument("--per-band", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("data/dom_testset150.jsonl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with args.probes.open() as fh:
        probe_q = {r["question_id"] for r in csv.DictReader(fh)}
    rows = [json.loads(line) for line in args.hard.open()]
    print(
        f"hard832: {len(rows)}; overlap with the probe traces: "
        f"{sum(1 for r in rows if r['question_id'] in probe_q)}"
    )
    rows = [r for r in rows if r["question_id"] not in probe_q]
    by_band: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_band[r["band"]].append(r)
    rng = random.Random(args.seed)
    out: list[dict] = []
    for band in BANDS:
        v = by_band[band]
        rng.shuffle(v)
        out.extend(v[: args.per_band])
        print(f"  {band:<14}{min(args.per_band, len(v)):>4} of {len(v)} available")
    out.sort(key=lambda r: r["question_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {args.out} ({len(out)} questions)")


if __name__ == "__main__":
    main()
