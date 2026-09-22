#!/usr/bin/env python3
"""Is a shorter continuation a less repetitive one? Measure it; don't assume it.

RESEARCH-DIRECTION.md's caution applies directly: fewer tokens alone do not
establish greater coherence. These are the behaviour-based measures from
`loop_metrics.py`, applied to the CONTINUATION only -- the prefix is shared by
every arm, so including it would dilute every difference identically.

Also writes a blinded sample for manual annotation: arm labels are replaced by
opaque ids and the key is kept in a separate file, so reading the traces cannot
be biased by knowing which intervention produced them.

    uv run python scripts/prefix_loops.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import random
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from loop_metrics import dup_blocks, rep_n, think  # noqa: E402


def gz_ratio(body: str) -> float:
    raw = body.encode()
    return len(gzip.compress(raw)) / max(len(raw), 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--texts", type=Path, default=Path("data/prefix/branches_texts.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/prefix/loops.csv"))
    ap.add_argument("--blind", type=Path, default=Path("data/prefix/blind_sample.md"))
    ap.add_argument("--key", type=Path, default=Path("data/prefix/blind_key.json"))
    ap.add_argument("--blind-n", type=int, default=30, help="questions in the blinded set")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(x) for x in args.texts.open()]
    by_arm: dict[str, list[dict[str, float]]] = defaultdict(list)
    by_q: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        # The continuation has no opening tag; prepend one so `think()` scopes to
        # post-freeze reasoning and stops at the model's own </think>.
        body = think("<think>\n" + r["text"])
        words = body.split()
        m = {
            "rep20": rep_n(words, 20) if len(words) > 20 else 0.0,
            "dupblk": dup_blocks(body),
            "gzip": gz_ratio(body) if body else 0.0,
            "words": float(len(words)),
        }
        by_arm[r["arm"]].append(m)
        by_q[r["question_id"]][r["arm"]].append(r)

    print(f"{len(rows)} continuations\n")
    print(f"{'arm':6s} {'n':>5s} {'rep20':>8s} {'dupblk':>8s} {'gzip':>7s} {'words':>8s}")
    for a in sorted(by_arm):
        v = by_arm[a]
        print(f"{a:6s} {len(v):5d} " + " ".join(
            f"{st.median([x[k] for x in v]):8.3f}" for k in ("rep20", "dupblk", "gzip"))
            + f" {st.median([x['words'] for x in v]):8.0f}")

    with args.out.open("w") as fh:
        fh.write("arm,rep20,dupblk,gzip,words\n")
        for a, v in by_arm.items():
            for x in v:
                fh.write(f"{a},{x['rep20']:.4f},{x['dupblk']:.4f},{x['gzip']:.4f},"
                         f"{x['words']:.0f}\n")

    # Blinded annotation set: one seed-0 continuation per arm per question.
    rng = random.Random(args.seed)
    qs = sorted(q for q, d in by_q.items() if len(d) >= 2)
    rng.shuffle(qs)
    key: dict[str, str] = {}
    with args.blind.open("w") as fh:
        fh.write("# Blinded continuations\n\nFor each question the same prefix was "
                 "continued several ways. Rate each continuation for repeated "
                 "derivations and coherence WITHOUT knowing which arm produced it; "
                 "the mapping is in blind_key.json.\n\n")
        for n, q in enumerate(qs[: args.blind_n]):
            fh.write(f"## Question {n + 1} (`{q}`)\n\n")
            items = [(a, v[0]) for a, v in sorted(by_q[q].items()) if v]
            rng.shuffle(items)
            for j, (arm, r) in enumerate(items):
                tag = f"Q{n + 1}{chr(ord('A') + j)}"
                key[tag] = arm
                fh.write(f"### {tag}\n\n```\n{r['text'][:4000]}\n```\n\n")
    args.key.write_text(json.dumps(key, indent=1))
    print(f"\nwrote {args.out}, {args.blind} ({len(qs[:args.blind_n])} questions), {args.key}")


if __name__ == "__main__":
    main()
