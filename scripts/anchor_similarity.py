#!/usr/bin/env python3
"""Is the pre-doubt state already the answer state?

Tests the "decorative thinking" hypothesis representationally: if the model
already holds its answer before it starts second-guessing, then `h_l` just before
the first self-doubt marker should be unusually close to `h_l` where the gold
answer first appears. arXiv:2510.24941 establishes the behavioural version of
this (early-exit answers are stable, backtracking steps score lowest on their
causal TTS metric); this asks whether the *representation* is the same, which
their step-ablation cannot see.

Every pair is scored against a null of random position pairs at the SAME
separation, drawn from inside `<think>` — see `tokenview.baseline`. A raw cosine
is uninterpretable: arbitrary layer-20 pairs inside one trace already sit near
0.5, so only the z-score is evidence.

    uv run python scripts/anchor_similarity.py

Writes data/anchor_similarity.csv (one row per trace per pair).
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_attention.config import MODEL_ID, NLAConfig  # noqa: E402
from reasoning_attention.tokenview import (  # noqa: E402
    StateCache,
    baseline,
    build_view,
    cosine,
)

# Ordered so the earliest anchor comes first in each pair.
PAIRS = [
    ("pre_doubt", "gold_first"),
    ("after_doubts", "gold_first"),
    ("pre_doubt", "end"),
    ("after_doubts", "end"),
    ("gold_first", "end"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/explorer_traces.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/anchor_similarity.csv"))
    p.add_argument("--base", default=MODEL_ID)
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows_in = [json.loads(line) for line in args.traces.open()]
    if args.limit:
        rows_in = rows_in[: args.limit]

    cache = StateCache(args.base, NLAConfig().extraction_layer)
    out: list[dict[str, Any]] = []

    for n, row in enumerate(rows_in, 1):
        view = build_view(cache.tokenizer, row["question"], row["response"], row["gold"])
        cache.fill(view)
        states = view.states
        assert states is not None
        anchors = {a.name: a.token_index for a in view.anchors}
        closed = "</think>" in row["response"]

        for left, right in PAIRS:
            if left not in anchors or right not in anchors:
                continue
            i, j = anchors[left], anchors[right]
            if i == j:
                continue
            sep = abs(i - j)
            sim = cosine(states, i, j)
            mu, sd = baseline(states, sep, region=view.think)
            out.append(
                {
                    "question_id": row["question_id"],
                    "rollout_index": row["rollout_index"],
                    "dataset": row["dataset"],
                    "outcome": row["outcome"],
                    "is_correct": row["is_correct"],
                    "n_tokens": states.shape[0],
                    "think_closed": closed,
                    "pair": f"{left}->{right}",
                    "sep": sep,
                    "cosine": round(sim, 4),
                    "null_mean": round(mu, 4),
                    "null_sd": round(sd, 4),
                    "z": round((sim - mu) / sd, 3) if sd > 0 else "",
                }
            )
        print(
            f"  [{n}/{len(rows_in)}] {row['question_id']}#{row['rollout_index']} "
            f"n={states.shape[0]} anchors={sorted(anchors)}"
        )
        # The states tensor is up to 134 MB; drop it before the next trace.
        view.states = None
        del states
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"\nwrote {args.out} ({len(out)} rows)\n")

    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in out:
        if r["z"] == "":
            continue
        label = "correct" if str(r["is_correct"]).lower() in ("true", "1") else "wrong"
        groups[(r["pair"], label)].append(float(r["z"]))

    print(f"{'pair':<26}{'label':<9}{'n':>4}{'mean z':>9}{'median':>9}{'| z|>2':>8}")
    for (pair, label), zs in sorted(groups.items()):
        big = sum(1 for z in zs if abs(z) > 2)
        med = st.median(zs)
        print(f"{pair:<26}{label:<9}{len(zs):>4}{st.mean(zs):>9.2f}{med:>9.2f}{big:>8}")


if __name__ == "__main__":
    main()
