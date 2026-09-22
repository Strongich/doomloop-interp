#!/usr/bin/env python3
"""Was the model already done thinking before it started doubting?

Only rollouts that ended with a correct `\\boxed{}` answer, so "done" is
well-defined. Anchors:

  think_close  `h_l` at `</think>` — the model has stopped reasoning and is about
               to state its answer. The cleanest "decision made" state available,
               and unlike the gold-answer string it always exists.
  last         `h_l` at the final token, after the boxed answer is closed.
  doubt_block  the last token of each reasoning block whose *next* block opens
               with "wait" — the model had a finished step in hand and chose to
               second-guess it.

If the doubting were decorative, a doubt_block state should already look like
think_close. Scored against a null of random pairs at the same separation drawn
from inside `<think>`, because raw cosine within one trace is uninterpretable
(arbitrary pairs sit near 0.5).

    uv run python scripts/think_close_similarity.py --traces data/correct_sample.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_attention.config import MODEL_ID, NLAConfig  # noqa: E402
from reasoning_attention.tokenview import (  # noqa: E402
    StateCache,
    baseline,
    block_boundaries,
    build_view,
    cosine,
    think_close_char,
    think_last_content_char,
)


@dataclass
class Ctx:
    """Everything `score` needs, so it is a plain function and not a closure.

    A nested function capturing the loop's `states`/`view` reads whichever trace
    happens to be current when it runs — the same late-binding bug that produced
    all-NaN controls in the trajectory run.
    """

    row: dict[str, Any]
    view: Any
    states: torch.Tensor
    close_idx: int
    out: list[dict[str, Any]]


def score(ctx: Ctx, name: str, i: int, j: int, block_frac: Any) -> None:
    sep = abs(i - j)
    if sep == 0:
        return
    sim = cosine(ctx.states, i, j)
    mu, sd = baseline(ctx.states, sep, region=ctx.view.think)
    ctx.out.append(
        {
            "question_id": ctx.row["question_id"],
            "rollout_index": ctx.row["rollout_index"],
            "dataset": ctx.row["dataset"],
            "n_tokens": ctx.states.shape[0],
            "think_close_token": ctx.close_idx,
            "pair": name,
            "token": i,
            "sep": sep,
            "cosine": round(sim, 4),
            "null_mean": round(mu, 4),
            "null_sd": round(sd, 4),
            "z": round((sim - mu) / sd, 3) if sd > 0 else "",
            "block_frac": block_frac,
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/correct_sample.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/think_close_similarity.csv"))
    p.add_argument("--base", default=MODEL_ID)
    p.add_argument("--marker", default="wait")
    p.add_argument(
        "--max-blocks",
        type=int,
        default=12,
        help="cap doubt blocks scored per trace, so a 200-block looper cannot "
        "dominate the pooled statistics",
    )
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows_in = [json.loads(line) for line in args.traces.open()]
    rows_in = [r for r in rows_in if r["is_correct"]]
    if args.limit:
        rows_in = rows_in[: args.limit]
    print(f"{len(rows_in)} correct traces")

    cache = StateCache(args.base, NLAConfig().extraction_layer)
    out: list[dict[str, Any]] = []
    skipped = 0

    for n, row in enumerate(rows_in, 1):
        text = row["response"]
        close_char = think_close_char(text)
        if close_char is None:
            skipped += 1
            continue
        view = build_view(cache.tokenizer, row["question"], text, row["gold"])
        cache.fill(view)
        states = view.states
        assert states is not None
        n_tok = states.shape[0]

        close_idx = view.char_to_token(close_char)
        content_char = think_last_content_char(text)
        content_idx = view.char_to_token(content_char) if content_char is not None else None
        if close_idx is None or content_idx is None:
            skipped += 1
            view.states = None
            continue
        last_idx = n_tok - 1

        ctx = Ctx(row=row, view=view, states=states, close_idx=close_idx, out=out)
        score(ctx, "think_close->last", last_idx, close_idx, "")
        score(ctx, "think_last->think_close", content_idx, close_idx, "")
        bounds = block_boundaries(text, view.think_chars, args.marker)
        doubt = [b for b in bounds if b[2]][: args.max_blocks]
        plain = [b for b in bounds if not b[2]][: args.max_blocks]
        for end_char, _next_char, is_doubt in doubt + plain:
            idx = view.char_to_token(end_char)
            if idx is None:
                continue
            kind = "doubt" if is_doubt else "plain"
            frac = round(idx / max(content_idx, 1), 3)
            # Both anchors, so the format-token artefact is visible rather than
            # silently baked into the result.
            score(ctx, f"{kind}_block->think_close", idx, close_idx, frac)
            score(ctx, f"{kind}_block->think_last", idx, content_idx, frac)

        print(
            f"  [{n}/{len(rows_in)}] {row['question_id']}#{row['rollout_index']} "
            f"n={n_tok} </think>@{close_idx} blocks={len(bounds)} doubt={len(doubt)}"
        )
        view.states = None
        del states
        torch.cuda.empty_cache()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out[0]))
        writer.writeheader()
        writer.writerows(out)
    print(f"\nwrote {args.out} ({len(out)} rows); skipped {skipped} without </think>\n")

    for pair in (
        "think_close->last",
        "think_last->think_close",
        "doubt_block->think_close",
        "plain_block->think_close",
        "doubt_block->think_last",
        "plain_block->think_last",
    ):
        zs = [float(r["z"]) for r in out if r["pair"] == pair and r["z"] != ""]
        if not zs:
            continue
        pos = sum(1 for z in zs if z > 2)
        neg = sum(1 for z in zs if z < -2)
        t = st.mean(zs) / (st.stdev(zs) / math.sqrt(len(zs))) if len(zs) > 1 else 0.0
        print(
            f"{pair:<26} n={len(zs):<5} mean z {st.mean(zs):+.2f}  "
            f"median {st.median(zs):+.2f}  z>+2: {pos} ({100 * pos / len(zs):.0f}%)  "
            f"z<-2: {neg} ({100 * neg / len(zs):.0f}%)  t={t:+.1f}"
        )


if __name__ == "__main__":
    main()
