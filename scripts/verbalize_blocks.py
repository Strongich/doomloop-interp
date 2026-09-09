#!/usr/bin/env python3
"""What does the AV say about the state right before the model doubts itself?

Applies the trained NLA at the positions where the geometry showed an effect: the
last token of a reasoning block whose next block opens with "wait", versus the
last token of a block the model moved on from normally.

Per trace it verbalizes three states:

  doubt       end of a block followed by self-doubt
  plain       end of a block NOT followed by self-doubt, chosen so its distance
              to `think_last` is as close as possible to the doubt block's —
              similarity and, plausibly, explanation content both vary with
              position in the trace, so an unmatched control would be measuring
              depth rather than doubt
  think_last  last reasoning token before `</think>`, the model's final state
              (NOT `</think>` itself, which is a format token sitting ~0.23
              cosine from its own neighbour where neighbours average ~0.68)

Two passes — target model, then AV — because both plus a trace's states do not
fit on a 16 GB card.

    uv run python scripts/verbalize_blocks.py --limit 300
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
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
    think_last_content_char,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/correct_sample.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/block_explanations.csv"))
    p.add_argument("--base", default=MODEL_ID)
    p.add_argument("--av", type=Path, default=Path("checkpoints/av_rl_ep1"))
    p.add_argument("--marker", default="wait")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=300)
    return p.parse_args()


def pick_matched(view: Any, text: str, marker: str, content_idx: int) -> list[tuple[str, int]]:
    """One doubt boundary and the plain boundary closest to it in depth."""
    bounds = block_boundaries(text, view.think_chars, marker)
    doubt = [(b[0], b[2]) for b in bounds if b[2]]
    plain = [(b[0], b[2]) for b in bounds if not b[2]]
    if not doubt or not plain:
        return []

    # Take the first doubt boundary: it is the model's first turn against a
    # completed step, and later ones are already conditioned on it.
    d_idx = view.char_to_token(doubt[0][0])
    if d_idx is None:
        return []
    d_sep = abs(content_idx - d_idx)
    best: tuple[int, int] | None = None
    for char, _ in plain:
        idx = view.char_to_token(char)
        if idx is None:
            continue
        gap = abs(abs(content_idx - idx) - d_sep)
        if best is None or gap < best[0]:
            best = (gap, idx)
    if best is None:
        return []
    return [("doubt", d_idx), ("plain", best[1]), ("think_last", content_idx)]


def main() -> None:
    args = parse_args()
    rows_in = [json.loads(line) for line in args.traces.open()]
    rows_in = [r for r in rows_in if r["is_correct"]][: args.limit]
    print(f"{len(rows_in)} correct traces")

    cache = StateCache(args.base, NLAConfig().extraction_layer)
    items: list[dict[str, Any]] = []

    for n, row in enumerate(rows_in, 1):
        text = row["response"]
        content_char = think_last_content_char(text)
        if content_char is None:
            continue
        view = build_view(cache.tokenizer, row["question"], text, row["gold"])
        cache.fill(view)
        states = view.states
        assert states is not None
        content_idx = view.char_to_token(content_char)
        if content_idx is None:
            view.states = None
            continue
        picks = pick_matched(view, text, args.marker, content_idx)
        for kind, idx in picks:
            sep = abs(content_idx - idx)
            sim = cosine(states, idx, content_idx) if sep else 1.0
            mu, sd = baseline(states, sep, region=view.think) if sep else (0.0, 0.0)
            items.append(
                {
                    "question_id": row["question_id"],
                    "rollout_index": row["rollout_index"],
                    "dataset": row["dataset"],
                    "gold": row["gold"],
                    "kind": kind,
                    "token": idx,
                    "sep_to_think_last": sep,
                    "cosine_to_think_last": round(sim, 4),
                    "z": round((sim - mu) / sd, 3) if sd > 0 else "",
                    "context_tail": text[max(0, view.offsets[idx][0] - 220) : view.offsets[idx][1]],
                    "h_l": states[idx].float().cpu().clone(),
                }
            )
        if n % 25 == 0 or n == len(rows_in):
            print(f"  extracted [{n}/{len(rows_in)}] {len(items)} states")
        view.states = None
        del states
        torch.cuda.empty_cache()

    del cache
    torch.cuda.empty_cache()

    from reasoning_attention.nla.model import NLA

    print(f"\nverbalizing {len(items)} states")
    nla = NLA.av_only(str(args.av))
    nla.av.eval()
    for i, item in enumerate(items, 1):
        item["explanation"] = nla.verbalize(
            item.pop("h_l"), max_new_tokens=args.max_new_tokens, return_explanation=True
        ).strip()
        if i % 50 == 0 or i == len(items):
            print(f"  [{i}/{len(items)}] {item['explanation'][:90]!r}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
    print(f"\nwrote {args.out} ({len(items)} rows)")


if __name__ == "__main__":
    main()
