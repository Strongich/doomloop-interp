#!/usr/bin/env python3
"""What does the AV say about the state right before the model doubts itself?

Applies the trained NLA at the positions where the geometry showed an effect: the
last token of a reasoning block whose next block opens with "wait", versus the
last token of a block the model moved on from normally.

Per trace it verbalizes three states:

  doubt_wait  end of a block whose next block opens with "wait"
  doubt_other end of a block whose next block opens with some OTHER doubt marker
              (But/Hmm/Maybe/Actually/Alternatively/However/let me double-check…)
              but NOT "wait". This is the group that decides what the result
              means: if the AV predicts "wait" here too, it is tracking
              self-doubt; if it names the marker actually coming, it is
              predicting a token.
  plain       end of a block followed by no doubt marker at all
  think_last  last reasoning token before `</think>`, the model's final state
              (NOT `</think>` itself, which is a format token sitting ~0.23
              cosine from its own neighbour where neighbours average ~0.68)

Every probe is chosen so its distance to `think_last` is as close as possible to
the doubt_wait block's: cosine and, plausibly, explanation content both vary with
depth in the trace, so unmatched controls would measure depth rather than doubt.

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
from reasoning_attention.loops import (  # noqa: E402
    DOUBT_MARKERS_NON_WAIT,
)
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
    p.add_argument("--out", type=Path, default=Path("data/block_explanations_3way.csv"))
    p.add_argument("--base", default=MODEL_ID)
    p.add_argument("--av", type=Path, default=Path("checkpoints/av_rl_ep1"))
    p.add_argument("--marker", default="wait")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--limit", type=int, default=300)
    return p.parse_args()


def pick_matched(view: Any, text: str, marker: str, content_idx: int) -> list[tuple[str, int, str]]:
    """One probe per group, each matched in depth to the doubt_wait block.

    Groups are keyed on what the NEXT block opens with, so all three probes are
    the same kind of position — the last token of a completed reasoning block —
    and differ only in what the model chose to do next.
    """
    markers = (marker, *DOUBT_MARKERS_NON_WAIT)
    bounds = block_boundaries(text, view.think_chars, markers)

    groups: dict[str, list[tuple[int, str]]] = {"doubt_wait": [], "doubt_other": [], "plain": []}
    for end_char, _nxt, hit in bounds:
        idx = view.char_to_token(end_char)
        if idx is None:
            continue
        if hit is None:
            groups["plain"].append((idx, ""))
        elif hit.lower() == marker.lower():
            groups["doubt_wait"].append((idx, hit))
        else:
            groups["doubt_other"].append((idx, hit))

    if not groups["doubt_wait"]:
        return []
    # The first doubt_wait boundary anchors the depth; later ones are already
    # conditioned on the doubt that preceded them.
    d_idx, _ = groups["doubt_wait"][0]
    target = abs(content_idx - d_idx)

    out: list[tuple[str, int, str]] = [("doubt_wait", d_idx, marker)]
    for name in ("doubt_other", "plain"):
        best: tuple[int, int, str] | None = None
        for idx, hit in groups[name]:
            gap = abs(abs(content_idx - idx) - target)
            if best is None or gap < best[0]:
                best = (gap, idx, hit)
        if best is not None:
            out.append((name, best[1], best[2]))
    out.append(("think_last", content_idx, ""))
    return out


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
        for kind, idx, hit in picks:
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
                    "next_marker": hit,
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
