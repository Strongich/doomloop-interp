#!/usr/bin/env python3
"""Verbalize the residual stream at the moment a trace commits to looping.

For every trace in the corpus that contains a verbatim loop (Antidoom's
criterion — see `reasoning_attention.loops`):

  1. find the loop's onset token — the first readable token of repeat #2;
  2. rebuild the exact context the model had *just before* emitting it:
     chat template + user turn + assistant text truncated at the onset;
  3. read `h_l` (layer 20 residual, `hidden_states[21]`) at the FINAL token of
     that context — the position whose next-token distribution chose to repeat;
  4. inject it into the trained AV and let it explain itself;
  5. write one CSV row, labelled by the trace's outcome.

Two passes, base model then AV, rather than both resident: a 28k-token forward
pass plus two 1.7B models does not fit in 16 GB.

    uv run python scripts/loop_onset_verbalize.py --av checkpoints/av_rl_ep1

The forward pass uses a hook on layer 20 instead of `output_hidden_states=True`:
the latter materializes all 29 layers for the whole sequence (~3.9 GB at 32k
tokens) when we want one 2048-vector. It also calls the inner transformer
directly, skipping the lm_head that would otherwise build an 8.7 GB logit
tensor for a 28k-token prefix.
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
from reasoning_attention.data.math_datasets import build_messages  # noqa: E402
from reasoning_attention.loops import (  # noqa: E402
    find_inner_repetition,
    onset_token_index,
    tokenize_with_spans,
)
from reasoning_attention.nla.arch import inner_transformer  # noqa: E402
from reasoning_attention.nla.model import NLA  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/traces"))
    p.add_argument("--datasets", nargs="+", default=["aime2025", "amc23", "gsm8k"])
    p.add_argument("--av", type=Path, default=Path("checkpoints/av_rl_ep1"))
    p.add_argument("--out", type=Path, default=Path("data/loop_onsets.csv"))
    p.add_argument("--base", default=MODEL_ID, help="model the traces came from")
    p.add_argument("--system", default=None, help="system prompt, if the traces used one")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="AV sampling temperature; 0 = greedy, so the explanation is reproducible",
    )
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def find_onsets(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    """Every trace with a verbatim loop, with its onset located in token space."""
    out: list[dict[str, Any]] = []
    for name in args.datasets:
        path = args.traces / f"{name}.jsonl"
        if not path.exists():
            print(f"  skip {name}: {path} missing")
            continue
        n_rows = n_hit = 0
        for line in path.open():
            row = json.loads(line)
            n_rows += 1
            hit = find_inner_repetition(row["response"])
            if hit is None:
                continue
            spans = tokenize_with_spans(tokenizer, row["response"])
            onset_idx = onset_token_index(spans, hit)
            if onset_idx is None:
                # A loop whose period cannot be placed in token space. Counted,
                # not silently dropped — it is a detector limitation, not a
                # trace without a loop.
                print(f"  {row['question_id']}#{row['rollout_index']}: onset unresolved")
                continue
            n_hit += 1
            onset_char = spans.offsets[onset_idx][0]
            out.append(
                {
                    "row": row,
                    "onset_token_index": onset_idx,
                    "onset_char": onset_char,
                    "onset_token": row["response"][slice(*spans.offsets[onset_idx])],
                    "period": hit.period,
                    "repeats": hit.repeats,
                    "loop_start_char": hit.start,
                    "snippet": hit.snippet,
                }
            )
        print(f"  {name}: {n_hit}/{n_rows} looped")
    if args.limit:
        out = out[: args.limit]
    return out


def build_prefix(args: argparse.Namespace, tokenizer: Any, item: dict[str, Any]) -> str:
    """The full system+user+assistant context, cut at the onset token.

    Must reproduce what the model actually saw during generation, so the chat
    template is applied the same way `scripts/generate_traces.py` applied it
    (`enable_thinking=True`, `add_generation_prompt=True`) and the assistant text
    is appended raw — the template does not emit `<think>`, the model does.
    """
    row = item["row"]
    messages = build_messages(row["question"])
    if args.system:
        messages = [{"role": "system", "content": args.system}, *messages]
    header = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    return header + row["response"][: item["onset_char"]]


@torch.no_grad()
def extract_activations(args: argparse.Namespace, items: list[dict[str, Any]]) -> None:
    """Fill each item's `h_l`: layer-20 residual at the final prefix token."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = NLAConfig()
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    captured: dict[str, torch.Tensor] = {}

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        captured["h"] = hidden[:, -1, :].detach().float().cpu()

    # Call the inner transformer, NOT the causal-LM wrapper: the wrapper runs
    # lm_head over every position, which at 28k tokens is a
    # 28k x 151936 logit tensor (~8.7 GB) we throw away. The hook fires either
    # way, so nothing about the activation changes.
    trunk = inner_transformer(model)
    layer = trunk.layers[cfg.extraction_layer]
    handle = layer.register_forward_hook(hook)
    try:
        for i, item in enumerate(items, 1):
            prefix = build_prefix(args, tokenizer, item)
            enc = tokenizer(prefix, return_tensors="pt").to(model.device)
            trunk(**enc)
            h = captured["h"].reshape(-1)
            item["h_l"] = h
            item["prefix_tokens"] = int(enc["input_ids"].shape[1])
            item["h_norm"] = float(h.norm())
            print(
                f"  [{i}/{len(items)}] {item['row']['question_id']}"
                f"#{item['row']['rollout_index']} "
                f"tokens={item['prefix_tokens']} ‖h‖={item['h_norm']:.1f}"
            )
    finally:
        handle.remove()

    del model
    torch.cuda.empty_cache()


@torch.no_grad()
def verbalize_all(args: argparse.Namespace, items: list[dict[str, Any]]) -> None:
    """Fill each item's `explanation` by injecting `h_l` into the trained AV."""
    nla = NLA.av_only(str(args.av))
    nla.av.eval()
    gen: dict[str, Any] = {"do_sample": False}
    if args.temperature > 0:
        gen = {"do_sample": True, "temperature": args.temperature}
    for i, item in enumerate(items, 1):
        item["explanation"] = nla.verbalize(
            item["h_l"],
            max_new_tokens=args.max_new_tokens,
            return_explanation=True,
            **gen,
        ).strip()
        print(f"  [{i}/{len(items)}] {item['explanation'][:110]!r}")


FIELDS = [
    "question_id",
    "dataset",
    "split",
    "rollout_index",
    "label",
    "outcome",
    "is_correct",
    "has_answer",
    "exited_think",
    "stop_reason",
    "gold",
    "prefix_tokens",
    "onset_token_index",
    "onset_token",
    "h_norm",
    "loop_period_chars",
    "loop_repeats",
    "loop_snippet",
    "explanation",
    "question",
]


def write_csv(path: Path, items: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for item in items:
            row = item["row"]
            writer.writerow(
                {
                    "question_id": row["question_id"],
                    "dataset": row["dataset"],
                    "split": row.get("split", ""),
                    "rollout_index": row["rollout_index"],
                    # The study's label. `outcome` keeps the third case visible:
                    # a trace that never boxed an answer is not the same thing as
                    # one that boxed a wrong one.
                    "label": "correct" if row["is_correct"] else "incorrect",
                    "outcome": row["outcome"],
                    "is_correct": row["is_correct"],
                    "has_answer": row["has_answer"],
                    "exited_think": row["exited_think"],
                    "stop_reason": row["stop_reason"],
                    "gold": row["gold"],
                    "prefix_tokens": item["prefix_tokens"],
                    "onset_token_index": item["onset_token_index"],
                    "onset_token": item["onset_token"],
                    "h_norm": round(item["h_norm"], 2),
                    "loop_period_chars": item["period"],
                    "loop_repeats": item["repeats"],
                    "loop_snippet": item["snippet"],
                    "explanation": item["explanation"],
                    "question": row["question"],
                }
            )


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    print("== detecting loops ==")
    items = find_onsets(args, AutoTokenizer.from_pretrained(args.base))
    if not items:
        print("no loops found; nothing to do")
        return
    print(f"{len(items)} traces with a located onset\n")

    print("== extracting h_l at the onset ==")
    extract_activations(args, items)

    print("\n== verbalizing with the AV ==")
    verbalize_all(args, items)

    write_csv(args.out, items)
    print(f"\nwrote {args.out} ({len(items)} rows)")
    torch.save(
        {
            "h_l": torch.stack([i["h_l"] for i in items]),
            "keys": [(i["row"]["question_id"], i["row"]["rollout_index"]) for i in items],
        },
        args.out.with_suffix(".pt"),
    )
    print(f"wrote {args.out.with_suffix('.pt')} (activations, for reuse)")


if __name__ == "__main__":
    main()
