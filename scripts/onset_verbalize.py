#!/usr/bin/env python3
"""Verbalize the residual stream at the token before a trace turns on itself.

Two ways to pick that token, `--onset`:

  `wait`  the FIRST occurrence of a self-doubt marker, default "Wait". One per
          trace: a rollout interrupts itself many times (median 4, max 2950) and
          only the first is comparable across traces — later ones are already
          conditioned on the doubt that preceded them.
          This is the study's main contrast: read `h_l` at the *same* marker in
          traces that answered correctly and traces that did not, and ask the AV
          what differs. Per D33 the marker itself is not a failure signal — 99.9%
          of traces in this corpus contain one — so it selects a position, never
          a label. The label is always the outcome.
  `loop`  the onset of a verbatim loop, by Antidoom's criterion. Precise but
          rare: 34 of 35,728 traces.

Either way:

  1. find the onset token;
  2. rebuild the exact context the model had *just before* emitting it:
     chat template + user turn + assistant text truncated at the onset;
  3. read `h_l` (layer 20 residual, `hidden_states[21]`) at the FINAL token of
     that context — the position whose next-token distribution chose the marker;
  4. inject it into the trained AV and let it explain itself;
  5. write a CSV row, labelled by whether the trace ended up correct.

Two passes, base model then AV, rather than both resident: a 28k-token forward
pass plus two 1.7B models does not fit in 16 GB.

    # two CSVs, correct vs not, 500 traces each
    uv run python scripts/onset_verbalize.py --onset wait --per-class 500 \
        --split-by-label --out data/wait_onsets.csv

    # every verbatim loop, one file
    uv run python scripts/onset_verbalize.py --onset loop --out data/loop_onsets.csv

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
    DEFAULT_MARKER,
    find_inner_repetition,
    marker_matches,
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
    p.add_argument("--out", type=Path, default=Path("data/onsets.csv"))
    p.add_argument("--onset", choices=["wait", "loop"], default="wait")
    p.add_argument("--marker", default=DEFAULT_MARKER, help="--onset wait: the phrase")
    p.add_argument(
        "--marker-fraction",
        type=float,
        default=None,
        help="which occurrence to read, as a fraction of how many the trace has. "
        "0.7 with 10 markers picks the 7th; omit for the first. Normalizing by count "
        "compares equivalent *stages* of traces that doubt wildly different amounts "
        "(median 14 markers when the rollout succeeds, 18 when it does not, max 2950)",
    )
    p.add_argument("--ignore-case", action="store_true")
    p.add_argument(
        "--per-class",
        type=int,
        default=None,
        help="sample this many correct and this many incorrect traces, spread evenly "
        "over the datasets. Required for --onset wait: 99.9%% of traces qualify",
    )
    p.add_argument(
        "--paired",
        action="store_true",
        help="emit matched pairs: for each question, one correct and one incorrect "
        "rollout, so the two files differ only in how the rollout went. --per-class "
        "then counts pairs",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--split-by-label",
        action="store_true",
        help="write <stem>_correct.csv and <stem>_incorrect.csv instead of one file",
    )
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


def _marker_rank(n_matches: int, fraction: float | None) -> int:
    """1-based occurrence to read: the first, or `fraction` of the way through.

    Rounded, not floored, so 0.7 of 10 markers is the 7th as one would expect;
    clamped into [1, n] so a trace with a single marker still yields it.
    """
    if fraction is None:
        return 1
    return max(1, min(n_matches, round(fraction * n_matches)))


def _locate(args: argparse.Namespace, tokenizer: Any, row: dict[str, Any]) -> dict[str, Any] | None:
    """Onset token for one trace, or None if this trace has none."""
    text = row["response"]

    if args.onset == "loop":
        hit = find_inner_repetition(text)
        if hit is None:
            return None
        spans = tokenize_with_spans(tokenizer, text)
        onset_idx = onset_token_index(spans, hit)
        extra: dict[str, Any] = {
            "loop_period_chars": hit.period,
            "marker_rank": 2,  # repeat #2 is where a loop's onset lives
            "marker_count": hit.repeats,
            "marker": hit.snippet,
        }
    else:
        # Tokenize only as far as the marker, not the whole trace. Tokenizing all
        # 35,728 responses in full costs minutes, and every token past the onset
        # is discarded anyway. The window keeps a few chars past the marker so
        # the " Wait" merge can still form at the boundary.
        matches = marker_matches(text, args.marker, case_sensitive=not args.ignore_case)
        if not matches:
            return None
        rank = _marker_rank(len(matches), args.marker_fraction)
        at = matches[rank - 1]
        spans = tokenize_with_spans(tokenizer, text[: at + len(args.marker) + 8])
        onset_idx = spans.char_to_token(at)
        extra = {
            "loop_period_chars": "",
            "marker_rank": rank,
            "marker_count": len(matches),
            "marker": args.marker,
        }

    if onset_idx is None:
        return None
    return {
        "row": row,
        "onset_token_index": onset_idx,
        "onset_char": spans.offsets[onset_idx][0],
        "onset_token": text[slice(*spans.offsets[onset_idx])],
        **extra,
    }


def _iter_traces(args: argparse.Namespace) -> Any:
    for name in args.datasets:
        path = args.traces / f"{name}.jsonl"
        if not path.exists():
            print(f"  skip {name}: {path} missing")
            continue
        with path.open() as fh:
            for line in fh:
                yield name, json.loads(line)


def _round_robin(buckets: list[list[Any]], want: int, label: str) -> list[Any]:
    """Take `want` items, one at a time from each bucket in turn.

    Proportional sampling would be useless here: GSM8K is 98% of the corpus, so
    a proportional draw would contain almost no AIME/AMC — the hard problems the
    study is actually about. A bucket that runs dry hands its share to the rest.
    """
    picked: list[Any] = []
    buckets = [b for b in buckets if b]
    while len(picked) < want and buckets:
        for bucket in buckets:
            if len(picked) >= want:
                break
            picked.append(bucket.pop())
        buckets = [b for b in buckets if b]
    if len(picked) < want:
        print(f"  only {len(picked)} {label} available (wanted {want})")
    return picked


def _sample(
    args: argparse.Namespace, pools: dict[tuple[str, bool], list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    """`--per-class` traces per label, spread as evenly as the datasets allow."""
    import random

    rng = random.Random(args.seed)
    chosen: list[dict[str, Any]] = []
    for label in (True, False):
        buckets = [
            rng.sample(pool, len(pool)) for d in args.datasets if (pool := pools.get((d, label)))
        ]
        chosen.extend(
            _round_robin(buckets, args.per_class, f"label={'correct' if label else 'incorrect'}")
        )
    return chosen


def _sample_paired(
    args: argparse.Namespace, by_question: dict[str, dict[bool, list[dict[str, Any]]]]
) -> list[dict[str, Any]]:
    """One correct and one incorrect rollout from each of `--per-class` questions.

    This is the contrast the study wants: holding the question fixed removes
    problem difficulty as a confound, so a difference between the two files is a
    difference between *this rollout recovering and that one not*, on the same
    problem. An unpaired draw cannot separate those.
    """
    import random

    rng = random.Random(args.seed)
    per_dataset: dict[str, list[str]] = {}
    for qid, sides in by_question.items():
        if not (sides.get(True) and sides.get(False)):
            continue  # not a mixed-outcome question — no pair to make
        per_dataset.setdefault(qid.split(":", 1)[0], []).append(qid)

    buckets = [
        rng.sample(per_dataset[d], len(per_dataset[d])) for d in args.datasets if per_dataset.get(d)
    ]
    questions = _round_robin(buckets, args.per_class, "paired questions")
    print(f"  paired on {len(questions)} questions -> {2 * len(questions)} rows")

    out: list[dict[str, Any]] = []
    for qid in questions:
        for label in (True, False):
            out.append(rng.choice(by_question[qid][label]))
    return out


def find_onsets(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    """Locate the onset in every eligible trace, then sample if asked."""
    keep: set[str] | None = None
    if args.paired:
        keep = _mixed_outcome_questions(args)
        print(f"  {len(keep)} mixed-outcome questions in the corpus")

    pools: dict[tuple[str, bool], list[dict[str, Any]]] = {}
    by_question: dict[str, dict[bool, list[dict[str, Any]]]] = {}
    seen: dict[str, int] = {}
    unresolved = 0
    for name, row in _iter_traces(args):
        seen[name] = seen.get(name, 0) + 1
        if keep is not None and row["question_id"] not in keep:
            continue
        item = _locate(args, tokenizer, row)
        if item is None:
            unresolved += 1
            continue
        label = bool(row["is_correct"])
        pools.setdefault((name, label), []).append(item)
        by_question.setdefault(row["question_id"], {}).setdefault(label, []).append(item)

    for name in args.datasets:
        n = len(pools.get((name, True), [])) + len(pools.get((name, False), []))
        if seen.get(name):
            print(f"  {name}: {n}/{seen[name]} with an onset")
    if unresolved:
        print(f"  {unresolved} traces had no locatable onset")

    if args.paired:
        if not args.per_class:
            raise SystemExit("--paired needs --per-class (it counts pairs)")
        return _sample_paired(args, by_question)
    if args.per_class:
        return _sample(args, pools)
    items = [i for pool in pools.values() for i in pool]
    return items[: args.limit] if args.limit else items


def _mixed_outcome_questions(args: argparse.Namespace) -> set[str]:
    """Questions where some rollouts answered correctly and some did not."""
    outcomes: dict[str, set[bool]] = {}
    for _name, row in _iter_traces(args):
        outcomes.setdefault(row["question_id"], set()).add(bool(row["is_correct"]))
    return {q for q, v in outcomes.items() if len(v) > 1}


def build_prefix(
    args: argparse.Namespace, tokenizer: Any, item: dict[str, Any]
) -> tuple[str, list[dict[str, str]]]:
    """The full system+user+assistant context, cut at the onset token.

    Must reproduce what the model actually saw during generation, so the chat
    template is applied the same way `scripts/generate_traces.py` applied it
    (`enable_thinking=True`, `add_generation_prompt=True`) and the assistant text
    is appended raw — the template does not emit `<think>`, the model does.

    Returns both the flat string that gets tokenized and the message list it came
    from, with the assistant turn truncated to exactly the tokens `h_l` is read
    over. The messages go in the CSV so a row can be replayed without re-deriving
    the onset.
    """
    row = item["row"]
    messages = build_messages(row["question"])
    if args.system:
        messages = [{"role": "system", "content": args.system}, *messages]
    header = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
    )
    assistant = row["response"][: item["onset_char"]]
    return header + assistant, [*messages, {"role": "assistant", "content": assistant}]


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
            prefix, messages = build_prefix(args, tokenizer, item)
            item["messages"] = messages
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
    "marker_rank",
    "marker_count",
    "marker",
    "explanation",
    "question",
    # The exact context h_l was read over: user turn plus the assistant turn
    # truncated at the onset token. JSON, so it round-trips.
    "messages",
]


def write_split(path: Path, items: list[dict[str, Any]]) -> list[Path]:
    """One CSV per label — correct and incorrect kept in separate files."""
    out = []
    for label, want in (("correct", True), ("incorrect", False)):
        target = path.with_name(f"{path.stem}_{label}{path.suffix}")
        subset = [i for i in items if bool(i["row"]["is_correct"]) is want]
        write_csv(target, subset)
        print(f"wrote {target} ({len(subset)} rows)")
        out.append(target)
    return out


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
                    "loop_period_chars": item["loop_period_chars"],
                    "marker_rank": item["marker_rank"],
                    "marker_count": item["marker_count"],
                    "marker": item["marker"],
                    "explanation": item["explanation"],
                    "question": row["question"],
                    "messages": json.dumps(item["messages"], ensure_ascii=False),
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

    print()
    if args.split_by_label:
        write_split(args.out, items)
    else:
        write_csv(args.out, items)
        print(f"wrote {args.out} ({len(items)} rows)")
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
