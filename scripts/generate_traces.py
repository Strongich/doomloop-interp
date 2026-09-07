"""Sample reasoning traces from the target model over the math eval sets.

This produces the raw material for the doom-loop study: several rollouts per
question, each labelled with the outcome the study splits on. From `CLAUDE.md`:

    the case study reads `h_l` at the same marker phrases in traces that
    **reach a correct `\\boxed{}` answer** versus traces that **produce no
    answer at all** (budget exhausted inside `<think>`)

so every record carries enough to sort itself into those buckets without
re-running anything:

    outcome      "correct" | "incorrect" | "no_answer"
    stop_reason  vLLM's finish_reason — "length" means the window ran out
    exited_think whether `</think>` was ever emitted

`no_answer` + `stop_reason="length"` + `exited_think=False` is the derailment
case; `no_answer` with `exited_think=True` is a trace that stopped talking
without boxing anything, which is a *different* failure and must not be pooled
with it.

Sampling is Qwen3's thinking-mode recommendation (T=0.6, top_p=0.95, top_k=20,
min_p=0). Generation is NOT capped by a fixed `max_tokens`: each request gets
`max_model_len - len(prompt)`, so the whole 32k window is the budget and a trace
that loops until exhaustion is *recorded as such* rather than truncated early by
a limit of ours. That matters — the budget boundary is the phenomenon.

All rollouts for one question go in ONE request via `SamplingParams(n=...)`, so
the prompt is prefilled once rather than once per sample.

SHARDING. One vLLM engine per GPU, each taking a stride of the questions
(`--shard-index i --num-shards n` -> questions i, i+n, i+2n, ...). A stride
rather than contiguous blocks because datasets are often ordered by difficulty,
and contiguous blocks would leave one GPU with all the long traces. Each shard
writes its own file; `--merge` concatenates them.

Per-question RNG seeds use the GLOBAL question index, so a question gets the same
seed no matter which shard draws it and the sharding does not change results.

Output is JSONL, appended per question, so a crash loses only the question in
flight. Re-running a shard skips questions already in its own file.

Usage:
    # one GPU
    uv run python scripts/generate_traces.py --datasets aime2025 amc23

    # two GPUs, then merge (see generate_traces.sh for the driver)
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/generate_traces.py \\
        --shard-index 0 --num-shards 2 &
    CUDA_VISIBLE_DEVICES=1 uv run python scripts/generate_traces.py \\
        --shard-index 1 --num-shards 2 &
    wait
    uv run python scripts/generate_traces.py --merge
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from reasoning_attention.config import VLLMConfig, load_project_env
from reasoning_attention.data.math_datasets import DATASETS, build_messages, load_one
from reasoning_attention.grading import grade
from reasoning_attention.metrics import has_doom_loop, ngram_repetition_ratio
from reasoning_attention.serving.vllm_server import build_llm

# The whole prompt+generation budget. Qwen3-1.7B's native window is 32768.
MAX_MODEL_LEN = 32_768
THINK_CLOSE = "</think>"

# Rollouts per question, per dataset. GSM8K is the *recovered* control and is
# huge (8792 questions), so 4 is plenty; AIME/AMC are tiny (30 and 40) and are
# where derailment actually happens, so 8 buys per-question outcome variance —
# the mixed-outcome questions, where some rollouts recover and some derail, are
# the matched pairs the study wants.
DEFAULT_ROLLOUTS: dict[str, int] = {"gsm8k": 4, "aime2025": 8, "amc23": 8}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--out", default="data/traces", help="output directory")
    p.add_argument("--datasets", nargs="+", default=sorted(DATASETS), choices=sorted(DATASETS))
    p.add_argument(
        "--n",
        type=int,
        default=None,
        help=f"rollouts per question, overriding the per-dataset default {DEFAULT_ROLLOUTS}",
    )
    p.add_argument("--limit", type=int, default=None, help="first N questions per dataset")
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument(
        "--merge",
        action="store_true",
        help="merge <dataset>.shard*.jsonl into <dataset>.jsonl and print a summary; "
        "runs no model and needs no GPU",
    )
    p.add_argument("--max-model-len", type=int, default=MAX_MODEL_LEN)
    p.add_argument(
        "--system",
        default=None,
        help="optional system prompt. Default NONE: the eval sets are locked to a "
        "single user turn (CLAUDE.md), and Qwen3's template inserts no system "
        "message of its own, so adding one changes the eval protocol. The full "
        "message list is recorded either way.",
    )
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--min-p", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--min-completion-tokens",
        type=int,
        default=64,
        help="skip a question whose prompt leaves less room than this (never happens "
        "at 32k, but a smaller --max-model-len would silently produce stubs)",
    )
    args = p.parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        p.error(f"--shard-index must be in [0, {args.num_shards})")
    return args


def shard_path(out_dir: Path, name: str, index: int, total: int) -> Path:
    """Where one shard writes. `total == 1` keeps the plain, unsharded name."""
    if total == 1:
        return out_dir / f"{name}.jsonl"
    return out_dir / f"{name}.shard{index}of{total}.jsonl"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Rows from a JSONL file, tolerating a truncated final line."""
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue  # a hard kill mid-write leaves one partial line
    return rows


def summarize(rows: list[dict[str, Any]], label: str) -> None:
    counts: dict[str, int] = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    stuck = sum(1 for r in rows if r["outcome"] == "no_answer" and not r["exited_think"])
    loops = sum(1 for r in rows if r["has_doom_loop"])
    qs = len({r["question_id"] for r in rows})
    # Questions whose rollouts disagree: the matched pairs the study wants.
    by_q: dict[str, set[str]] = {}
    for r in rows:
        by_q.setdefault(r["question_id"], set()).add(r["outcome"])
    mixed = sum(1 for v in by_q.values() if len(v) > 1)
    print(f"  {label}: {len(rows)} rollouts over {qs} questions")
    print(f"    outcomes: {counts}")
    print(f"    derailed inside <think> (no answer, tag never closed): {stuck}")
    print(f"    doom-loop detector fired: {loops}")
    print(f"    mixed-outcome questions (recovered AND failed): {mixed}")


def do_merge(out_dir: Path, datasets: list[str]) -> None:
    for name in datasets:
        shards = sorted(out_dir.glob(f"{name}.shard*of*.jsonl"))
        if not shards:
            print(f"{name}: no shards found, skipping")
            continue
        # Dedupe on (question_id, rollout_index): re-running a shard after a
        # partial write can otherwise duplicate a question's rollouts.
        merged: dict[tuple[str, int], dict[str, Any]] = {}
        for sh in shards:
            for row in read_jsonl(sh):
                merged[(row["question_id"], row["rollout_index"])] = row
        rows = [merged[k] for k in sorted(merged, key=lambda k: (k[0], k[1]))]
        dest = out_dir / f"{name}.jsonl"
        with dest.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{name}: {len(shards)} shards -> {dest}")
        summarize(rows, name)


def classify(text: str, gold: str, stop_reason: str, token_ids: list[int]) -> dict[str, Any]:
    """Outcome plus the degeneracy signals, for one completion."""
    g = grade(text, gold)
    outcome = "correct" if g.is_correct else ("incorrect" if g.has_answer else "no_answer")
    return {
        "outcome": outcome,
        "has_answer": g.has_answer,
        "is_correct": g.is_correct,
        "grade_status": g.status,
        "stop_reason": stop_reason,
        # `</think>` absent + stop_reason "length" == derailed inside the CoT,
        # which is the study's failure case. Absent WITH a normal stop is a
        # different animal and stays distinguishable.
        "exited_think": THINK_CLOSE in text,
        "repetition_ratio_4gram": round(ngram_repetition_ratio(text, n=4), 4),
        "has_doom_loop": has_doom_loop(token_ids),
    }


def main() -> None:
    args = parse_args()
    load_project_env()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        do_merge(out_dir, args.datasets)
        return

    cfg = replace(VLLMConfig(), max_model_len=args.max_model_len)
    llm = build_llm(cfg)
    tokenizer = llm.get_tokenizer()

    from vllm import SamplingParams

    for name in args.datasets:
        n_rollouts = args.n if args.n is not None else DEFAULT_ROLLOUTS.get(name, 4)
        ds = load_one(name)
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))
        path = shard_path(out_dir, name, args.shard_index, args.num_shards)
        already = {r["question_id"] for r in read_jsonl(path)}

        prompts: list[str] = []
        params: list[Any] = []
        meta: list[dict[str, Any]] = []
        for i, row in enumerate(ds):
            # Stride assignment on the GLOBAL index, so shards are disjoint and
            # a question's seed does not depend on how many shards there are.
            if i % args.num_shards != args.shard_index:
                continue
            qid = f"{name}:{i}"
            if qid in already:
                continue
            messages = build_messages(row["question"])
            if args.system:
                messages = [{"role": "system", "content": args.system}, *messages]
            prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
            n_prompt = len(tokenizer.encode(prompt))
            room = args.max_model_len - n_prompt
            if room < args.min_completion_tokens:
                print(f"  skip {qid}: prompt {n_prompt} tok leaves only {room}")
                continue
            prompts.append(prompt)
            params.append(
                SamplingParams(
                    n=n_rollouts,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
                    # The window, not an arbitrary generation limit.
                    max_tokens=room,
                    seed=args.seed + i,
                )
            )
            meta.append(
                {
                    "question_id": qid,
                    "dataset": name,
                    "question_index": i,
                    "question": row["question"],
                    "gold": row["answer"],
                    "source": row["source"],
                    # gsm8k is train+test concatenated (8792 rows); keep the
                    # split so the traces can be filtered to test-only later.
                    "split": row.get("split", ""),
                    "subset": row.get("subset", ""),
                    "prompt_tokens": n_prompt,
                    "_messages_prompt": messages,
                }
            )

        print(
            f"\n=== {name} shard {args.shard_index}/{args.num_shards}: "
            f"{len(prompts)} questions x {n_rollouts} rollouts "
            f"({len(already)} already done) ==="
        )
        if not prompts:
            continue

        outputs = llm.generate(prompts, params)
        with path.open("a", encoding="utf-8") as fh:
            for m, out in zip(meta, outputs, strict=True):
                prompt_messages = m.pop("_messages_prompt")
                for k, comp in enumerate(out.outputs):
                    rec = {
                        **m,
                        "rollout_index": k,
                        "n_rollouts": n_rollouts,
                        "response": comp.text,
                        "completion_tokens": len(comp.token_ids),
                        "messages": [
                            *prompt_messages,
                            {"role": "assistant", "content": comp.text},
                        ],
                        **classify(comp.text, m["gold"], comp.finish_reason, list(comp.token_ids)),
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

        summarize(read_jsonl(path), f"{name} shard {args.shard_index}")


if __name__ == "__main__":
    main()
