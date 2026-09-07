"""Sample reasoning traces from the target model over the math eval sets.

This produces the raw material for the doom-loop study: many rollouts per
question, each labelled with the outcome the study splits on. From `CLAUDE.md`:

    the case study reads `h_l` at the same marker phrases in traces that
    **reach a correct `\\boxed{}` answer** versus traces that **produce no
    answer at all** (budget exhausted inside `<think>`)

so every record carries enough to sort itself into those buckets without
re-running anything:

    outcome     "correct" | "incorrect" | "no_answer"
    stop_reason vLLM's finish_reason — "length" means the window ran out
    exited_think whether `</think>` was ever emitted

`no_answer` + `stop_reason="length"` + `exited_think=False` is the derailment
case; `no_answer` with `exited_think=True` is a trace that finished talking
without boxing anything, which is a *different* failure and must not be pooled
with it.

Sampling is Qwen3's thinking-mode recommendation (T=0.6, top_p=0.95, top_k=20,
min_p=0). Generation is NOT capped by a fixed `max_tokens`: each request gets
`max_model_len - len(prompt)`, so the whole 32k window is the budget and a trace
that loops until exhaustion is *recorded as such* rather than truncated early by
a limit of ours. That matters — the budget boundary is the phenomenon.

All rollouts for a question go in ONE request via `SamplingParams(n=...)`, so the
prompt is prefilled once rather than once per sample.

Output is JSONL (one row per rollout), streamed and appended, so a crash or a
pod restart loses only the question in flight. Re-running skips questions already
present.

Usage:
    uv run python scripts/generate_traces.py --out data/traces
    uv run python scripts/generate_traces.py --datasets aime2025 amc23
    uv run python scripts/generate_traces.py --datasets gsm8k --limit 200
    uv run python scripts/generate_traces.py --system "You are a helpful assistant."
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
ROLLOUTS_PER_QUESTION = 4
THINK_CLOSE = "</think>"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/traces", help="output directory for <dataset>.jsonl")
    p.add_argument("--datasets", nargs="+", default=sorted(DATASETS), choices=sorted(DATASETS))
    p.add_argument("--n", type=int, default=ROLLOUTS_PER_QUESTION, help="rollouts per question")
    p.add_argument("--limit", type=int, default=None, help="first N questions per dataset")
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
    return p.parse_args()


def done_question_ids(path: Path) -> set[str]:
    """Question ids already fully written, for resume."""
    if not path.exists():
        return set()
    seen: dict[str, int] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # truncated last line from a hard kill
            seen[row["question_id"]] = seen.get(row["question_id"], 0) + 1
    return {q for q, n in seen.items() if n > 0}


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

    cfg = replace(VLLMConfig(), max_model_len=args.max_model_len)
    llm = build_llm(cfg)
    tokenizer = llm.get_tokenizer()

    from vllm import SamplingParams

    for name in args.datasets:
        ds = load_one(name)
        if args.limit is not None:
            ds = ds.select(range(min(args.limit, len(ds))))
        path = out_dir / f"{name}.jsonl"
        already = done_question_ids(path)

        prompts: list[str] = []
        params: list[Any] = []
        meta: list[dict[str, Any]] = []
        for i, row in enumerate(ds):
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
            # Per-request cap: the window, not an arbitrary generation limit.
            params.append(
                SamplingParams(
                    n=args.n,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    min_p=args.min_p,
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
                    "messages_prompt": messages,
                    "prompt_tokens": n_prompt,
                }
            )

        print(f"\n=== {name}: {len(prompts)} questions x {args.n} rollouts "
              f"({len(already)} already done) ===")
        if not prompts:
            continue

        # generate() returns results aligned with `prompts`; write per question so
        # an interrupted run leaves complete questions behind, never half of one.
        outputs = llm.generate(prompts, params)
        with path.open("a", encoding="utf-8") as fh:
            for m, out in zip(meta, outputs, strict=True):
                for k, comp in enumerate(out.outputs):
                    rec = {
                        **m,
                        "rollout_index": k,
                        "response": comp.text,
                        "completion_tokens": len(comp.token_ids),
                        "messages": [*m["messages_prompt"], {"role": "assistant", "content": comp.text}],
                        **classify(comp.text, m["gold"], comp.finish_reason, list(comp.token_ids)),
                    }
                    rec.pop("messages_prompt")
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()

        rows = [json.loads(line) for line in path.open(encoding="utf-8")]
        n = len(rows)
        counts: dict[str, int] = {}
        for r in rows:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
        stuck = sum(1 for r in rows if r["outcome"] == "no_answer" and not r["exited_think"])
        print(f"  {n} rollouts -> {path}")
        print(f"  outcomes: {counts}")
        print(f"  derailed inside <think> (no answer, never closed the tag): {stuck}")
        print(f"  doom-loop detector fired: {sum(1 for r in rows if r['has_doom_loop'])}")


if __name__ == "__main__":
    main()
