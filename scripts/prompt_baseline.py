#!/usr/bin/env python3
"""Can a SYSTEM PROMPT do what the activation edit does?

This is the biggest single threat to the steering result. If instructing the
model not to second-guess itself reproduces Finding 6's -61% length at unchanged
accuracy, then the residual-stream edit is an interpretability finding -- we can
read the direction the model uses -- but not a method. arXiv:2605.10664
("Prompt-Activation Duality") argues the two are closer than people assume, so a
reviewer will ask.

## Both conditions are generated HERE, on purpose

Findings 5/6 were generated with a hand-rolled HF decode loop, because injecting
into the residual stream mid-forward needs a hook. Nothing is injected here, so
vLLM runs it ~10x faster -- but vLLM and HF do not sample identically, so
comparing a vLLM-prompted arm against an HF baseline would confound the prompt
with the engine. This script therefore regenerates the unprompted baseline too,
and the prompt-vs-baseline comparison is entirely within-engine. Cross-engine
comparisons against the HF `suppress` column are reported as indicative only.

## The prompt

P2 is the strongest form, and the one a practitioner would actually write: it
instructs against the behaviour AND names the marker words. It deliberately says
nothing about length or brevity -- "be concise" changes the task rather than
suppressing doubt, and would not be a fair comparison.

## Resumable

Results are appended to <out>.partial after every chunk and the file is renamed
to <out> only on success, so a killed run loses at most one chunk and re-running
the same command continues from where it stopped.

    uv run python scripts/prompt_baseline.py --set hard
    uv run python scripts/prompt_baseline.py --set easy --n 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suppress_answer import doubt_stats  # noqa: E402

from reasoning_attention.config import MODEL_ID, SamplingDefaults  # noqa: E402
from reasoning_attention.grading import grade  # noqa: E402

# Grading is `\boxed{}`-only (math_verify over the boxed span). A bare P2 makes the
# model answer in prose -- "**Answer:** 26" -- which scores as no answer at all and
# would report a formatting change as an accuracy collapse. Both arms therefore
# carry the SAME format instruction, so the only difference between them is the
# doubt instruction. Verified on a smoke run: without this, 10 of 16 P2 rollouts
# were unboxed and graded wrong while reaching the correct value.
FORMAT = "Give your final answer in \\boxed{}."

P2 = (
    "Solve the problem by working through it once. Commit to each step as you "
    "complete it: do not re-derive, re-check, or reconsider steps you have "
    "already finished. When you reach the answer, state it. Do not write phrases "
    'such as "wait", "hmm", "let me verify", "but actually", or "let me '
    'reconsider".'
)

SETS = {
    "hard": ("data/hard832.jsonl", 12288, "data/prompt/hard"),
    "easy": ("data/correct120_clean.jsonl", 6144, "data/prompt/easy"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--set", choices=sorted(SETS), default="hard")
    p.add_argument("--n", type=int, default=1, help="rollouts per question")
    p.add_argument("--chunk", type=int, default=64, help="questions per checkpoint")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    # 0.95 (the VLLMConfig default) needs more than a desktop session leaves free.
    p.add_argument("--gpu-util", type=float, default=0.86)
    p.add_argument("--dump", type=Path, default=None, help="write generated texts")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    traces_path, max_new, out_stem = SETS[args.set]
    out = Path(f"{out_stem}_p2.csv")
    partial = Path(str(out) + ".partial")
    out.parent.mkdir(parents=True, exist_ok=True)

    rows_done: list[dict[str, str]] = []
    if partial.exists():
        rows_done = list(csv.DictReader(partial.open()))
        print(f"resuming from {partial}: {len(rows_done)} questions done")
    done_ids = {r["question_id"] for r in rows_done}

    traces = [json.loads(line) for line in Path(traces_path).open()]
    if args.limit:
        traces = traces[: args.limit]
    todo = [t for t in traces if t["question_id"] not in done_ids]
    print(f"{len(traces)} questions in {args.set}, {len(todo)} left to do, "
          f"n={args.n} rollouts each, max_new={max_new}")
    if not todo:
        if partial.exists():
            os.replace(partial, out)
            print(f"nothing left; finalised {out}")
        return

    from transformers import AutoTokenizer

    from reasoning_attention.config import VLLMConfig
    from reasoning_attention.serving.vllm_server import build_llm

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    cfg = VLLMConfig()
    cfg = replace(cfg, gpu_memory_utilization=args.gpu_util, max_model_len=max_new + 1024)
    llm = build_llm(cfg)
    from vllm import SamplingParams

    s = SamplingDefaults()
    sp = SamplingParams(
        temperature=s.temperature, top_p=s.top_p, top_k=s.top_k,
        max_tokens=max_new, n=args.n, seed=args.seed,
    )

    def prompt_for(q: str, system: str | None) -> str:
        msgs: list[dict[str, str]] = [
            {"role": "system", "content": f"{system} {FORMAT}" if system else FORMAT}
        ]
        msgs.append({"role": "user", "content": q})
        return str(
            tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )
        )

    pf = partial.open("a" if rows_done else "w", newline="")
    pw: Any = csv.DictWriter(pf, fieldnames=list(rows_done[0])) if rows_done else None

    for start in range(0, len(todo), args.chunk):
        chunk = todo[start : start + args.chunk]
        prompts = [prompt_for(t["question"], None) for t in chunk]
        prompts += [prompt_for(t["question"], P2) for t in chunk]
        outs = llm.generate(prompts, sp)
        m = len(chunk)
        if args.dump:
            with args.dump.open("a") as dh:
                for i2, t2 in enumerate(chunk):
                    dh.write(json.dumps({
                        "question_id": t2["question_id"], "gold": t2["gold"],
                        "plain": outs[i2].outputs[0].text,
                        "p2": outs[m + i2].outputs[0].text}) + "\n")
        new_rows = []
        for i, t in enumerate(chunk):
            row: dict[str, Any] = {
                "question_id": t["question_id"],
                "dataset": t["dataset"],
                "gold": t["gold"],
                "band": t.get("band", "-"),
            }
            for cond, res in (("plain", outs[i]), ("p2", outs[m + i])):
                # per-question RATE over n rollouts, so k>1 is a rate not a bit
                hits = fin = mark = blk = dbt = 0
                toks: list[int] = []
                thinks: list[int] = []
                for o in res.outputs:
                    text = o.text
                    g = grade(text, t["gold"])
                    hits += int(g.is_correct)
                    fin += int(g.has_answer)
                    nm, nb, nd = doubt_stats(text)
                    mark += nm
                    blk += nb
                    dbt += nd
                    toks.append(len(o.token_ids))
                    body = text.split("</think>")[0]
                    thinks.append(
                        len(tok(body, add_special_tokens=False)["input_ids"])
                        if "</think>" in text
                        else -1
                    )
                k = len(res.outputs)
                row.update(
                    {
                        f"{cond}_correct": hits,
                        f"{cond}_n": k,
                        f"{cond}_has_answer": fin,
                        f"{cond}_tokens": sum(toks) // k,
                        f"{cond}_think_tokens": sorted(thinks)[k // 2],
                        f"{cond}_capped": sum(1 for x in toks if x >= max_new - 2),
                        f"{cond}_markers": mark / k,
                        f"{cond}_blocks": blk / k,
                        f"{cond}_doubt_blocks": dbt / k,
                    }
                )
            new_rows.append(row)
        if pw is None:
            pw = csv.DictWriter(pf, fieldnames=list(new_rows[0]))
            pw.writeheader()
        pw.writerows(new_rows)
        pf.flush()
        os.fsync(pf.fileno())
        rows_done.extend(new_rows)  # type: ignore[arg-type]
        done = len(rows_done)
        pc = sum(int(r["plain_correct"]) for r in rows_done)
        p2c = sum(int(r["p2_correct"]) for r in rows_done)
        nn = sum(int(r["plain_n"]) for r in rows_done)
        print(f"[{done}/{len(traces)}]  plain {pc}/{nn}  p2 {p2c}/{nn}", flush=True)

    pf.close()
    os.replace(partial, out)
    print(f"\nwrote {out} ({len(rows_done)} questions)")


if __name__ == "__main__":
    main()
