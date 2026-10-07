#!/usr/bin/env python3
r"""DEER (Yang et al. 2025, arXiv 2504.15895, "Dynamic Early Exit in Reasoning Models"),
EXPERIMENT-reviewer-baselines.md §B4b, ported from the official vLLM script for Qwen3
(github.com/iie-ycx/DEER @ c9dd19f, `vllm-deer-qwen3.py`, `bashes/bash-vllm-deer-qwen3.sh`).

Per rollout, a state machine over one vLLM engine queue:

  think   sample (our sampler: T 0.6, top_p 0.95, top_k 20) until the STRING "Wait" (the
          official `stop=["Wait", "</think>", eos]`), `</think>`, or the thinking budget
          int(max_len * think_ratio). After `max_judge_steps` (10) checks, only `</think>`
          or the budget stop a chunk.
  trial   at a "Wait" stop (or the budget): context = prompt + thinking up to, not
          including, "Wait" + "\n**Final Answer**\n\\boxed"; decode greedily up to 20
          tokens, stopping at `</think>`.
          C = exp(mean log p(top token)) over trial tokens 1..n-1 (the first, "{", is
          skipped; `</think>` included); C = 0 unless the trial ends in `</think>`.
  decide  C > lambda (0.95), or thinking >= budget - 50: exit -- thinking up to "Wait" +
          "\n</think>\n\n", then the conclusion is sampled with the remaining budget.
          Otherwise the branch is discarded and thinking continues from the "Wait".
  natural `</think>` sampled by the model: the conclusion continues from it, as in base.

Token accounting: `total_tokens` is the delivered message (thinking + close + conclusion);
`trial_tokens` are the generated trial-answer tokens, real compute, reported separately
and in `total_with_trial`. Grading (D69) reads the delivered message only; trial answers
are journaled per check (position g, C, text, exit) for the audit.

Work is token-level: the model's own "Wait" token is kept on continue. Only when "Wait" is
not the whole of the stop token (e.g. "Waiting") is the chunk re-encoded from text, which
is what the official string-level script does every step; those are counted (`retok`).

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/baselines_deer.py --model Qwen/Qwen3-1.7B \
        --cohort data/xfer8b/confirm_mathtest400.jsonl --outdir data/baselines/a17_confirm_deer \
        --seeds 4
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from branch_continue_vllm import digest, read_journal, request_seed  # noqa: E402
from reasoning_policy_vllm import FIELDS, FORMAT  # noqa: E402

CONTINUE_STR = "Wait"  # official --points 1
ANSWER_PROMPT = "\n**Final Answer**\n\\boxed"
EXIT_STR = "\n</think>\n\n"
DEER_FIELDS = FIELDS + ["trial_tokens", "total_with_trial", "deer_checks", "deer_exit",
                        "exit_g", "retok"]


def confidence(token_ids: list[int], logprobs: list[dict], close_id: int) -> float:
    """Official `calculate_average_max_prob_from_logprobs(policy='avg2')`."""
    n = len(token_ids)
    if n < 2 or token_ids[-1] != close_id:
        return 0.0
    s = 0.0
    for i in range(1, n):
        s += max(logprobs[i][token_ids[i]].logprob, math.log(1e-10))
    return math.exp(s / (n - 1))


def sub_seed(question_id: str, seed: int, k: int) -> int:
    h = hashlib.sha256(f"{question_id}:{seed}:deer{k}".encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--seed-start", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--threshold", type=float, default=0.95)
    ap.add_argument("--think-ratio", type=float, default=0.8)
    ap.add_argument("--max-judge-steps", type=int, default=10)
    ap.add_argument("--trial-max-tokens", type=int, default=20)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--checkpoint-every", type=int, default=64)
    args = ap.parse_args()

    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from reasoning_attention.config import SamplingDefaults
    from reasoning_attention.data.math_datasets import build_messages
    from reasoning_attention.grading import grade
    from suppress_answer import NEWLINE_CHAR, doubt_stats

    tok = AutoTokenizer.from_pretrained(args.model)
    sampling = SamplingDefaults()
    close_id = int(tok.convert_tokens_to_ids("</think>"))
    stop_ids = sorted({int(tok.eos_token_id), int(tok.convert_tokens_to_ids("<|endoftext|>"))})
    boundary_set = {i for i in range(len(tok)) if tok.convert_ids_to_tokens(i).count(NEWLINE_CHAR) >= 2}
    answer_ids = tok.encode(ANSWER_PROMPT, add_special_tokens=False)
    exit_ids = tok.encode(EXIT_STR, add_special_tokens=False)
    wait_ids = tok.encode(CONTINUE_STR, add_special_tokens=False)
    think_limit = int(args.max_new_tokens * args.think_ratio)

    rows = [json.loads(x) for x in args.cohort.read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]

    def prompt_ids(question: str) -> list[int]:
        messages = [{"role": "system", "content": FORMAT}, *build_messages(question)]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
        return list(tok(text, add_special_tokens=False)["input_ids"])

    prompts = {r["question_id"]: prompt_ids(r["question"]) for r in rows}
    max_len = max(len(p) for p in prompts.values()) + args.max_new_tokens + len(answer_ids) + \
        args.trial_max_tokens + 16

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "runner": "baselines_deer", "method": "DEER (official vllm-deer-qwen3.py @ c9dd19f)",
        "model": args.model, "versions": {p: version(p) for p in ("vllm", "torch", "transformers")},
        "cohort_sha256": digest(args.cohort), "limit": args.limit,
        "question_ids": [r["question_id"] for r in rows], "seeds": args.seeds,
        "seed_start": args.seed_start, "max_new_tokens": args.max_new_tokens,
        "threshold": args.threshold, "think_ratio": args.think_ratio, "think_limit": think_limit,
        "max_judge_steps": args.max_judge_steps, "trial_max_tokens": args.trial_max_tokens,
        "continue_str": CONTINUE_STR, "answer_prompt": ANSWER_PROMPT, "exit_str": EXIT_STR,
        "confidence": "avg2 (geometric mean of top-token p over trial tokens 1..n-1; 0 unless "
                      "ending in </think>)",
        "trial_decoding": "greedy", "format_instruction": FORMAT,
        "temperature": sampling.temperature, "top_p": sampling.top_p, "top_k": sampling.top_k,
        "max_num_seqs": args.batch, "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization, "prefix_caching": True,
        "tensor_parallel_size": args.tp,
        "code": {p: digest(ROOT / p) for p in ("scripts/baselines_deer.py",
                                               "src/reasoning_attention/grading.py")},
        "seed_scheme": "first chunk sha256(question_id:seed) as every arm; later requests "
                       "sha256(question_id:seed:deer<k>)",
    }
    manifest_path = args.outdir / "run_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != manifest:
            changed = sorted(k for k in set(old) | set(manifest) if old.get(k) != manifest.get(k))
            raise ValueError(f"Manifest differs in {changed}; use a new output directory")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    journal = args.outdir / "rollouts.jsonl"
    records = read_journal(journal)
    done = {(r["question_id"], r["seed"]) for r in records}

    def export(final: bool = False) -> None:
        path = args.outdir / ("rollouts.csv" if final else "rollouts.csv.partial")
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=DEER_FIELDS)
            w.writeheader()
            w.writerows({k: r.get(k, "") for k in DEER_FIELDS} for r in records)
        os.replace(tmp, path)

    todo = [(r, s) for s in range(args.seed_start, args.seeds) for r in rows
            if (r["question_id"], s) not in done]
    total = len(rows) * (args.seeds - args.seed_start)
    print(f"{len(rows)} questions x {args.seeds - args.seed_start} seeds = {total}; "
          f"{len(done)} resumed; think limit {think_limit}; max context {max_len}", flush=True)
    if not todo:
        export(final=True)
        return

    llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp,
              enable_prefix_caching=True, enable_chunked_prefill=True, max_model_len=max_len,
              max_num_seqs=args.batch, max_num_batched_tokens=args.max_num_batched_tokens,
              gpu_memory_utilization=args.gpu_memory_utilization, max_logprobs=1)
    engine = llm.llm_engine

    def think_params(st: dict) -> Any:
        k = st["requests"]
        stop = [CONTINUE_STR] if st["steps"] < args.max_judge_steps else []
        return SamplingParams(
            temperature=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k,
            max_tokens=think_limit - len(st["gen"]), stop=stop,
            stop_token_ids=[close_id, *stop_ids], include_stop_str_in_output=True,
            seed=request_seed(st["row"]["question_id"], st["seed"]) if k == 0 else
            sub_seed(st["row"]["question_id"], st["seed"], k))

    def answer_params(st: dict, budget: int) -> Any:
        return SamplingParams(
            temperature=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k,
            max_tokens=budget, stop_token_ids=stop_ids,
            seed=sub_seed(st["row"]["question_id"], st["seed"], st["requests"]))

    trial_params = SamplingParams(temperature=0.0, max_tokens=args.trial_max_tokens,
                                  stop_token_ids=[close_id], logprobs=1)
    alias: dict[str, str] = {}
    states: dict[str, dict] = {}

    def submit(st: dict, phase: str, ctx: list[int], params: Any) -> None:
        rid = f"{st['key']}#{st['requests']}{phase}"
        st["requests"] += 1
        st["phase"] = phase
        out_id = engine.add_request(rid, {"prompt_token_ids": ctx}, params)
        alias[rid] = st["key"]
        if out_id is not None:
            alias[str(out_id)] = st["key"]

    def start_answer(st: dict, think_ids: list[int], reason: str) -> None:
        st["think"] = think_ids
        st["exit"] = reason
        budget = args.max_new_tokens - len(think_ids)
        if budget <= 0:
            st["answer"], st["answer_finish"] = [], "length"
            finalize(st)
            return
        submit(st, "answer", prompts[st["row"]["question_id"]] + think_ids,
               answer_params(st, budget))

    def start_think(st: dict) -> None:
        if think_limit - len(st["gen"]) <= 0:
            start_answer(st, st["gen"] + exit_ids, "think_limit")
            return
        submit(st, "think", prompts[st["row"]["question_id"]] + st["gen"], think_params(st))

    out = journal.open("a")
    started = time.monotonic()
    completed = 0

    def finalize(st: dict) -> None:
        nonlocal completed
        row = st["row"]
        ids = st["think"] + st["answer"]
        text = tok.decode(ids, skip_special_tokens=False)
        g = grade(text, row["gold"])
        close = ids.index(close_id) if close_id in ids else None
        think_end = close if close is not None else len(ids)
        markers, _, doubt_blocks = doubt_stats(text)
        n_bound = sum(t in boundary_set for t in ids[:think_end])
        trial = sum(c["n"] for c in st["checks"])
        rec = {
            "question_id": row["question_id"], "dataset": row.get("dataset", ""),
            "gold": str(row["gold"]), "policy": "DEER", "direction": "", "alpha": 0.0,
            "delay": 0, "brevity": "", "seed": st["seed"], "correct": int(g.is_correct),
            "has_answer": int(g.has_answer), "status": g.status, "total_tokens": len(ids),
            "think_tokens": think_end + int(close is not None),
            "answer_tokens": len(ids) - think_end - int(close is not None),
            "capped": int(st["answer_finish"] == "length"),
            "closed": int(close is not None), "boundaries": n_bound, "eligible_sites": 0,
            "injections": 0, "first_injection_g": -1, "markers": markers,
            "doubt_blocks": doubt_blocks,
            "doubt_rate": round(doubt_blocks / n_bound, 4) if n_bound else -1.0,
            "trial_tokens": trial, "total_with_trial": len(ids) + trial,
            "deer_checks": len(st["checks"]), "deer_exit": st["exit"],
            "exit_g": st.get("exit_g", -1), "retok": st["retok"],
            "checks": st["checks"], "text": text, "token_ids": ids,
            "prompt_tokens": len(prompts[row["question_id"]]),
            "request_seed": request_seed(row["question_id"], st["seed"]),
            "finish_reason": st["answer_finish"], "requests": st["requests"],
        }
        out.write(json.dumps(rec) + "\n")
        records.append(rec)
        completed += 1
        del states[st["key"]]

    for r, s in todo:
        key = f"{r['question_id']}|DEER|{s}"
        st = {"key": key, "row": r, "seed": s, "gen": [], "steps": 0, "checks": [],
              "requests": 0, "retok": 0, "pre": None}
        states[key] = st
        start_think(st)

    n_total = len(todo)
    while engine.has_unfinished_requests():
        finished = [o for o in engine.step() if o.finished]
        for o in finished:
            key = alias.pop(o.request_id, None)
            if key is None or key not in states:
                raise RuntimeError(f"Unknown finished request {o.request_id}")
            st = states[key]
            c = o.outputs[0]
            ids = list(c.token_ids)
            if st["phase"] == "think":
                if ids and ids[-1] == close_id:
                    st["gen"] += ids
                    start_answer(st, st["gen"], "natural")
                elif ids and ids[-1] in stop_ids:
                    st["gen"] += ids
                    st["answer"], st["answer_finish"], st["think"] = [], "stop", st["gen"]
                    st["exit"] = "eos"
                    finalize(st)
                else:
                    if c.stop_reason == CONTINUE_STR:
                        last = tok.decode(ids[-1:])
                        if last.strip() == CONTINUE_STR and last.endswith(CONTINUE_STR):
                            pre, cont = st["gen"] + ids[:-1], st["gen"] + ids
                        else:  # "Wait" not a whole token: re-encode as the official does
                            t = tok.decode(ids, skip_special_tokens=False)
                            j = t.rindex(CONTINUE_STR)
                            pre = st["gen"] + tok.encode(t[:j], add_special_tokens=False)
                            cont = pre + wait_ids
                            st["retok"] += 1
                        st["steps"] += 1
                    else:  # thinking budget
                        pre = cont = st["gen"] + ids
                    st["pre"], st["cont"] = pre, cont
                    submit(st, "trial", prompts[st["row"]["question_id"]] + pre + answer_ids,
                           trial_params)
            elif st["phase"] == "trial":
                conf = confidence(ids, c.logprobs, close_id)
                pre = st["pre"]
                limit_hit = len(pre) >= think_limit - 50
                exit_now = conf > args.threshold or limit_hit
                st["checks"].append({"g": len(pre) + 1, "C": round(conf, 6), "n": len(ids),
                                     "trial": tok.decode(ids, skip_special_tokens=False),
                                     "exit": int(exit_now)})
                if exit_now:
                    st["exit_g"] = len(pre) + 1
                    start_answer(st, pre + exit_ids,
                                 "confident" if conf > args.threshold else "think_limit")
                else:
                    st["gen"] = st["cont"]
                    start_think(st)
            else:  # answer
                st["answer"] = ids
                st["answer_finish"] = c.finish_reason
                finalize(st)
        if finished:
            out.flush()
            os.fsync(out.fileno())
            if completed and completed % args.checkpoint_every < len(finished):
                export()
                el = time.monotonic() - started
                print(f"deer {completed}/{n_total}; {el / completed:.2f}s/rollout; "
                      f"{len(states)} active", flush=True)
    out.close()
    if states:
        raise RuntimeError(f"{len(states)} rollouts never finished")
    export(final=True)
    print(f"Wrote {args.outdir / 'rollouts.csv'}", flush=True)


if __name__ == "__main__":
    main()
