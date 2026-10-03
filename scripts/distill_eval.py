#!/usr/bin/env python3
r"""Student evaluation, LOCKED PROTOCOL v2 §L7: plain vLLM, **no steering, no hooks**.

Same prompt (system FORMAT + user question, thinking template), sampler (T 0.6, top_p
0.95, top_k 20), 32,768-token cap and D69 grader as the teacher generation (§L3), and
the same `sha256(question_id:seed)` request seeds for every model. LoRA students are
evaluated from their merged checkpoint (distill_train.py saves merged weights).

Output: <outdir>/rollouts.jsonl (resumable) + rollouts.csv + run_manifest.json. The
columns match reasoning_policy_vllm.py so the existing reports read them.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/distill_eval.py \
        --model data/distill/models/sft_short_lora --name sft_short_lora \
        --sets math500 confirm400 gsm8k_test val
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from importlib.metadata import version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from branch_continue_vllm import digest, read_journal, request_seed  # noqa: E402

SETS = {
    "math500": "data/policy/math500.jsonl",
    "confirm400": "data/xfer8b/confirm_mathtest400.jsonl",
    "gsm8k_test": "data/distill/eval_gsm8k_test.jsonl",
    "val": "data/distill/val.jsonl",
    "aime_amc": "data/distill/eval_aime_amc.jsonl",
    "mathfresh": "data/distill_v3/mathtest_fresh1000.jsonl",  # PROTOCOL v3 §V6
    "mathpipe": "data/pipeline_sft/mathtest_pipe1000.jsonl",  # EXPERIMENT-pipeline-sft §P4
}
SEEDS = {"aime_amc": 8}  # §L7: 70 x 8; everything else 4
FORMAT = "Give your final answer in \\boxed{}."
FIELDS = ["set", "question_id", "dataset", "gold", "level", "seed", "correct", "has_answer",
          "status", "total_tokens", "think_tokens", "answer_tokens", "capped", "closed",
          "boundaries", "markers", "doubt_blocks", "doubt_rate", "prompt_tokens",
          "request_seed", "finish_reason"]


def ensure_eval_files() -> None:
    """GSM8K official test (combined-index ids gsm8k:7473..8791) and AIME2025+AMC23."""
    from reasoning_attention.data.math_datasets import load_one

    p = ROOT / SETS["gsm8k_test"]
    if not p.exists():
        ds = load_one("gsm8k")
        rows = [{"question_id": f"gsm8k:{i}", "dataset": "gsm8k", "question": r["question"],
                 "gold": r["answer"]} for i, r in enumerate(ds) if r["split"] == "test"]
        assert len(rows) == 1319 and rows[0]["question_id"] == "gsm8k:7473"
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    p = ROOT / SETS["aime_amc"]
    if not p.exists():
        rows = []
        for name in ("aime2025", "amc23"):
            for i, r in enumerate(load_one(name)):
                rows.append({"question_id": f"{name}:{i}", "dataset": name,
                             "question": r["question"], "gold": r["answer"]})
        p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF id or a local (merged) checkpoint")
    ap.add_argument("--name", required=True)
    ap.add_argument("--sets", nargs="+", default=["math500", "confirm400", "gsm8k_test", "val"])
    ap.add_argument("--outroot", type=Path, default=ROOT / "data/distill/eval")
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N questions per set")
    args = ap.parse_args()

    # Same runner family as the teacher generation; pp=1 so D70 does not apply.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    ensure_eval_files()
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from reasoning_attention.config import SamplingDefaults
    from reasoning_attention.data.math_datasets import build_messages
    from reasoning_attention.grading import grade
    from suppress_answer import NEWLINE_CHAR, doubt_stats

    tok = AutoTokenizer.from_pretrained(args.model)
    sampling = SamplingDefaults()
    boundary_set = {i for i in range(len(tok))
                    if tok.convert_ids_to_tokens(i).count(NEWLINE_CHAR) >= 2}
    close_id = int(tok.convert_tokens_to_ids("</think>"))
    stop_ids = sorted({int(tok.eos_token_id), int(tok.convert_tokens_to_ids("<|endoftext|>"))})

    def prompt_ids(question: str) -> list[int]:
        messages = [{"role": "system", "content": FORMAT}, *build_messages(question)]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
        return list(tok(text, add_special_tokens=False)["input_ids"])

    outdir = args.outroot / args.name
    outdir.mkdir(parents=True, exist_ok=True)
    work = []  # (set, row, seed)
    for s in args.sets:
        rows = [json.loads(x) for x in (ROOT / SETS[s]).read_text().splitlines() if x.strip()]
        rows = rows[: args.limit] if args.limit else rows
        work += [(s, r, k) for k in range(SEEDS.get(s, 4)) for r in rows]
    manifest = {
        "runner": "distill_eval", "protocol": "LOCKED PROTOCOL v2 §L7", "model": args.model,
        "versions": {p: version(p) for p in ("vllm", "torch", "transformers")},
        "sets": {s: digest(ROOT / SETS[s]) for s in args.sets},
        "seeds": {s: SEEDS.get(s, 4) for s in args.sets},
        "format_instruction": FORMAT, "max_new_tokens": args.max_new_tokens,
        "temperature": sampling.temperature, "top_p": sampling.top_p, "top_k": sampling.top_k,
        "max_num_seqs": args.batch, "max_num_batched_tokens": args.max_num_batched_tokens,
        "enforce_eager": False, "prefix_caching": False, "steering": None,
        "code": digest(Path(__file__)), "grader": digest(ROOT / "src/reasoning_attention/grading.py"),
        "seed_scheme": "sha256(question_id:seed)", "limit": args.limit,
    }
    mpath = outdir / "run_manifest.json"
    if mpath.exists() and json.loads(mpath.read_text()) != manifest:
        old = json.loads(mpath.read_text())
        raise ValueError(f"Manifest differs in {sorted(k for k in manifest if old.get(k) != manifest[k])}")
    mpath.write_text(json.dumps(manifest, indent=2) + "\n")
    journal = outdir / "rollouts.jsonl"
    records = read_journal(journal)
    done = {(r["set"], r["question_id"], r["seed"]) for r in records}
    todo = [w for w in work if (w[0], w[1]["question_id"], w[2]) not in done]
    print(f"{args.name}: {len(work)} rollouts, {len(done)} resumed, {len(todo)} to run", flush=True)

    def export(final: bool) -> None:
        p = outdir / ("rollouts.csv" if final else "rollouts.csv.partial")
        with p.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows({k: r[k] for k in FIELDS} for r in records)

    if todo:
        prompts = {w[1]["question_id"]: prompt_ids(w[1]["question"]) for w in todo}
        llm = LLM(model=args.model, dtype="bfloat16", enable_prefix_caching=False,
                  max_model_len=max(map(len, prompts.values())) + args.max_new_tokens,
                  max_num_seqs=args.batch, max_num_batched_tokens=args.max_num_batched_tokens,
                  gpu_memory_utilization=args.gpu_memory_utilization, seed=0)
        eng = llm.llm_engine
        pending = {}
        for s, r, k in todo:
            aid = f"{s}|{r['question_id']}|{k}"
            sp = SamplingParams(temperature=sampling.temperature, top_p=sampling.top_p,
                                top_k=sampling.top_k, max_tokens=args.max_new_tokens,
                                seed=request_seed(r["question_id"], k), stop_token_ids=stop_ids)
            rid = eng.add_request(aid, {"prompt_token_ids": prompts[r["question_id"]]}, sp)
            pending[aid] = pending[rid] = (s, r, k)
        t0, n = time.monotonic(), 0
        with journal.open("a") as out:
            while eng.has_unfinished_requests():
                fin = [o for o in eng.step() if o.finished]
                for o in fin:
                    s, r, k = pending[o.request_id]
                    gen = o.outputs[0]
                    ids = list(gen.token_ids)
                    close = ids.index(close_id) if close_id in ids else None
                    think_end = close if close is not None else len(ids)
                    text = tok.decode(ids, skip_special_tokens=False)
                    g = grade(text, r["gold"])
                    markers, _, doubt_blocks = doubt_stats(text)
                    nb = sum(t in boundary_set for t in ids[:think_end])
                    rec = {
                        "set": s, "question_id": r["question_id"],
                        "dataset": r.get("dataset", ""), "gold": str(r["gold"]),
                        "level": r.get("level", ""), "seed": k,
                        "correct": int(g.is_correct), "has_answer": int(g.has_answer),
                        "status": g.status, "total_tokens": len(ids),
                        "think_tokens": think_end + int(close is not None),
                        "answer_tokens": len(ids) - think_end - int(close is not None),
                        "capped": int(gen.finish_reason == "length"),
                        "closed": int(close is not None), "boundaries": nb,
                        "markers": markers, "doubt_blocks": doubt_blocks,
                        "doubt_rate": round(doubt_blocks / nb, 4) if nb else -1.0,
                        "prompt_tokens": len(o.prompt_token_ids),
                        "request_seed": request_seed(r["question_id"], k),
                        "finish_reason": gen.finish_reason, "text": text,
                    }
                    out.write(json.dumps(rec) + "\n")
                    records.append(rec)
                    n += 1
                    if n % 500 == 0:
                        out.flush()
                        export(False)
                        print(f"{args.name}: {n}/{len(todo)} "
                              f"{(time.monotonic() - t0) / n:.2f}s/rollout", flush=True)
    export(True)
    by: dict[str, list] = {}
    for r in records:
        by.setdefault(r["set"], []).append(r)
    for s, rs in by.items():
        print(f"{args.name} {s}: acc {sum(r['correct'] for r in rs) / len(rs):.4f} "
              f"tokens {sum(r['total_tokens'] for r in rs) / len(rs):.0f} "
              f"capped {sum(r['capped'] for r in rs) / len(rs):.4f} n {len(rs)}", flush=True)


if __name__ == "__main__":
    main()
