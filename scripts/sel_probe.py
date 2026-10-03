#!/usr/bin/env python3
r"""EXPERIMENT-selective-doubt.md §S3: forced-answer probe at EVERY paragraph boundary.

For each selected rollout: append `</think>` + the answer stem at every boundary inside
<think> (and at the end of thinking), decode greedily for 96 tokens, read the box. Same
method as probe_candidates.py (boxed_value / normalize; correctness by the D69 grader),
on the stored token ids and the generation prompt (FORMAT system turn). No steering.

Prefix caching: per chunk, the end-of-thinking probe of every rollout runs first, which
leaves the whole trace's KV cached; the boundary probes of that chunk then reuse it.

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/sel_probe.py \
        --journals 'data/distill/gen/shard*/rollouts.jsonl' \
        --select data/selective/source1000.jsonl --questions data/distill/train.jsonl \
        --shard 0/6 --out data/selective/probes/source_s0.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sel_common import Paragraphs  # noqa: E402


def key_of(r: dict) -> str:
    return f"{r['question_id']}|{r.get('policy', 'base')}|{r['seed']}"


def load_selected(journals: str, select: Path, shard: str) -> list[dict]:
    want = {key_of(json.loads(x)) for x in select.open()}
    recs = {}
    for f in sorted(glob.glob(str(ROOT / journals))):
        for line in open(f):
            r = json.loads(line)
            k = key_of(r)
            if k in want:
                recs[k] = r
    missing = want - set(recs)
    if missing:
        raise SystemExit(f"{len(missing)} selected rollouts not found, e.g. {sorted(missing)[:3]}")
    i, n = map(int, shard.split("/"))
    # length-sorted round robin balances the shards
    order = sorted(recs.values(), key=lambda r: (-len(r["token_ids"]), key_of(r)))
    return order[i::n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--journals", required=True)
    ap.add_argument("--select", type=Path, required=True,
                    help="jsonl with question_id, seed[, policy]")
    ap.add_argument("--questions", type=Path, required=True, help="pool/cohort jsonl")
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--probe-tokens", type=int, default=96)
    ap.add_argument("--chunk-tokens", type=int, default=240_000,
                    help="trace tokens per chunk (must stay resident in the KV cache)")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.88)
    args = ap.parse_args()
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from probe_candidates import boxed_value, normalize
    from reasoning_attention.config import MODEL_ID
    from reasoning_attention.grading import grade

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    P = Paragraphs(tok)
    qs = {json.loads(x)["question_id"]: json.loads(x) for x in args.questions.open()}
    todo = load_selected(args.journals, args.select, args.shard)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(x)["key"] for x in args.out.open()} if args.out.exists() else set()
    todo = [r for r in todo if key_of(r) not in done]
    print(f"shard {args.shard}: {len(todo)} rollouts to probe ({len(done)} resumed)", flush=True)
    if not todo:
        return

    items = []
    for r in todo:
        ids = r["token_ids"]
        prompt = P.prompt_ids(qs[r["question_id"]]["question"])
        if len(prompt) != r["prompt_tokens"]:
            raise SystemExit(f"prompt length mismatch {key_of(r)}")
        sp = P.split(ids)
        end_cut = max(sp["think_end"] - 1, 0)
        cuts = sorted(set(sp["boundaries"]) | {end_cut})
        items.append({"r": r, "prompt": prompt, "split": sp, "cuts": cuts, "end_cut": end_cut})

    max_len = max(len(it["prompt"]) + len(it["r"]["token_ids"]) for it in items) + 64 + args.probe_tokens
    llm = LLM(model=MODEL_ID, dtype="bfloat16", enable_prefix_caching=True,
              max_model_len=max_len, max_num_seqs=512, max_num_batched_tokens=16384,
              gpu_memory_utilization=args.gpu_memory_utilization, seed=0)
    # Stop at the first `$` or newline (kept in the text): boxed_value ends the expression
    # there anyway, so the candidate is unchanged unless such a character sits inside nested
    # braces (§S9 item 1). Without it every probe decodes all 96 tokens over a long context.
    sp_greedy = SamplingParams(temperature=0.0, max_tokens=args.probe_tokens,
                               stop=["$", "\n"], include_stop_str_in_output=True)

    def run(prompts: list[list[int]]) -> list[str]:
        outs = llm.generate([{"prompt_token_ids": p} for p in prompts], sp_greedy, use_tqdm=False)
        return [o.outputs[0].text for o in outs]

    t0, n_probes = time.monotonic(), 0
    with args.out.open("a") as fh:
        i = 0
        while i < len(items):
            chunk, budget = [], 0
            while i < len(items) and (not chunk or budget + len(items[i]["r"]["token_ids"]) <= args.chunk_tokens):
                chunk.append(items[i])
                budget += len(items[i]["r"]["token_ids"])
                i += 1
            # phase A: end-of-thinking probes (longest prefixes) -> whole traces cached
            raw_end = run([P.probe_ids(it["prompt"], it["r"]["token_ids"], it["end_cut"]) for it in chunk])
            # phase B: every other boundary
            jobs = [(j, c) for j, it in enumerate(chunk) for c in it["cuts"] if c != it["end_cut"]]
            raw_b = run([P.probe_ids(chunk[j]["prompt"], chunk[j]["r"]["token_ids"], c) for j, c in jobs])
            raws: dict = {(j, it["end_cut"]): raw_end[j] for j, it in enumerate(chunk)}
            raws.update({jc: x for jc, x in zip(jobs, raw_b)})
            for j, it in enumerate(chunk):
                r = it["r"]
                gold = str(qs[r["question_id"]]["gold"])
                probes = {}
                for c in it["cuts"]:
                    cand = normalize(boxed_value(raws[(j, c)]))
                    ok = bool(cand) and bool(grade("\\boxed{" + cand + "}", gold).is_correct)
                    probes[c] = {"cand": cand, "correct": int(ok), "raw": raws[(j, c)][:120]}
                sp = it["split"]
                fh.write(json.dumps({
                    "key": key_of(r), "question_id": r["question_id"], "seed": r["seed"],
                    "policy": r.get("policy", "base"), "gold": gold,
                    "level": qs[r["question_id"]].get("level", 0),
                    "final_correct": r["correct"], "capped": r["capped"],
                    "total_tokens": r["total_tokens"], "prompt_tokens": r["prompt_tokens"],
                    "doubt_blocks_col": r["doubt_blocks"],
                    "boundaries": sp["boundaries"], "doubt": sp["doubt"], "opens": sp["opens"],
                    "think_end": sp["think_end"], "end_cut": it["end_cut"],
                    "probes": {str(c): v for c, v in probes.items()},
                }) + "\n")
            fh.flush()
            n_probes += len(jobs) + len(chunk)
            el = time.monotonic() - t0
            print(f"shard {args.shard}: {i}/{len(items)} rollouts, {n_probes} probes, "
                  f"{n_probes / el:.1f} probes/s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
