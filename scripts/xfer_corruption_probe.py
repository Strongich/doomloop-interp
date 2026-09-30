#!/usr/bin/env python3
"""Prefill-corruption probe for the 235B serving config.

Symptom (EXPERIMENT-transfer-235b-log.md): at high concurrency some requests' first
generated token is already garbage -- no `<think>` opener, replies as if no prompt was
seen. That shows up at token 1, so 8 greedy tokens per prompt are enough to count it.

Variants: --no-hook (plain vLLM, our worker extension not installed), --batch
(max_num_seqs), --stream (engine.add_request/step loop, as the runner's --stream) vs
llm.generate, --mbt (max_num_batched_tokens), --pp/--tp.

    uv run python scripts/xfer_corruption_probe.py --pp 6 --batch 320 --stream
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FORMAT = "Give your final answer in \\boxed{}."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B-FP8")
    ap.add_argument("--layer", type=int, default=67)
    ap.add_argument("--pp", type=int, default=6)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--batch", type=int, default=320)
    ap.add_argument("--mbt", type=int, default=8192)
    ap.add_argument("--stream", action="store_true")
    ap.add_argument("--no-hook", action="store_true")
    ap.add_argument("--cohort", default="data/xfer8b/fit_math400.jsonl")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--repeat", type=int, default=1, help="prompts are repeated this many times")
    ap.add_argument("--runner", choices=["v1", "v2"], default="v1",
                    help="v1 = what the steering engine uses; v2 = vLLM 0.22's default for Qwen3")
    ap.add_argument("--quantization", default=None, help="e.g. fp8 (online, for a bf16 model)")
    args = ap.parse_args()

    if args.runner == "v1":
        os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    else:
        assert args.no_hook, "the steering hook needs the V1 runner"
        os.environ.pop("VLLM_USE_V2_MODEL_RUNNER", None)
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(x) for x in open(args.cohort)][: args.n] * args.repeat
    prompts = [tok(tok.apply_chat_template(
        [{"role": "system", "content": FORMAT}, {"role": "user", "content": r["question"].strip()}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True),
        add_special_tokens=False)["input_ids"] for r in rows]
    max_len = max(map(len, prompts)) + 16
    if args.no_hook:
        llm = LLM(model=args.model, dtype="bfloat16", tensor_parallel_size=args.tp,
                  pipeline_parallel_size=args.pp, enforce_eager=True, async_scheduling=False,
                  enable_prefix_caching=False, enable_chunked_prefill=True,
                  max_model_len=max_len, max_num_seqs=args.batch,
                  max_num_batched_tokens=args.mbt, gpu_memory_utilization=0.90,
                  quantization=args.quantization)
    else:
        from reasoning_attention.serving.vllm_steering import build_steering_llm

        is_b = [i for i in range(len(tok)) if tok.convert_ids_to_tokens(i).count("Ċ") >= 2]
        llm = build_steering_llm(args.model, args.layer, is_b,
                                 tok.convert_tokens_to_ids("</think>"), max_len, args.batch,
                                 0.90, args.mbt, args.pp, args.tp)
    sp = SamplingParams(max_tokens=8, temperature=0.0)
    t0 = time.monotonic()
    if args.stream:
        eng = llm.llm_engine
        ids = {}
        for i, p in enumerate(prompts):
            ids[eng.add_request(f"p{i}", {"prompt_token_ids": p}, sp)] = i
            ids[f"p{i}"] = i
        texts: dict[int, str] = {}
        while eng.has_unfinished_requests():
            for o in eng.step():
                if o.finished:
                    texts[ids[o.request_id]] = o.outputs[0].text
        out = [texts[i] for i in range(len(prompts))]
    else:
        res = llm.generate([{"prompt_token_ids": p} for p in prompts], sp, use_tqdm=False)
        out = [r.outputs[0].text for r in res]
    bad = [i for i, t in enumerate(out) if not t.startswith("<think>")]
    print(f"PROBE model={args.model.split('/')[-1]} runner={args.runner} q={args.quantization} "
          f"pp={args.pp} tp={args.tp} batch={args.batch} mbt={args.mbt} "
          f"stream={args.stream} hook={not args.no_hook} n={len(out)}: "
          f"bad {len(bad)} ({100 * len(bad) / len(out):.1f}%) first bad idx {bad[:12]} "
          f"[{time.monotonic() - t0:.0f}s]", flush=True)
    for i in bad[:3]:
        print("   ", i, repr(out[i]))


if __name__ == "__main__":
    main()
