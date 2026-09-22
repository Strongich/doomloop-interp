#!/usr/bin/env python3
"""Derive per-block suppression deltas over MANY traces, saving them raw.

Every direction the study has used so far varied three things at once — how many
deltas were averaged, how many distinct traces they came from, and where in the
trace they sat. Finding 5's vector is 297 deltas from 297 traces, each the FIRST
doubt boundary; a single-trace vector is ~40 deltas from 1 trace at ALL depths.
Nothing separates count from diversity from depth.

This derives once over N traces at all depths and stores the deltas **raw**, each
tagged with its trace and its depth index. Every vector variant is then free
arithmetic over subsets of that pool, so only the generation arms cost GPU.

    uv run python scripts/derive_pool.py --n-per-dataset 14 --out data/pool/raw.pt
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from derive_direction import doubt_boundaries  # noqa: E402
from steer_demo import (  # noqa: E402
    CONTINUE_PARAGRAPH,
    edit_explanation,
    load_ar,
    reconstruct,
)

from reasoning_attention.config import MODEL_ID, NLAConfig  # noqa: E402
from reasoning_attention.nla.arch import inner_transformer  # noqa: E402
from reasoning_attention.tokenview import _chat_header  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/correct_sample.jsonl"))
    p.add_argument("--out", type=Path, default=Path("data/pool/raw.pt"))
    p.add_argument("--n-per-dataset", type=int, default=14)
    p.add_argument("--min-blocks", type=int, default=6)
    p.add_argument("--max-blocks", type=int, default=40)
    p.add_argument("--max-prefix", type=int, default=4000)
    p.add_argument("--av", type=Path, default=Path("checkpoints/av_rl_ep1"))
    p.add_argument("--ar", type=Path, default=Path("checkpoints/ar_rl_ep1"))
    p.add_argument("--av-max-new", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NLAConfig()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reasoning_attention.nla.model import NLA

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    traces = [json.loads(line) for line in args.traces.open()]

    # Random among the eligible, never the top-N by block count: picking the
    # doubtiest traces would bias the pool toward unusually doubt-heavy reasoning.
    by_ds: dict[str, list[dict[str, Any]]] = {}
    for t in traces:
        by_ds.setdefault(t["dataset"], []).append(t)
    chosen: list[dict[str, Any]] = []
    for ds, ts in sorted(by_ds.items()):
        rng = random.Random(args.seed)
        rng.shuffle(ts)
        picked = 0
        for t in ts:
            b = doubt_boundaries(t["response"], tokenizer, args.max_prefix)
            if len(b) < args.min_blocks:
                continue
            t["_bounds"] = b[: args.max_blocks]
            chosen.append(t)
            picked += 1
            if picked >= args.n_per_dataset:
                break
        print(f"{ds:<9} selected {picked} traces")
    print(f"total {len(chosen)} traces, "
          f"{sum(len(t['_bounds']) for t in chosen)} boundaries\n")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    trunk = inner_transformer(model)
    nla = NLA.av_only(str(args.av))
    nla.av.eval()
    ar_tok, ar_backbone, affine = load_ar(args.ar)

    captured: dict[str, Any] = {"h": None, "site": -1}

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > captured["site"]:
            captured["h"] = hidden[0, captured["site"]].detach().float().cpu()
        return output

    handle = trunk.layers[cfg.extraction_layer].register_forward_hook(hook)

    deltas: list[torch.Tensor] = []
    meta: list[dict[str, Any]] = []
    try:
        for ti, t in enumerate(chosen, 1):
            header = _chat_header(tokenizer, t["question"])
            n_header = len(tokenizer(header, add_special_tokens=False)["input_ids"])
            for depth, (tok_i, cut) in enumerate(t["_bounds"]):
                enc = tokenizer(header + t["response"][:cut], return_tensors="pt").to("cuda")
                if enc["input_ids"].shape[1] > args.max_prefix:
                    continue
                captured["site"] = n_header + tok_i
                with torch.no_grad():
                    trunk(**enc)
                h = captured["h"]
                if h is None:
                    continue
                e0 = nla.verbalize(
                    h, max_new_tokens=args.av_max_new, return_explanation=True
                ).strip()
                try:
                    ed = edit_explanation(e0, CONTINUE_PARAGRAPH)
                except SystemExit:
                    continue
                with torch.no_grad():
                    a = reconstruct(ar_tok, ar_backbone, affine, e0)
                    d = (reconstruct(ar_tok, ar_backbone, affine, ed) - a).float().cpu()
                deltas.append(d)
                meta.append(
                    {
                        "question_id": t["question_id"],
                        "rollout_index": t["rollout_index"],
                        "dataset": t["dataset"],
                        "depth": depth,          # 0 = first doubt boundary of the trace
                        "token": tok_i,
                    }
                )
            print(f"[{ti}/{len(chosen)}] {t['question_id']}: "
                  f"{len(t['_bounds'])} boundaries, pool now {len(deltas)}", flush=True)
    finally:
        handle.remove()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"deltas": torch.stack(deltas), "meta": meta}, args.out)
    print(f"\nwrote {len(deltas)} raw deltas from {len(chosen)} traces -> {args.out}")


if __name__ == "__main__":
    main()
