#!/usr/bin/env python3
"""Paired residual states: the SAME token sequence through Qwen3-1.7B and Qwen3-8B.

The two models share tokenizer and chat template (verified: identical vocab,
template and ids), so a position means the same token in both. Never pairs two
separately generated rollouts.

Sources (each row keeps `origin`):
  web      Ultra-FineWeb docs, random positions
  r17      Qwen3-1.7B base rollouts (full prompt + generation), replayed through both
  r8b      Qwen3-8B base rollouts, replayed through both
For rollouts, positions are paragraph-boundary tokens inside <think> (the steering
sites) plus random generated positions. A boundary is labelled `doubt` when the
next block's first sentence carries a DOUBT_MARKER (Finding 1 / doubt_stats rule).

Saved: h_s (1.7B block-20 output == hidden_states[21]) and h_b at each 8B layer in
--layers (block-l output == hidden_states[l+1]), bf16, plus row metadata.

    uv run python scripts/xfer_extract_pairs.py --out data/xfer8b/pairs.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from reasoning_attention.loops import DOUBT_MARKERS, marker_matches  # noqa: E402

FORMAT = "Give your final answer in \\boxed{}."


def split_of(doc_id: str) -> str:
    """Deterministic document-level split: 80 train / 10 val / 10 test."""
    h = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if h < 8 else ("val" if h == 8 else "test")


def first_sentence_doubt(text: str) -> bool:
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return any(marker_matches(first, m, case_sensitive=False) for m in DOUBT_MARKERS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--web", type=Path, default=Path("data/xfer8b/web3000.jsonl"))
    ap.add_argument("--rollouts", nargs="*", default=[
        "r17=data/xfer8b/r17_fit_base/rollouts.jsonl",
        "r17=data/reasoning_policy_v1/stage1_shard0/rollouts.jsonl",
        "r8b=data/xfer8b/r8b_fit_base/rollouts.jsonl",
    ])
    ap.add_argument("--layers", type=int, nargs="+", default=[14, 18, 20, 22, 24, 26, 28, 30, 32])
    ap.add_argument("--web-docs", type=int, default=3000)
    ap.add_argument("--web-positions", type=int, default=6)
    ap.add_argument("--web-max-tokens", type=int, default=768)
    ap.add_argument("--boundaries-per-trace", type=int, default=24)
    ap.add_argument("--random-per-trace", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--big-model", default="Qwen/Qwen3-8B")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    rng = random.Random(args.seed)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    close_id = tok.convert_tokens_to_ids("</think>")
    nl = "Ċ"  # byte-level newline
    is_boundary = [tok.convert_ids_to_tokens(i).count(nl) >= 2 for i in range(len(tok))]

    qtext = {}
    for c in list(Path("data/xfer8b").glob("*_math*.jsonl")) + list(Path("data/policy").glob("*.jsonl")):
        for line in c.open():
            r = json.loads(line)
            qtext[r["question_id"]] = r["question"]

    # ---- build (doc_id, origin, ids, positions, labels) work items -------------------
    items: list[dict] = []
    for i, line in enumerate(args.web.open()):
        if i >= args.web_docs:
            break
        r = json.loads(line)
        ids = tok(r["text"], add_special_tokens=False)["input_ids"][: args.web_max_tokens]
        if len(ids) < 64:
            continue
        pos = sorted(rng.sample(range(16, len(ids)), min(args.web_positions, len(ids) - 16)))
        items.append({"doc_id": r["doc_id"], "origin": "web", "ids": ids,
                      "pos": pos, "kind": ["random"] * len(pos), "doubt": [-1] * len(pos)})

    for spec in args.rollouts:
        origin, _, path = spec.partition("=")
        if not Path(path).exists():
            print(f"skip missing {path}")
            continue
        n0 = len(items)
        for line in open(path):
            r = json.loads(line)
            if r.get("policy", "base") != "base" or r.get("seed", 0) != 0:
                continue
            # Rebuild the exact prompt the runner used (FORMAT system turn, thinking on).
            q = qtext.get(r["question_id"])
            if q is None:
                continue
            msgs = [{"role": "system", "content": FORMAT}, {"role": "user", "content": q.strip()}]
            prompt = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=True),
                         add_special_tokens=False)["input_ids"]
            if len(prompt) != r["prompt_tokens"]:
                raise RuntimeError(f"prompt mismatch {r['question_id']}")
            gen = r["token_ids"]
            ids = (prompt + gen)[: args.max_tokens]
            P = len(prompt)
            close = gen.index(close_id) if close_id in gen else len(gen)
            bpos = [P + i for i, t in enumerate(gen[:close]) if is_boundary[t] and P + i + 1 < len(ids)]
            if len(bpos) > args.boundaries_per_trace:
                bpos = sorted(rng.sample(bpos, args.boundaries_per_trace))
            labels = [int(first_sentence_doubt(tok.decode(ids[p + 1 : p + 48]))) for p in bpos]
            others = [p for p in range(P + 1, len(ids) - 1) if not is_boundary[ids[p]]]
            rpos = sorted(rng.sample(others, min(args.random_per_trace, len(others))))
            doc = f"{origin}:{r['question_id']}"
            items.append({"doc_id": doc, "origin": origin, "ids": ids,
                          "pos": bpos + rpos, "kind": ["boundary"] * len(bpos) + ["random"] * len(rpos),
                          "doubt": labels + [-1] * len(rpos)})
        print(f"{origin} {path}: {len(items) - n0} traces")
    print(f"{len(items)} sequences, {sum(len(x['pos']) for x in items)} positions")

    # ---- forward both models ----------------------------------------------------------
    small = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    big = AutoModelForCausalLM.from_pretrained(args.big_model, dtype=torch.bfloat16,
                                               device_map="cuda").eval()
    grab: dict[str, torch.Tensor] = {}

    def hook(name: str):
        def f(_m, _i, out):
            grab[name] = out[0] if isinstance(out, tuple) else out
        return f

    hs = [small.model.layers[20].register_forward_hook(hook("s20"))]
    for l in args.layers:
        hs.append(big.model.layers[l].register_forward_hook(hook(f"b{l}")))
    # After the gate below, truncate both after the last needed block: later blocks
    # and lm_head never affect the grabbed outputs.

    # Mechanical gate: the hooked block output must equal hidden_states[l+1] of the
    # untruncated model on the same input.
    with torch.inference_mode():
        x = torch.tensor([items[-1]["ids"][:256]], device="cuda")
        ref_s = small(input_ids=x, output_hidden_states=True).hidden_states[21]
        ref_b = big(input_ids=x, output_hidden_states=True).hidden_states[args.layers[-1] + 1]
        assert torch.equal(grab["s20"], ref_s), "1.7B hook != hidden_states[21]"
        assert torch.equal(grab[f"b{args.layers[-1]}"], ref_b), "8B hook != hidden_states[l+1]"
    print("hook == hidden_states[l+1] on both models")
    small.model.layers = small.model.layers[:21]
    big.model.layers = big.model.layers[: max(args.layers) + 1]

    H_s, H_b, meta = [], {l: [] for l in args.layers}, []
    items.sort(key=lambda x: len(x["ids"]))
    with torch.inference_mode():
        for k, it in enumerate(items):
            x = torch.tensor([it["ids"]], device="cuda")
            p = torch.tensor(it["pos"], device="cuda")
            small.model(input_ids=x, use_cache=False)
            big.model(input_ids=x, use_cache=False)
            H_s.append(grab["s20"][0, p].cpu())
            for l in args.layers:
                H_b[l].append(grab[f"b{l}"][0, p].cpu())
            for j, pp in enumerate(it["pos"]):
                meta.append({"doc_id": it["doc_id"], "origin": it["origin"], "pos": pp,
                             "kind": it["kind"][j], "doubt": it["doubt"][j],
                             "split": split_of(it["doc_id"].split(":", 1)[1]
                                               if it["origin"] != "web" else it["doc_id"]),
                             "token": it["ids"][pp]})
            if k % 200 == 0:
                print(f"{k}/{len(items)} len={len(it['ids'])}", flush=True)
    out = {"h_s": torch.cat(H_s), "h_b": {l: torch.cat(v) for l, v in H_b.items()},
           "meta": meta, "layers": args.layers,
           "convention": "block outputs: 1.7B l=20 (hidden_states[21]); 8B l (hidden_states[l+1])"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}: {out['h_s'].shape[0]} rows")


if __name__ == "__main__":
    main()
