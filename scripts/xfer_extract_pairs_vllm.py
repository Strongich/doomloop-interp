#!/usr/bin/env python3
"""Paired residual states for a target too large for HF (Qwen3-235B-A22B-FP8).

Same data design as `xfer_extract_pairs.py` (web docs + 1.7B traces + target traces,
boundary and random positions, doc-level 80/10/10 split), with two changes:

  * The web and 1.7B-trace rows are REUSED from an earlier pairs file (default: the
    30B one). Their 1.7B side (h_s, HF, block-20 output) is taken as-is; their token
    sequences are rebuilt with the extractor's own rules and every row's stored token id
    is checked against the rebuilt sequence. So the 1.7B side of the web/r17 rows is
    bit-identical to the 30B fit, and the target sees exactly the same positions.
  * The target side is captured through vLLM (`CaptureWorkerExtension`, PP x TP),
    not HF. Capture-vs-HF parity is validated separately (`xfer_vllm_capture --check`);
    no fit mixes captured and HF rows for the SAME model.

Two phases, because the target engine takes every GPU:
  --phase small   HF 1.7B on the NEW target traces -> <out>.small.pt (items + h_s)
  --phase big     vLLM capture of the target at --layers on all items -> <out>

    uv run python scripts/xfer_extract_pairs_vllm.py --phase small --out data/xfer235b/pairs.pt \\
        --target rtg=data/xfer235b/r235_fit_base/rollouts.jsonl
    uv run python scripts/xfer_extract_pairs_vllm.py --phase big --out data/xfer235b/pairs.pt \\
        --model Qwen/Qwen3-235B-A22B-FP8 --layers 56 62 67 72 78 84 --pp 3 --tp 2
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import OrderedDict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from xfer_extract_pairs import FORMAT, first_sentence_doubt, split_of  # noqa: E402

R17_TRACES = ["data/xfer8b/r17_fit_base/rollouts.jsonl",
              "data/reasoning_policy_v1/stage1_shard0/rollouts.jsonl"]


def load_qtext() -> dict[str, str]:
    qtext = {}
    for c in list(Path("data/xfer8b").glob("*_math*.jsonl")) + list(Path("data/policy").glob("*.jsonl")):
        for line in c.open():
            r = json.loads(line)
            qtext[r["question_id"]] = r["question"]
    return qtext


def trace_ids(tok, r: dict, qtext: dict[str, str], max_tokens: int) -> tuple[list[int], int]:
    """prompt + generation, exactly as the extractor rebuilds it; returns (ids, prompt_len)."""
    msgs = [{"role": "system", "content": FORMAT},
            {"role": "user", "content": qtext[r["question_id"]].strip()}]
    prompt = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True), add_special_tokens=False)["input_ids"]
    if len(prompt) != r["prompt_tokens"]:
        raise RuntimeError(f"prompt mismatch {r['question_id']}")
    return (prompt + r["token_ids"])[:max_tokens], len(prompt)


def reused_items(tok, base_pairs: Path, web: Path, web_max_tokens: int,
                 max_tokens: int) -> tuple[list[dict], torch.Tensor]:
    """web + r17 items from an earlier pairs file, with their h_s rows in item order."""
    P = torch.load(base_pairs, weights_only=False)
    meta = P["meta"]
    groups: OrderedDict[str, list[int]] = OrderedDict()
    for i, m in enumerate(meta):
        if m["origin"] in ("web", "r17"):
            groups.setdefault(m["doc_id"], []).append(i)
    web_text = {}
    for line in web.open():
        r = json.loads(line)
        web_text[r["doc_id"]] = r["text"]
    qtext = load_qtext()
    r17 = {}
    for path in R17_TRACES:
        for line in open(path):
            r = json.loads(line)
            if r.get("policy", "base") == "base" and r.get("seed", 0) == 0:
                r17[f"r17:{r['question_id']}"] = r
    items, rows = [], []
    for doc, idx in groups.items():
        m0 = meta[idx[0]]
        if m0["origin"] == "web":
            ids = tok(web_text[doc], add_special_tokens=False)["input_ids"][:web_max_tokens]
        else:
            ids, _ = trace_ids(tok, r17[doc], qtext, max_tokens)
        pos = [meta[i]["pos"] for i in idx]
        bad = [p for p, i in zip(pos, idx) if ids[p] != meta[i]["token"]]
        if bad:
            raise RuntimeError(f"{doc}: rebuilt ids disagree with stored tokens at {bad[:5]}")
        items.append({"doc_id": doc, "origin": m0["origin"], "ids": ids, "pos": pos,
                      "kind": [meta[i]["kind"] for i in idx],
                      "doubt": [meta[i]["doubt"] for i in idx],
                      "split": [meta[i]["split"] for i in idx]})
        rows.extend(idx)
    return items, P["h_s"][torch.tensor(rows)]


def target_items(tok, specs: list[str], max_tokens: int, per_trace: int, random_per_trace: int,
                 seed: int) -> list[dict]:
    """Target-model traces: the extractor's boundary/random sampling and doubt labels."""
    rng = random.Random(seed)
    close_id = tok.convert_tokens_to_ids("</think>")
    is_boundary = [tok.convert_ids_to_tokens(i).count("Ċ") >= 2 for i in range(len(tok))]
    qtext = load_qtext()
    items = []
    for spec in specs:
        origin, _, path = spec.partition("=")
        for line in open(path):
            r = json.loads(line)
            if r.get("policy", "base") != "base" or r.get("seed", 0) != 0:
                continue
            ids, P = trace_ids(tok, r, qtext, max_tokens)
            gen = r["token_ids"]
            close = gen.index(close_id) if close_id in gen else len(gen)
            bpos = [P + i for i, t in enumerate(gen[:close]) if is_boundary[t] and P + i + 1 < len(ids)]
            if len(bpos) > per_trace:
                bpos = sorted(rng.sample(bpos, per_trace))
            labels = [int(first_sentence_doubt(tok.decode(ids[p + 1 : p + 48]))) for p in bpos]
            others = [p for p in range(P + 1, len(ids) - 1) if not is_boundary[ids[p]]]
            rpos = sorted(rng.sample(others, min(random_per_trace, len(others))))
            pos = bpos + rpos
            if not pos:  # e.g. an empty or corrupted trace: nothing to pair
                continue
            items.append({"doc_id": f"{origin}:{r['question_id']}", "origin": origin, "ids": ids,
                          "pos": pos, "kind": ["boundary"] * len(bpos) + ["random"] * len(rpos),
                          "doubt": labels + [-1] * len(rpos),
                          "split": [split_of(r["question_id"])] * len(pos)})
        print(f"{origin} {path}: {sum(1 for x in items if x['origin'] == origin)} traces")
    return items


def phase_small(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    reused, hs_reused = reused_items(tok, args.base_pairs, args.web, args.web_max_tokens,
                                     args.max_tokens)
    print(f"reused {len(reused)} sequences / {hs_reused.shape[0]} rows from {args.base_pairs}")
    new = target_items(tok, args.target, args.max_tokens, args.boundaries_per_trace,
                       args.random_per_trace, args.seed)
    small = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    grab: dict[str, torch.Tensor] = {}
    small.model.layers[20].register_forward_hook(
        lambda _m, _i, out: grab.__setitem__("h", out[0] if isinstance(out, tuple) else out))
    with torch.inference_mode():  # gate: hook == hidden_states[21] on the untruncated model
        x = torch.tensor([new[0]["ids"][:256]], device="cuda")
        ref = small(input_ids=x, output_hidden_states=True).hidden_states[21]
        assert torch.equal(grab["h"], ref), "1.7B hook != hidden_states[21]"
    small.model.layers = small.model.layers[:21]
    hs_new = []
    with torch.inference_mode():
        for it in new:
            small.model(input_ids=torch.tensor([it["ids"]], device="cuda"), use_cache=False)
            hs_new.append(grab["h"][0, torch.tensor(it["pos"], device="cuda", dtype=torch.long)].cpu())
    out = {"items": reused + new, "h_s": torch.cat([hs_reused, torch.cat(hs_new)]),
           "base_pairs": str(args.base_pairs), "n_reused_rows": int(hs_reused.shape[0])}
    path = args.out.with_suffix(".small.pt")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    print(f"wrote {path}: {len(out['items'])} sequences, {out['h_s'].shape[0]} rows")


def phase_big(args: argparse.Namespace) -> None:
    from xfer_vllm_capture import build_capture_llm, capture

    S = torch.load(args.out.with_suffix(".small.pt"), weights_only=False)
    items = S["items"]
    max_len = max(len(it["ids"]) for it in items) + 8
    llm = build_capture_llm(args.model, args.layers, max_len, pp=args.pp, tp=args.tp,
                            gpu_memory_utilization=args.gpu_memory_utilization,
                            max_num_seqs=args.chunk)
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    got: dict[int, dict[int, torch.Tensor]] = {}
    t0 = time.monotonic()
    for c in range(0, len(order), args.chunk):
        idx = order[c : c + args.chunk]
        res = capture(llm, [items[i]["ids"] for i in idx], [items[i]["pos"] for i in idx],
                      args.layers, str(args.out.parent / "_captmp"))
        off = 0
        for i in idx:
            n = len(items[i]["pos"])
            got[i] = {layer: res[layer][off : off + n].to(torch.bfloat16) for layer in args.layers}
            off += n
        print(f"{c + len(idx)}/{len(items)} max_len={len(items[idx[-1]]['ids'])} "
              f"{time.monotonic() - t0:.0f}s", flush=True)
    meta, H_b = [], {layer: [] for layer in args.layers}
    for i, it in enumerate(items):  # original item order == h_s row order
        for layer in args.layers:
            H_b[layer].append(got[i][layer])
        for j, p in enumerate(it["pos"]):
            meta.append({"doc_id": it["doc_id"], "origin": it["origin"], "pos": p,
                         "kind": it["kind"][j], "doubt": it["doubt"][j], "split": it["split"][j],
                         "token": it["ids"][p]})
    out = {"h_s": S["h_s"], "h_b": {layer: torch.cat(v) for layer, v in H_b.items()},
           "meta": meta, "layers": args.layers,
           "convention": f"1.7B l=20 block output (HF, hidden_states[21]); {args.model} block-l "
                         "output (vLLM capture, hidden+residual)",
           "model": args.model, "base_pairs": S["base_pairs"]}
    assert out["h_s"].shape[0] == len(meta) == out["h_b"][args.layers[0]].shape[0]
    torch.save(out, args.out)
    print(f"wrote {args.out}: {len(meta)} rows")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["small", "big"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--base-pairs", type=Path, default=Path("data/xfer30b/pairs.pt"))
    ap.add_argument("--web", type=Path, default=Path("data/xfer8b/web3000.jsonl"))
    ap.add_argument("--target", nargs="*", default=[])
    ap.add_argument("--model", default="Qwen/Qwen3-235B-A22B-FP8")
    ap.add_argument("--layers", type=int, nargs="+", default=[56, 62, 67, 72, 78, 84])
    ap.add_argument("--pp", type=int, default=3)
    ap.add_argument("--tp", type=int, default=2)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--web-max-tokens", type=int, default=768)
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--boundaries-per-trace", type=int, default=24)
    ap.add_argument("--random-per-trace", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    {"small": phase_small, "big": phase_big}[args.phase](args)


if __name__ == "__main__":
    main()
