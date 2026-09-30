#!/usr/bin/env python3
"""Single-site injection screen through vLLM (targets too large for HF, e.g. 235B-A22B).

Same sites, same edit and same outcome as `xfer_local_injection.py`: at a paragraph-
boundary token p inside <think>, add alpha * ||h_p|| * u to the block-L output at p only
and read P(next token is a doubt opener). The edit goes through the steering worker's
`steer_positions` override (vllm_steering.py), so it is literally the rollout policy's
edit applied at one prompt position. Each site is a prefill-only request (prompt =
ids[:p+1], max_tokens=1) whose first-token logprobs give the next-token distribution.

Differences from the HF screen, both small and one-sided:
  * P(doubt) sums opener probabilities over the returned top-K (K=--topk, default 200);
    an opener outside the top-K contributes < p_K each and is dropped.
  * KL(base || steered) is computed over the base top-K, with a steered token missing
    from the steered top-K assigned the steered K-th logprob. An approximation: on the
    1.7B it lands within ~4% of the exact HF KL (validated, EXPERIMENT-transfer-235b-log).

Sites are selected exactly as in the HF screen (same rng, same traces order), so rows
pair with `xfer_local_injection.py` output on the same traces by (qid, pos).

    uv run python scripts/xfer_local_injection_vllm.py --model Qwen/Qwen3-235B-A22B-FP8 \\
        --layer 67 --pp 3 --tp 2 --spec NU:data/xfer235b/dirs_clean/L67_NU.pt ... \\
        --alphas -0.5 0.25 0.5 1 --out data/xfer235b/local/235b_L67.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from xfer_local_injection import FORMAT, first_sentence_doubt, split_of  # noqa: E402

from reasoning_attention.loops import DOUBT_MARKERS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--spec", nargs="+", required=True, help="NAME:PATH (unit vectors)")
    ap.add_argument("--alphas", type=float, nargs="+", default=[-0.5, 0.25, 0.5, 1.0])
    ap.add_argument("--traces", nargs="+", required=True)
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--per-trace", type=int, default=3)
    ap.add_argument("--max-prefix", type=int, default=6144)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--pp", type=int, default=1)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--limit", type=int, default=0, help="first N sites only (smoke)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from reasoning_attention.serving.vllm_steering import build_steering_llm, steering_stats

    tok = AutoTokenizer.from_pretrained(args.model)
    nl = "Ċ"
    is_b = [tok.convert_ids_to_tokens(i).count(nl) >= 2 for i in range(len(tok))]
    close_id = tok.convert_tokens_to_ids("</think>")
    openers = sorted({tok(m.split()[0], add_special_tokens=False)["input_ids"][0]
                      for m in DOUBT_MARKERS if not m.startswith("Let")})
    print("doubt opener ids:", [tok.decode([i]) for i in openers])

    qtext = {}
    for c in list(Path("data/xfer8b").glob("*_math*.jsonl")) + list(Path("data/policy").glob("*.jsonl")):
        for line in c.open():
            r = json.loads(line)
            qtext[r["question_id"]] = r["question"]

    # ---- site selection: verbatim logic of xfer_local_injection.py ------------------
    rng = random.Random(0)
    sites = []
    for path in args.traces:
        origin = "r17" if "r17_" in path else ("r8b" if "r8b" in path else "rtg")
        for line in open(path):
            r = json.loads(line)
            if r["policy"] != "base" or r["seed"] != 0 or split_of(r["question_id"]) not in args.splits:
                continue
            msgs = [{"role": "system", "content": FORMAT},
                    {"role": "user", "content": qtext[r["question_id"]].strip()}]
            prompt = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=True), add_special_tokens=False)["input_ids"]
            assert len(prompt) == r["prompt_tokens"]
            gen = r["token_ids"]
            close = gen.index(close_id) if close_id in gen else len(gen)
            P = len(prompt)
            cands = {0: [], 1: []}
            for i, t in enumerate(gen[: close - 1]):
                if is_b[t] and P + i < args.max_prefix and i >= 16:
                    lab = int(first_sentence_doubt(tok.decode(gen[i + 1 : i + 48])))
                    cands[lab].append(P + i)
            for lab in (0, 1):
                for p in rng.sample(cands[lab], min(args.per_trace, len(cands[lab]))):
                    sites.append((origin, r["question_id"], (prompt + gen)[: p + 1], lab))
    if args.limit:
        sites = sites[: args.limit]
    print(f"{len(sites)} sites: doubt {sum(s[3] for s in sites)}")

    specs = []
    for s in args.spec:
        name, path = s.split(":", 1)
        u = torch.load(path, weights_only=False)["unit"].float()
        assert abs(float(u.norm()) - 1) < 1e-3
        specs.append((name, u))

    boundaries = [i for i, b in enumerate(is_b) if b]
    max_len = max(len(s[2]) for s in sites) + 8
    llm = build_steering_llm(args.model, args.layer, boundaries, close_id, max_len, 64,
                             args.gpu_memory_utilization, 8192, args.pp, args.tp,
                             max_logprobs=args.topk)

    def run(unit: torch.Tensor | None, alpha: float) -> list[dict[int, float]]:
        llm.collective_rpc("configure_reasoning_steering",
                           args=(None if unit is None else unit.tolist(), alpha))
        params = [SamplingParams(max_tokens=1, temperature=1.0, logprobs=args.topk,
                                 extra_args={"steer_positions": [len(ids) - 1],
                                             "steering_audit_id": f"s{k}"})
                  for k, (_, _, ids, _) in enumerate(sites)]
        outs = llm.generate([{"prompt_token_ids": s[2]} for s in sites], params, use_tqdm=False)
        st = steering_stats(llm)
        if unit is not None and alpha != 0:
            bad = [k for k, s in enumerate(sites)
                   if st["requests"].get(f"s{k}", {}).get("injections") != [len(s[2]) - 1]]
            if bad:
                raise RuntimeError(f"injection audit failed at {len(bad)} sites, e.g. {bad[:3]}")
        return [{tid: lp.logprob for tid, lp in o.outputs[0].logprobs[0].items()} for o in outs]

    def p_doubt(d: dict[int, float]) -> float:
        return sum(math.exp(d[t]) for t in openers if t in d)

    def kl(base: dict[int, float], st: dict[int, float]) -> float:
        floor = min(st.values())
        return sum(math.exp(lb) * (lb - st.get(t, floor)) for t, lb in base.items())

    base = run(None, 0.0)
    rows = [{"origin": o, "qid": q, "label": lab, "pos": len(ids) - 1,
             "p_base": p_doubt(base[k]), "arms": {}}
            for k, (o, q, ids, lab) in enumerate(sites)]
    for name, u in specs:
        for a in args.alphas:
            got = run(u, a)
            for k, d in enumerate(got):
                rows[k]["arms"][f"L{args.layer}:{name}:{a}"] = {
                    "p": p_doubt(d), "kl": kl(base[k], d),
                    "top": tok.decode([max(d, key=d.get)])}
            dd = [r["arms"][f"L{args.layer}:{name}:{a}"]["p"] for r in rows if r["label"] == 1]
            print(f"{name} a={a}: mean P(doubt) at doubt sites {sum(dd) / max(1, len(dd)):.3f}",
                  flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
