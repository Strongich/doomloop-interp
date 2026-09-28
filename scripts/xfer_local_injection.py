#!/usr/bin/env python3
"""Single-site injection screen: does a direction move the NEXT block's opener the
same way in 1.7B and 8B, on the same text?

At a paragraph-boundary token p inside <think> (a steering site), add
alpha * ||h_p|| * u to the block-L output at p only (exactly the vLLM policy's edit
at one site) and read the next-token distribution. Outcome:

  P_doubt = total probability of the first token of a doubt opener (Wait, Hmm, But,
            Actually, Alternatively, However, Maybe, Hold ...), the tokens that start
            a DOUBT_MARKER block after "\\n\\n".

Sites are split by what the model actually wrote next (doubt vs plain), from traces
of held-out (val/test) questions only. The same token sequence goes through both
models, so every 1.7B/8B comparison is paired by site.

    uv run python scripts/xfer_local_injection.py --model Qwen/Qwen3-8B \\
        --spec 26:NU:data/xfer8b/dirs/L26_NU.pt ... --alphas -1 0.5 1 --out ...
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
from reasoning_attention.loops import DOUBT_MARKERS, marker_matches  # noqa: E402

FORMAT = "Give your final answer in \\boxed{}."


def split_of(qid: str) -> str:
    h = int(hashlib.sha256(qid.encode()).hexdigest()[:8], 16) % 10
    return "train" if h < 8 else ("val" if h == 8 else "test")


def first_sentence_doubt(text: str) -> bool:
    first = re.split(r"(?<=[.!?])\s", text.strip(), maxsplit=1)[0]
    return any(marker_matches(first, m, case_sensitive=False) for m in DOUBT_MARKERS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--spec", nargs="+", required=True, help="LAYER:NAME:PATH")
    ap.add_argument("--alphas", type=float, nargs="+", default=[-1.0, -0.5, 0.25, 0.5, 1.0])
    ap.add_argument("--traces", nargs="+", default=[
        "data/xfer8b/r8b_fit_base/rollouts.jsonl", "data/xfer8b/r17_fit_base/rollouts.jsonl"])
    ap.add_argument("--splits", nargs="+", default=["val", "test"])
    ap.add_argument("--per-trace", type=int, default=3, help="doubt AND plain sites per trace")
    ap.add_argument("--max-prefix", type=int, default=6144)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

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

    rng = random.Random(0)
    sites = []  # (origin, qid, ids[:p+1], label)
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
    print(f"{len(sites)} sites: doubt {sum(s[3] for s in sites)}")

    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                 device_map="cuda").eval()
    specs = []
    for s in args.spec:
        layer, name, path = s.split(":", 2)
        u = torch.load(path, weights_only=False)["unit"].float().cuda()
        assert u.shape[0] == model.config.hidden_size and abs(float(u.norm()) - 1) < 1e-3
        specs.append((int(layer), name, u))

    state = {"layer": None, "u": None, "alpha": 0.0}

    def make_hook(layer: int):
        def hook(_m, _i, out):
            if state["layer"] != layer or state["alpha"] == 0:
                return out
            h = out[0] if isinstance(out, tuple) else out
            last = h[:, -1:, :]
            push = state["alpha"] * last.float().norm(dim=-1, keepdim=True) * state["u"]
            h = torch.cat([h[:, :-1], (last + push.to(h.dtype))], dim=1)
            return (h, *out[1:]) if isinstance(out, tuple) else h
        return hook

    for layer in sorted({s[0] for s in specs}):
        model.model.layers[layer].register_forward_hook(make_hook(layer))

    op = torch.tensor(openers, device="cuda")
    rows = []
    with torch.inference_mode():
        for k, (origin, qid, ids, lab) in enumerate(sites):
            x = torch.tensor([ids], device="cuda")
            state["alpha"] = 0.0
            out = model(input_ids=x[:, :-1], use_cache=True)
            cache = out.past_key_values
            P0 = x.shape[1] - 1

            def step() -> torch.Tensor:
                cache.crop(P0)
                return model(input_ids=x[:, -1:], past_key_values=cache, use_cache=True).logits[0, -1].float()

            base = step().log_softmax(-1)
            row = {"origin": origin, "qid": qid, "label": lab, "pos": P0,
                   "p_base": float(base[op].exp().sum()), "arms": {}}
            for layer, name, u in specs:
                for a in args.alphas:
                    state.update(layer=layer, u=u, alpha=a)
                    lp = step().log_softmax(-1)
                    state["alpha"] = 0.0
                    row["arms"][f"L{layer}:{name}:{a}"] = {
                        "p": float(lp[op].exp().sum()),
                        "kl": float((base.exp() * (base - lp)).sum()),
                        "top": tok.decode([int(lp.argmax())]),
                    }
            rows.append(row)
            if k % 50 == 0:
                print(f"{k}/{len(sites)}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
