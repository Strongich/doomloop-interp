#!/usr/bin/env python3
r"""EXPERIMENT-selective-doubt.md §S4/§S5: raw layer-20 block output at every episode's
*before* boundary (the doubt boundary b_k), from one HF forward per rollout over the stored
prompt + generated ids (causal, so one pass serves every boundary of the trace).

Output: <out>.pt = {"h": [N, 2048] bf16, "index": [(key, k)], "layer": 20}

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/sel_states.py \
        --episodes data/selective/episodes_source.jsonl \
        --journals 'data/distill/gen/shard*/rollouts.jsonl' --questions data/distill/train.jsonl \
        --shard 0/2 --out data/selective/states/source_s0.pt
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sel_common import Paragraphs  # noqa: E402
from sel_probe import key_of  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=Path, required=True)
    ap.add_argument("--journals", required=True)
    ap.add_argument("--questions", type=Path, required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import glob

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reasoning_attention.config import MODEL_ID, NLAConfig
    from reasoning_attention.nla.arch import inner_transformer

    cuts: dict[str, list[tuple[int, int]]] = collections.defaultdict(list)
    for x in args.episodes.open():
        e = json.loads(x)
        cuts[e["key"]].append((e["k"], e["cut"]))
    keys = sorted(cuts)
    i, n = map(int, args.shard.split("/"))
    keys = set(keys[i::n])
    recs = {}
    for f in sorted(glob.glob(str(ROOT / args.journals))):
        for line in open(f):
            if '"token_ids"' not in line:
                continue
            r = json.loads(line)
            if key_of(r) in keys:
                recs[key_of(r)] = r
    assert set(recs) == keys, f"missing {len(keys - set(recs))}"
    qs = {json.loads(x)["question_id"]: json.loads(x) for x in args.questions.open()}
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    P = Paragraphs(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="sdpa")
    model.eval()
    trunk = inner_transformer(model)
    layer = NLAConfig().extraction_layer
    cap: dict = {}

    def hook(_m, _i, out):
        cap["h"] = (out[0] if isinstance(out, tuple) else out)[0]

    handle = trunk.layers[layer].register_forward_hook(hook)
    hs, index = [], []
    with torch.no_grad():
        for j, key in enumerate(sorted(recs)):
            r = recs[key]
            prompt = P.prompt_ids(qs[r["question_id"]]["question"])
            assert len(prompt) == r["prompt_tokens"]
            cs = sorted(cuts[key])
            last = max(c for _, c in cs)
            ids = torch.tensor([prompt + r["token_ids"][: last + 1]], device="cuda")
            trunk(input_ids=ids)
            pos = torch.tensor([len(prompt) + c for _, c in cs], device="cuda")
            hs.append(cap["h"][pos].to(torch.bfloat16).cpu())
            index += [(key, k) for k, _ in cs]
            if j % 50 == 0:
                print(f"{j}/{len(recs)}", flush=True)
    handle.remove()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"h": torch.cat(hs), "index": index, "layer": layer}, args.out)
    print(f"wrote {len(index)} states -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
