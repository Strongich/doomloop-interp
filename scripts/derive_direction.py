#!/usr/bin/env python3
"""Derive one suppression direction from the doubt blocks of a SINGLE trace.

Finding 5 uses a direction averaged over 297 probes drawn from 297 different
traces. This asks a cheaper and more interesting question: is **one trace**
enough? If the doubt direction is a property of the model rather than of the
problem, then the ~14 doubt boundaries inside a single correct rollout should
contain it, and the resulting vector should steer *other* rollouts.

    Δ_i = AR(edit(e_i, CONTINUE_PARAGRAPH)) − AR(e_i)      for each doubt block i
    u   = normalize(mean_i Δ_i)

`block_explanations_3way.csv` cannot supply this: it holds exactly one
`doubt_wait` probe per trace. So the AV is run fresh over every doubt boundary
of the chosen trace.

    uv run python scripts/derive_direction.py --dataset gsm8k \\
        --out data/dir_gsm8k.pt

Evaluate the result with `steer_sweep.py --delta file:<path>`, holding out the
source trace — steering the trace you derived from is not a test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steer_demo import (  # noqa: E402
    CONTINUE_PARAGRAPH,
    DOUBT_PARAGRAPH,
    edit_explanation,
    load_ar,
    reconstruct,
)

from reasoning_attention.config import MODEL_ID, NLAConfig  # noqa: E402
from reasoning_attention.loops import DOUBT_MARKERS  # noqa: E402
from reasoning_attention.nla.arch import inner_transformer  # noqa: E402
from reasoning_attention.tokenview import _chat_header  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/correct_sample.jsonl"))
    p.add_argument("--dataset", default=None, help="restrict to one dataset")
    p.add_argument("--question-id", default=None, help="pin an exact trace")
    p.add_argument("--rollout-index", default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--base", default=MODEL_ID)
    p.add_argument("--av", type=Path, default=Path("checkpoints/av_rl_ep1"))
    p.add_argument("--ar", type=Path, default=Path("checkpoints/ar_rl_ep1"))
    p.add_argument("--min-blocks", type=int, default=6, help="min doubt blocks required")
    p.add_argument("--max-blocks", type=int, default=40)
    p.add_argument("--max-prefix", type=int, default=4000)
    p.add_argument("--av-max-new", type=int, default=300)
    return p.parse_args()


def doubt_boundaries(response: str, tokenizer: Any, max_prefix: int) -> list[tuple[int, int]]:
    """(token index, char cut) for every block boundary followed by a doubt marker.

    The probe is the last token before the `\\n\\n`, matching every other
    experiment here; `.\\n\\n` is a single Qwen3 token, so that token IS the break.
    """
    body = response.split("</think>")[0]
    enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = enc["offset_mapping"]
    out: list[tuple[int, int]] = []
    pos = 0
    for para in body.split("\n\n")[:-1]:
        pos += len(para) + 2
        nxt = response[pos : pos + 60].lstrip()
        if not any(nxt.lower().startswith(m.lower()) for m in DOUBT_MARKERS):
            continue
        # last token whose span ends at or before the break
        ti = max((i for i, (_, e) in enumerate(offsets) if 0 < e <= pos), default=-1)
        if ti < 0 or ti >= max_prefix:
            continue
        out.append((ti, offsets[ti][1]))
    return out


def main() -> None:
    args = parse_args()
    cfg = NLAConfig()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reasoning_attention.nla.model import NLA

    traces = [json.loads(line) for line in args.traces.open()]
    if args.dataset:
        traces = [t for t in traces if t["dataset"] == args.dataset]
    if args.question_id:
        traces = [t for t in traces if t["question_id"] == args.question_id]
    if args.rollout_index is not None:
        traces = [t for t in traces if str(t["rollout_index"]) == str(args.rollout_index)]
    if not traces:
        raise SystemExit("no trace matched the filters")

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    trunk = inner_transformer(model)

    # pick the first trace with enough doubt boundaries — a vector averaged over
    # 2 blocks would measure noise, not the trace.
    chosen, bounds = None, []
    for t in traces:
        b = doubt_boundaries(t["response"], tokenizer, args.max_prefix)
        if len(b) >= args.min_blocks:
            chosen, bounds = t, b[: args.max_blocks]
            break
    if chosen is None:
        raise SystemExit(f"no trace with >= {args.min_blocks} doubt boundaries")
    print(f"source trace: {chosen['question_id']} rollout {chosen['rollout_index']} "
          f"({chosen['dataset']}), {len(bounds)} doubt boundaries")

    nla = NLA.av_only(str(args.av))
    nla.av.eval()
    ar_tok, ar_backbone, affine = load_ar(args.ar)

    header = _chat_header(tokenizer, chosen["question"])
    n_header = len(tokenizer(header, add_special_tokens=False)["input_ids"])

    captured: dict[str, Any] = {"h": None, "site": -1}

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > captured["site"]:
            captured["h"] = hidden[0, captured["site"]].detach().float().cpu()
        return output

    handle = trunk.layers[cfg.extraction_layer].register_forward_hook(hook)

    deltas: dict[str, list[torch.Tensor]] = {"continue": [], "doubt": []}
    explanations: list[str] = []
    try:
        for k, (ti, cut) in enumerate(bounds, 1):
            enc = tokenizer(header + chosen["response"][:cut], return_tensors="pt").to("cuda")
            if enc["input_ids"].shape[1] > args.max_prefix:
                continue
            captured["site"] = n_header + ti
            with torch.no_grad():
                trunk(**enc)
            h = captured["h"]
            if h is None:
                continue
            e0 = nla.verbalize(
                h, max_new_tokens=args.av_max_new, return_explanation=True
            ).strip()
            explanations.append(e0)
            with torch.no_grad():
                a = reconstruct(ar_tok, ar_backbone, affine, e0)
                for name, para in (("continue", CONTINUE_PARAGRAPH), ("doubt", DOUBT_PARAGRAPH)):
                    try:
                        ed = edit_explanation(e0, para)
                    except SystemExit:
                        continue
                    deltas[name].append(
                        (reconstruct(ar_tok, ar_backbone, affine, ed) - a).float().cpu()
                    )
            print(f"  [{k}/{len(bounds)}] token {ti}", flush=True)
    finally:
        handle.remove()

    if not deltas["continue"]:
        raise SystemExit("no usable deltas — every explanation failed to edit")

    payload: dict[str, Any] = {
        "source": {
            "question_id": chosen["question_id"],
            "rollout_index": chosen["rollout_index"],
            "dataset": chosen["dataset"],
            "n_boundaries": len(deltas["continue"]),
        },
        "explanations": explanations,
    }
    for name, vs in deltas.items():
        stack = torch.stack(vs)
        mean = stack.mean(0)
        unit = mean / mean.norm()
        cos = torch.nn.functional.cosine_similarity(stack, mean[None], dim=1)
        payload[name] = {
            "unit": unit,
            "stats": {
                "n": float(stack.shape[0]),
                "mean_norm": float(stack.norm(dim=1).mean()),
                "norm_of_mean": float(mean.norm()),
                "cos_to_mean_mean": float(cos.mean()),
                "cos_to_mean_min": float(cos.min()),
            },
        }
        s = payload[name]["stats"]
        print(f"  {name}: n={s['n']:.0f} mean_norm={s['mean_norm']:.1f} "
              f"norm_of_mean={s['norm_of_mean']:.1f} cos_to_mean={s['cos_to_mean_mean']:.3f}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
