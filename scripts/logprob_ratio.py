#!/usr/bin/env python3
"""How far does the hooked policy pi' differ from the unhooked pi, per token?

This decides whether the suppression edit could be applied *during* GRPO rollouts.
Gradients never needed to flow through the injection -- a policy gradient updates
on sampled tokens, and the hook only changes which tokens get sampled. The real
obstacle is off-policyness: sampling from pi' while updating pi biases the update
unless the importance ratio pi'(t)/pi(t) is accounted for.

Our injection is very sparse (median ~1% of token positions), so pi' = pi
wherever the hook is a no-op -- EXCEPT through the cache: the hook rewrites layer
20's output, and layers 21-27 compute their K/V from it, so an edited position
changes every later token's attention. Whether that shifts logprobs enough to
matter is a measurement, not an argument.

Method: take steered traces already on disk, teacher-force each one twice -- once
clean, once with the hook active at exactly the positions it fired at during
generation -- and compare per-token logprobs.

  ratio ~ 1 almost everywhere -> importance correction is trivial; in-loop
                                 injection is viable.
  ratio heavy-tailed          -> off-policy bias is real; generate data with the
                                 hook and train without it (the DPO route).

Teacher-forced prefill is numerically not identical to incremental decoding, but
both passes here use the same regime, which is the right comparison.

    uv run python scripts/logprob_ratio.py --limit 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_attention.config import MODEL_ID, NLAConfig  # noqa: E402
from reasoning_attention.nla.arch import inner_transformer  # noqa: E402
from reasoning_attention.tokenview import _chat_header  # noqa: E402

NEWLINE_CHAR = "Ċ"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--texts", type=Path, default=Path("data/tier3/step1_A_texts.jsonl"))
    p.add_argument("--questions", type=Path, default=Path("data/hard832.jsonl"))
    p.add_argument("--direction", type=Path, default=Path("data/pool/dir_A_1trace.pt"))
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--out", type=Path, default=Path("data/logprob_ratio.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NLAConfig()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    qs = {json.loads(line)["question_id"]: json.loads(line)["question"]
          for line in args.questions.open()}
    recs = [json.loads(line) for line in args.texts.open()]
    recs = [r for r in recs if r["question_id"] in qs][: args.limit]
    print(f"{len(recs)} steered traces")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    trunk = inner_transformer(model)
    unit = torch.load(args.direction, map_location="cpu", weights_only=False)["unit"]
    unit = unit.to(model.device, torch.float32)

    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    is_break = torch.zeros(len(tok), dtype=torch.bool)
    for i, piece in enumerate(pieces):
        if piece and piece.count(NEWLINE_CHAR) >= 2:
            is_break[i] = True
    think_close = int(tok.convert_tokens_to_ids("</think>"))

    # Per-POSITION mask, unlike generation's last-position-only hook: in teacher
    # forcing the whole sequence goes through at once, so the hook must edit every
    # position whose own token fired the injection during generation.
    state: dict[str, Any] = {"mask": None}

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        mask = state["mask"]
        if mask is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        h = hidden[0].float()
        push = args.alpha * h.norm(dim=-1, keepdim=True) * unit[None]
        hidden[0] = torch.where(mask[:, None], (h + push).to(hidden.dtype), hidden[0])
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    trunk.layers[cfg.extraction_layer].register_forward_hook(hook)

    all_log_ratio: list[torch.Tensor] = []
    all_edited: list[torch.Tensor] = []
    n_skip = 0
    for i, r in enumerate(recs, 1):
        header = _chat_header(tok, qs[r["question_id"]])
        h_ids = tok(header, return_tensors="pt")["input_ids"][0]
        c_ids = tok(r["suppress"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
        if len(h_ids) + len(c_ids) > args.max_tokens:
            n_skip += 1
            continue
        ids = torch.cat([h_ids, c_ids]).to(model.device)

        # injection fired at a completion position iff its own token is a break
        # token and it is still inside <think>
        edited = torch.zeros(len(ids), dtype=torch.bool)
        close_at = (c_ids == think_close).nonzero()
        last = int(close_at[0]) if len(close_at) else len(c_ids)
        for j in range(last):
            if is_break[int(c_ids[j])]:
                edited[len(h_ids) + j] = True

        with torch.no_grad():
            state["mask"] = None
            lg = model(input_ids=ids[None]).logits[0].float()
            state["mask"] = edited.to(model.device)
            lg2 = model(input_ids=ids[None]).logits[0].float()
        state["mask"] = None

        tgt = ids[1:]
        lp = torch.log_softmax(lg[:-1], -1).gather(1, tgt[:, None])[:, 0]
        lp2 = torch.log_softmax(lg2[:-1], -1).gather(1, tgt[:, None])[:, 0]
        keep = slice(len(h_ids) - 1, None)          # completion tokens only
        all_log_ratio.append((lp2 - lp)[keep].cpu())
        all_edited.append(edited[1:][keep].cpu())
        if i % 25 == 0:
            print(f"  [{i}/{len(recs)}]", flush=True)

    lr = torch.cat(all_log_ratio)
    ed = torch.cat(all_edited)
    print(f"\nskipped {n_skip} traces over {args.max_tokens} tokens")
    print(f"{len(lr)} completion tokens, {int(ed.sum())} edited "
          f"({100*ed.float().mean():.2f}%)\n")

    def describe(x: torch.Tensor, lab: str) -> None:
        if not len(x):
            return
        r = x.exp()
        q = torch.tensor([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        qs_ = r.quantile(q)
        print(f"{lab:<22}n={len(x):>8}  "
              + "  ".join(f"p{round(100*a)}={b:.3f}" for a, b in zip(q.tolist(), qs_.tolist())))
        for t in (0.01, 0.05, 0.2):
            frac = (x.abs() < t).float().mean()
            print(f"{'':<22}  |log ratio| < {t:<5} : {100*frac:>5.1f}% of tokens")

    describe(lr, "all tokens")
    describe(lr[~ed], "unedited positions")
    describe(lr[ed], "edited positions")
    torch.save({"log_ratio": lr, "edited": ed}, args.out)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
