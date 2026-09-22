#!/usr/bin/env python3
"""One worked steering intervention: edit an explanation, inject the difference.

Demonstrates the causal test for Finding 2 on a single probe. The recipe is the
NLA paper's: verbalize a state, edit the *text*, push both versions back through
the reconstructor, and inject the difference at the layer the NLA was trained on.

    Δ = AR(e_edited) − AR(e_original)
    h → h + α·‖h‖·(Δ/‖Δ‖)      at one token, layer 20

`Δ` is not an arbitrary direction: the AR maps explanation text onto layer-20
activations, so both reconstructions land on the reasoning manifold and their
difference is the translation corresponding to a known change in meaning.

Runs the injection at several α alongside two controls — a random direction of
the same norm, and no injection — because a doubt rate that rises under *any*
perturbation shows only that block boundaries are fragile.

    uv run python scripts/steer_demo.py --question-id gsm8k:3931 --token 383
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_attention.config import MODEL_ID, NLAConfig  # noqa: E402
from reasoning_attention.loops import DOUBT_MARKERS  # noqa: E402
from reasoning_attention.nla.arch import (  # noqa: E402
    inner_transformer,
    strip_final_norm,
    strip_lm_head,
)
from reasoning_attention.nla.prompts import build_ar_prompt  # noqa: E402
from reasoning_attention.tokenview import _chat_header  # noqa: E402

# The AV's last paragraph is where it predicts what comes next. Only that
# paragraph is rewritten — register, topic and the quoted sentence structure are
# left alone, so Δ isolates the impending self-doubt and not the subject matter.
# The replacement is phrased the way the AV itself phrases an impending doubt
# (taken from this trace's real doubt_wait explanation at token 452).
DOUBT_PARAGRAPH = (
    "This indicates the sentence is concluding the thought about verification, "
    'likely expecting a concluding clause like "Wait, let me double-check that." '
    "to complete the reasoning."
)


# The decisive control: a rewrite that is just as much of an edit, in the same
# slot, but predicts an ordinary continuation instead of a doubt. If Δ from THIS
# also makes the model doubt, the effect is "any AR-shaped push at a block
# boundary", not "the doubt content".
CONTINUE_PARAGRAPH = (
    "This ends the sentence, implying the next token likely starts a new sentence "
    'or clause, such as "Therefore, the total is calculated as follows." to '
    "continue the derivation."
)


def edit_explanation(explanation: str, paragraph: str) -> str:
    """Replace the AV's prediction paragraph with `paragraph`."""
    paragraphs = [p for p in explanation.split("\n\n") if p.strip()]
    if len(paragraphs) < 2:
        raise SystemExit("explanation has no separate prediction paragraph")
    return "\n\n".join([*paragraphs[:-1], paragraph])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/correct_sample.jsonl"))
    p.add_argument("--explanations", type=Path, default=Path("data/block_explanations_3way.csv"))
    p.add_argument("--question-id", default="gsm8k:3931")
    p.add_argument("--kind", default="plain", help="which probe of that trace")
    p.add_argument("--base", default=MODEL_ID)
    p.add_argument("--ar", type=Path, default=Path("checkpoints/ar_rl_ep1"))
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    p.add_argument("--max-new-tokens", type=int, default=90)
    p.add_argument(
        "--seeds", type=int, default=8, help="samples per condition (T=0.6, so it varies)"
    )
    return p.parse_args()


def load_ar(path: Path) -> tuple[Any, Any, torch.nn.Linear]:
    """The trained AR: truncated backbone (norm/lm_head stripped) + affine map.

    `model.norm.weight` is reported MISSING on load, which is expected: the final
    norm was stripped before the checkpoint was written, and it is stripped again
    here. `last_hidden_state` is then the raw layer-l residual.
    """
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(path)
    backbone = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map="cuda")
    strip_lm_head(backbone)
    strip_final_norm(inner_transformer(backbone))
    backbone.eval()
    d_model = int(backbone.config.hidden_size)
    affine = torch.nn.Linear(d_model, d_model, bias=False)
    affine.load_state_dict(load_file(path / "value_head.safetensors"))
    affine.to("cuda", torch.float32).eval()
    return tokenizer, backbone, affine


@torch.no_grad()
def reconstruct(tokenizer: Any, backbone: Any, affine: Any, explanation: str) -> torch.Tensor:
    """AR(e): layer-20 activation predicted from an explanation's final token."""
    # add_special_tokens=True to match the extractor that produced the targets.
    enc = tokenizer(build_ar_prompt(explanation), return_tensors="pt").to("cuda")
    h = inner_transformer(backbone)(**enc).last_hidden_state[0, -1]
    return affine(h.float())


def main() -> None:
    args = parse_args()
    cfg = NLAConfig()
    import csv

    row = next(
        r
        for r in csv.DictReader(args.explanations.open())
        if r["question_id"] == args.question_id and r["kind"] == args.kind
    )
    trace = next(
        json.loads(line)
        for line in args.traces.open()
        if json.loads(line)["question_id"] == args.question_id
    )
    token = int(row["token"])
    e_original = row["explanation"]
    e_edited = edit_explanation(e_original, DOUBT_PARAGRAPH)
    e_control = edit_explanation(e_original, CONTINUE_PARAGRAPH)

    print(f"=== {args.question_id} {args.kind} token {token}  (gold {trace['gold']})")
    print(f"\n--- e_original (AV of h)\n{e_original}")
    print(f"\n--- e_edited (one sentence changed)\n{e_edited}")

    # --- Δ from the reconstructor -------------------------------------------
    ar_tok, ar_backbone, affine = load_ar(args.ar)
    a = reconstruct(ar_tok, ar_backbone, affine, e_original)
    b = reconstruct(ar_tok, ar_backbone, affine, e_edited)
    c = reconstruct(ar_tok, ar_backbone, affine, e_control)
    delta = (b - a).cpu()
    delta_ctrl = (c - a).cpu()
    cos = torch.nn.functional.cosine_similarity
    print(
        f"\n--- ‖AR(orig)‖={a.norm():.1f}  ‖AR(doubt)‖={b.norm():.1f}  "
        f"‖AR(continue)‖={c.norm():.1f}"
    )
    print(
        f"    ‖Δ_doubt‖={delta.norm():.1f}  ‖Δ_continue‖={delta_ctrl.norm():.1f}  "
        f"cos(Δ_doubt, Δ_continue)={cos(delta[None], delta_ctrl[None]).item():+.4f}"
    )
    del ar_backbone, affine
    torch.cuda.empty_cache()

    # --- inject into the target model ---------------------------------------
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    header = _chat_header(tokenizer, trace["question"])
    n_header = len(tokenizer(header, add_special_tokens=False)["input_ids"])
    resp_enc = tokenizer(trace["response"], add_special_tokens=False, return_offsets_mapping=True)
    cut = resp_enc["offset_mapping"][token][1]
    prefix = header + trace["response"][:cut]
    enc = tokenizer(prefix, return_tensors="pt").to("cuda")
    site = n_header + token
    print(f"\nprefix = {enc['input_ids'].shape[1]} tokens; injecting at absolute position {site}")
    print(f"prefix ends: ...{trace['response'][max(0, cut - 150) : cut]!r}")
    print(f"\nWhat the model ACTUALLY wrote next: {trace['response'][cut : cut + 90]!r}\n")

    state: dict[str, Any] = {"vec": None}

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        vec = state["vec"]
        if vec is None:
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        # Only during prefill: with a KV cache the later steps carry one token,
        # and the site has already been consumed.
        if hidden.shape[1] > site:
            hidden = hidden.clone()
            hidden[0, site] = hidden[0, site] + vec.to(hidden.dtype).to(hidden.device)
            return (hidden, *output[1:]) if isinstance(output, tuple) else hidden
        return output

    trunk = inner_transformer(model)
    handle = trunk.layers[cfg.extraction_layer].register_forward_hook(hook)

    # ‖h‖ at the site, needed to scale Δ the way the reference does.
    with torch.no_grad():
        probe: dict[str, torch.Tensor] = {}
        h_hook = trunk.layers[cfg.extraction_layer].register_forward_hook(
            lambda m, i, o: probe.__setitem__(
                "h", (o[0] if isinstance(o, tuple) else o)[0, site].detach().float().cpu()
            )
        )
        trunk(**enc)
        h_hook.remove()
    h_norm = float(probe["h"].norm())
    unit = delta / delta.norm()
    rand = torch.randn_like(unit)
    rand = rand / rand.norm()
    print(f"‖h‖ at the site = {h_norm:.1f}\n")

    def doubts(text: str) -> tuple[bool, str]:
        """Does the continuation OPEN with a doubt marker?

        Matches Finding 2's criterion — the marker must be in the next block's
        *first sentence*. A looser "anywhere in the continuation" test scores
        every sample positive, because an un-steered trace also doubts a few
        sentences later; the intervention is about whether it doubts *now*.
        """
        block = text.strip().split("\n\n")[0]
        first = re.split(r"(?<=[.!?])\s", block, maxsplit=1)[0].lower()
        hit = next((m for m in DOUBT_MARKERS if m.lower() in first), None)
        return (hit is not None), (hit or "-")

    unit_ctrl = delta_ctrl / delta_ctrl.norm()
    conditions: list[tuple[str, torch.Tensor | None, float]] = [("no injection", None, 0.0)]
    for alpha in args.alphas:
        if alpha == 0.0:
            continue
        conditions.append(("doubt Δ", unit, alpha))
        conditions.append(("continue Δ", unit_ctrl, alpha))
        conditions.append(("random Δ", rand, alpha))

    print(f"\n{'condition':<18}{'doubt rate':<12}{'uniq':>5}   example continuation")
    for name, direction, alpha in conditions:
        state["vec"] = None if direction is None else (alpha * h_norm) * direction
        hits, sample = 0, ""
        seen: set[str] = set()
        for seed in range(args.seeds):
            torch.manual_seed(seed)
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=0.6,
                    top_p=0.95,
                    top_k=20,
                )
            new = tokenizer.decode(out[0, enc["input_ids"].shape[1] :], skip_special_tokens=True)
            hit, which = doubts(new)
            hits += hit
            seen.add(new[:60])
            if seed == 0:
                sample = f"[{which}] {new[:80]!r}"
        label = name if direction is None else f"{name} α={alpha}"
        print(f"{label:<18}{hits}/{args.seeds:<7}{len(seen):>5}   {sample}")

    handle.remove()


if __name__ == "__main__":
    main()
