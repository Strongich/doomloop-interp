#!/usr/bin/env python3
"""Does the doubt direction induce self-doubt across many probes, not just one?

Turns `steer_demo.py` into a population test. For every `plain` probe — a block
boundary the model was NOT going to follow with doubt — rewrite the AV's
prediction paragraph two ways, reconstruct both through the AR, and inject the
difference:

    Δ_doubt    = AR(predict a "Wait…")      − AR(original)
    Δ_continue = AR(predict a continuation) − AR(original)

`Δ_continue` is the control that matters: an edit of the same kind, in the same
slot, with a non-doubt meaning. A random direction of matched norm is the weaker
control (random vectors in 2048-d are near-orthogonal to anything, so failing to
act is unsurprising).

Edits come from fixed templates, never per-probe by hand — hand-editing invites
fitting the edit to the outcome.

    uv run python scripts/steer_sweep.py --limit 200 --alpha 0.25

Baseline to beat: Finding 1's 23.9% doubt rate for non-answer-stating blocks.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import Counter
from functools import partial
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
    p.add_argument("--explanations", type=Path, default=Path("data/block_explanations_3way.csv"))
    p.add_argument("--out", type=Path, default=Path("data/steer_sweep.csv"))
    p.add_argument("--base", default=MODEL_ID)
    p.add_argument("--ar", type=Path, default=Path("checkpoints/ar_rl_ep1"))
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--samples", type=int, default=4, help="continuations per condition")
    p.add_argument("--max-new-tokens", type=int, default=60)
    p.add_argument(
        "--max-prefix",
        type=int,
        default=4000,
        help="skip probes with longer prefixes; the AIME traces run to 30k tokens "
        "and would dominate the wall clock without adding probes",
    )
    p.add_argument("--limit", type=int, default=200)
    return p.parse_args()


def opens_with_doubt(text: str) -> str | None:
    """Marker in the FIRST SENTENCE of the continuation, or None.

    Finding 2's criterion. "Anywhere in the continuation" scores nearly every
    sample positive, because an un-steered trace also doubts a few sentences
    later — the intervention is about whether it doubts *now*.
    """
    block = text.strip().split("\n\n")[0]
    first = re.split(r"(?<=[.!?])\s", block, maxsplit=1)[0].lower()
    return next((m for m in DOUBT_MARKERS if m.lower() in first), None)


def generate_and_score(
    model: Any,
    tokenizer: Any,
    enc: Any,
    state: dict[str, Any],
    max_new_tokens: int,
    samples: int,
    vec: torch.Tensor | None,
) -> list[str | None]:
    """Sample `samples` continuations under one injection and score each.

    A module-level function taking `enc` explicitly rather than a closure: a
    nested function capturing the loop's `enc` reads whichever probe happens to
    be current when it runs, which is the late-binding bug that produced all-NaN
    controls in the trajectory run.

    `num_return_sequences` shares the prefill across samples — the prefix is up
    to 4000 tokens and the continuation only 60, so re-prefilling per sample
    would dominate the cost.
    """
    state["vec"] = vec
    torch.manual_seed(0)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            num_return_sequences=samples,
        )
    n_in = enc["input_ids"].shape[1]
    return [opens_with_doubt(tokenizer.decode(o[n_in:], skip_special_tokens=True)) for o in out]


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float, float]:
    bb = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    cc = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    chi = (abs(bb - cc) - 1) ** 2 / (bb + cc) if bb + cc else 0.0
    p = math.erfc(math.sqrt(chi / 2)) if chi > 0 else 1.0
    return bb, cc, chi, p


def main() -> None:
    args = parse_args()
    cfg = NLAConfig()

    probes = [
        r
        for r in csv.DictReader(args.explanations.open())
        if r["kind"] == "plain" and r["explanation"].count("\n\n") >= 1
    ]
    traces = {json.loads(line)["question_id"]: json.loads(line) for line in args.traces.open()}
    print(f"{len(probes)} plain probes available")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    ar_tok, ar_backbone, affine = load_ar(args.ar)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    trunk = inner_transformer(model)

    state: dict[str, Any] = {"vec": None, "site": -1, "h": None}

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        site = state["site"]
        if hidden.shape[1] <= site:  # decode step, site already consumed
            return output
        state["h"] = hidden[0, site].detach().float().cpu()
        vec = state["vec"]
        if vec is None:
            return output
        hidden = hidden.clone()
        hidden[:, site] = hidden[:, site] + vec.to(hidden.dtype).to(hidden.device)
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    trunk.layers[cfg.extraction_layer].register_forward_hook(hook)

    rows: list[dict[str, Any]] = []
    done = 0
    for probe in probes:
        if done >= args.limit:
            break
        trace = traces.get(probe["question_id"])
        if trace is None:
            continue
        token = int(probe["token"])
        e0 = probe["explanation"]
        try:
            e_doubt = edit_explanation(e0, DOUBT_PARAGRAPH)
            e_cont = edit_explanation(e0, CONTINUE_PARAGRAPH)
        except SystemExit:
            continue

        resp_enc = tokenizer(
            trace["response"], add_special_tokens=False, return_offsets_mapping=True
        )
        if token >= len(resp_enc["offset_mapping"]):
            continue
        cut = resp_enc["offset_mapping"][token][1]
        header = _chat_header(tokenizer, trace["question"])
        n_header = len(tokenizer(header, add_special_tokens=False)["input_ids"])
        enc = tokenizer(header + trace["response"][:cut], return_tensors="pt").to("cuda")
        if enc["input_ids"].shape[1] > args.max_prefix:
            continue
        state["site"] = n_header + token

        with torch.no_grad():
            a = reconstruct(ar_tok, ar_backbone, affine, e0)
            d_doubt = (reconstruct(ar_tok, ar_backbone, affine, e_doubt) - a).cpu()
            d_cont = (reconstruct(ar_tok, ar_backbone, affine, e_cont) - a).cpu()

        run = partial(
            generate_and_score,
            model,
            tokenizer,
            enc,
            state,
            args.max_new_tokens,
            args.samples,
        )
        base_hits = run(None)  # also fills state["h"]
        h_norm = float(state["h"].norm())
        scale = args.alpha * h_norm
        rand = torch.randn_like(d_doubt)
        conds = {
            "none": base_hits,
            "doubt": run(scale * d_doubt / d_doubt.norm()),
            "continue": run(scale * d_cont / d_cont.norm()),
            "random": run(scale * rand / rand.norm()),
        }
        rows.append(
            {
                "question_id": probe["question_id"],
                "rollout_index": probe["rollout_index"],
                "dataset": probe["dataset"],
                "token": token,
                "prefix_tokens": enc["input_ids"].shape[1],
                "h_norm": round(h_norm, 1),
                "delta_doubt_norm": round(float(d_doubt.norm()), 1),
                "delta_cont_norm": round(float(d_cont.norm()), 1),
                **{f"{k}_hits": sum(1 for m in v if m) for k, v in conds.items()},
                **{
                    f"{k}_marker": (Counter(m for m in v if m).most_common(1) or [("", 0)])[0][0]
                    for k, v in conds.items()
                },
            }
        )
        done += 1
        if done % 20 == 0:
            print(
                f"  [{done}/{args.limit}] "
                + "  ".join(
                    f"{k} {sum(r[f'{k}_hits'] for r in rows)}/{done * args.samples}" for k in conds
                )
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {args.out} ({len(rows)} probes x {args.samples} samples)\n")

    n = len(rows)
    tot = n * args.samples
    print(f"alpha = {args.alpha}   {n} probes   {args.samples} samples each")
    print(f"{'condition':<12}{'doubt rate':>14}{'probes with >=1':>18}")
    for k in ("none", "doubt", "continue", "random"):
        hits = sum(r[f"{k}_hits"] for r in rows)
        any_ = sum(1 for r in rows if r[f"{k}_hits"] > 0)
        print(f"{k:<12}{hits}/{tot} = {100 * hits / tot:5.1f}%{any_:>13} ({100 * any_ / n:.0f}%)")

    print("\npaired per probe (>=1 doubt in the samples):")
    per = {k: [r[f"{k}_hits"] > 0 for r in rows] for k in ("none", "doubt", "continue", "random")}
    for k in ("doubt", "continue", "random"):
        bb, cc, chi, p = mcnemar(per[k], per["none"])
        tag = "vs none"
        print(f"  {k:<9} {tag}   b={bb:<4} c={cc:<4} chi2={chi:6.1f}  p = {p:.2g}")
    bb, cc, chi, p = mcnemar(per["doubt"], per["continue"])
    print(f"  doubt vs continue   b={bb:<4} c={cc:<4} chi2={chi:6.1f}  p = {p:.2g}")


if __name__ == "__main__":
    main()
