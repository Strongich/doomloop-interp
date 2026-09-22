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
from reasoning_attention.loops import DOUBT_MARKERS, marker_matches  # noqa: E402
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
    p.add_argument(
        "--kind",
        default="plain",
        choices=("plain", "doubt_wait", "doubt_other"),
        help="which probe population. `plain` (default) is Finding 3's induction "
        "test; `doubt_wait` is the mirrored suppression test.",
    )
    p.add_argument(
        "--mode",
        default="induce",
        choices=("induce", "suppress"),
        help="reporting only — it renames the conditions and picks which paired "
        "comparisons are printed. The injected vectors are identical either way: "
        "the continuation edit IS the suppression edit, it just acts on probes "
        "that were about to doubt instead of probes that were not.",
    )
    p.add_argument(
        "--dump-examples",
        type=Path,
        default=None,
        help="write the first --n-examples probes' full text (context, both "
        "explanations, one continuation per condition) to this json",
    )
    p.add_argument("--n-examples", type=int, default=6)
    p.add_argument(
        "--delta",
        default="per_probe",
        help="per_probe is Finding 3's recipe: Δ from THIS probe's own "
        "explanation. `mean` substitutes one direction averaged over the whole "
        "probe set — the cheap approximation a whole-trace intervention has to "
        "use, measured here against the thing it approximates. "
        "`file:<path>` loads a fixed direction from derive_direction.py, to test "
        "whether a vector built elsewhere transfers to these probes.",
    )
    p.add_argument(
        "--exclude-trace",
        default=None,
        help="question_id to drop from the probe set — use it to hold out the "
        "trace a `file:` direction was derived from. Steering the trace you "
        "derived from is not a test.",
    )
    p.add_argument("--datasets", default=None, help="comma-separated dataset filter")
    p.add_argument(
        "--only-traces",
        type=Path,
        default=None,
        help="json list of question_ids to keep — the held-out eval split for a "
        "direction built from other probes. Without it a pooled direction would "
        "be scored partly on the probes it was built from.",
    )
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


def opens_with_doubt_wb(text: str) -> str | None:
    """Word-bounded variant of `opens_with_doubt`, as a robustness column.

    The primary criterion is substring matching, kept verbatim so Finding 3 stays
    reproducible; it can fire on "but" inside "distributed". This one requires a
    whole-word match. It is recorded, never used in place of the primary.
    """
    block = text.strip().split("\n\n")[0]
    first = re.split(r"(?<=[.!?])\s", block, maxsplit=1)[0]
    return next((m for m in DOUBT_MARKERS if marker_matches(first, m, case_sensitive=False)), None)


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
    texts = [tokenizer.decode(o[n_in:], skip_special_tokens=True) for o in out]
    state["texts"] = texts
    state["wb"] = [opens_with_doubt_wb(t) for t in texts]
    return [opens_with_doubt(t) for t in texts]


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
        if r["kind"] == args.kind and r["explanation"].count("\n\n") >= 1
    ]
    if args.exclude_trace:
        n0 = len(probes)
        probes = [r for r in probes if r["question_id"] != args.exclude_trace]
        print(f"held out {n0 - len(probes)} probe(s) from {args.exclude_trace}")
    if args.datasets:
        keep = {d.strip() for d in args.datasets.split(",")}
        probes = [r for r in probes if r["dataset"] in keep]
    if args.only_traces:
        keep_ids = set(json.loads(args.only_traces.read_text()))
        probes = [r for r in probes if r["question_id"] in keep_ids]
        print(f"restricted to {len(probes)} held-out probes")
    traces = {json.loads(line)["question_id"]: json.loads(line) for line in args.traces.open()}
    print(f"{len(probes)} {args.kind} probes available  (mode={args.mode})")

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

    mean_dirs: dict[str, torch.Tensor] = {}
    if args.delta.startswith("file:"):
        payload = torch.load(args.delta[5:], map_location="cpu", weights_only=False)
        src = payload.get("source", {})
        print(f"direction from {args.delta[5:]}: {src}")
        for name in ("doubt", "continue"):
            if name in payload:
                # Stored as a UNIT vector; the injection rescales by alpha*||h||
                # anyway, so magnitude here is irrelevant.
                mean_dirs[name] = payload[name]["unit"].float()
            elif name == "continue" and "unit" in payload:
                mean_dirs[name] = payload["unit"].float()   # delta_suppress_mean.pt layout
        if "doubt" not in mean_dirs:
            mean_dirs["doubt"] = -mean_dirs["continue"]
    elif args.delta == "mean":
        acc: dict[str, list[torch.Tensor]] = {"doubt": [], "continue": []}
        with torch.no_grad():
            for probe in probes:
                e0 = probe["explanation"]
                a = reconstruct(ar_tok, ar_backbone, affine, e0)
                for name, para in (
                    ("doubt", DOUBT_PARAGRAPH),
                    ("continue", CONTINUE_PARAGRAPH),
                ):
                    acc[name].append(
                        (reconstruct(ar_tok, ar_backbone, affine, edit_explanation(e0, para)) - a)
                        .float()
                        .cpu()
                    )
        for name, vs in acc.items():
            stack = torch.stack(vs)
            m = stack.mean(0)
            cos = torch.nn.functional.cosine_similarity(stack, m[None], dim=-1)
            print(
                f"mean Δ_{name}: n={len(vs)}  ‖mean‖={m.norm():.1f}  "
                f"mean‖Δ_i‖={stack.norm(dim=-1).mean():.1f}  "
                f"cos(Δ_i, mean) mean={cos.mean():.3f} min={cos.min():.3f} "
                f"p10={cos.quantile(0.1):.3f}"
            )
            mean_dirs[name] = m / m.norm()

    rows: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
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

        if mean_dirs:
            d_doubt, d_cont = mean_dirs["doubt"], mean_dirs["continue"]
        else:
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
        example: dict[str, Any] = {"none": state["texts"][0]}
        wb: dict[str, list[str | None]] = {"none": state["wb"]}
        h_norm = float(state["h"].norm())
        scale = args.alpha * h_norm
        rand = torch.randn_like(d_doubt)
        conds = {"none": base_hits}
        for name, vec in (
            ("doubt", d_doubt),
            ("continue", d_cont),
            ("random", rand),
        ):
            conds[name] = run(scale * vec / vec.norm())
            example[name] = state["texts"][0]
            wb[name] = state["wb"]
        if args.dump_examples is not None and len(examples) < args.n_examples:
            examples.append(
                {
                    "question_id": probe["question_id"],
                    "kind": probe["kind"],
                    "token": token,
                    "gold": probe["gold"],
                    "next_marker": probe["next_marker"],
                    "context_tail": probe["context_tail"],
                    "actual_next": trace["response"][cut : cut + 200],
                    "explanation": e0,
                    "explanation_doubt": e_doubt,
                    "explanation_continue": e_cont,
                    "h_norm": round(h_norm, 1),
                    "delta_doubt_norm": round(float(d_doubt.norm()), 1),
                    "delta_cont_norm": round(float(d_cont.norm()), 1),
                    "hits": {k: [m or "-" for m in v] for k, v in conds.items()},
                    "continuation": example,
                }
            )
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
                **{f"{k}_seq": "".join("1" if m else "0" for m in v) for k, v in conds.items()},
                **{f"{k}_hits_wb": sum(1 for m in v if m) for k, v in wb.items()},
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

    if args.dump_examples is not None:
        args.dump_examples.parent.mkdir(parents=True, exist_ok=True)
        args.dump_examples.write_text(json.dumps(examples, indent=2))
        print(f"wrote {args.dump_examples} ({len(examples)} examples)")

    n = len(rows)
    tot = n * args.samples
    # The vectors are the same in both modes; only the framing differs. In
    # suppression mode the continuation edit is the *treatment* and the doubt
    # edit is the same-kind-of-edit control, so relabel rather than recompute.
    label = (
        {"none": "none", "doubt": "doubt", "continue": "continue", "random": "random"}
        if args.mode == "induce"
        else {"none": "none", "doubt": "doubt", "continue": "suppress", "random": "random"}
    )
    order = (
        ("none", "doubt", "continue", "random")
        if args.mode == "induce"
        else ("none", "continue", "doubt", "random")
    )
    print(f"alpha = {args.alpha}   kind = {args.kind}   mode = {args.mode}")
    print(f"{n} probes   {args.samples} samples each")
    print(f"{'condition':<12}{'doubt rate':>14}{'probes with >=1':>18}")
    for k in order:
        hits = sum(r[f"{k}_hits"] for r in rows)
        any_ = sum(1 for r in rows if r[f"{k}_hits"] > 0)
        print(
            f"{label[k]:<12}{hits}/{tot} = {100 * hits / tot:5.1f}%"
            f"{any_:>13} ({100 * any_ / n:.0f}%)"
        )

    print("\npaired per probe (>=1 doubt in the samples):")
    per = {k: [r[f"{k}_hits"] > 0 for r in rows] for k in ("none", "doubt", "continue", "random")}
    pairs = (
        [("doubt", "none"), ("continue", "none"), ("random", "none"), ("doubt", "continue")]
        if args.mode == "induce"
        else [
            ("continue", "none"),
            ("continue", "doubt"),
            ("continue", "random"),
            ("doubt", "none"),
            ("random", "none"),
        ]
    )
    for k, ref in pairs:
        bb, cc, chi, pv = mcnemar(per[k], per[ref])
        tag = f"{label[k]} vs {label[ref]}"
        print(f"  {tag:<22} b={bb:<4} c={cc:<4} chi2={chi:6.1f}  p = {pv:.2g}")

    if args.mode == "suppress":
        # "did this probe doubt at all" saturates here: these probes were
        # SELECTED as doubt-followed, so the baseline is at ceiling and the
        # per-probe presence test has no room to move. Two readouts with room:
        # the mirrored per-probe binary (did any sample NOT doubt) and the
        # sample-level pairing, which is legitimate because every condition is
        # generated from the same seed and therefore the same random stream.
        print("\nmirrored per probe (>=1 sample did NOT doubt):")
        anti = {k: [r[f"{k}_hits"] < args.samples for r in rows] for k in label}
        for k, ref in pairs:
            bb, cc, chi, pv = mcnemar(anti[k], anti[ref])
            tag = f"{label[k]} vs {label[ref]}"
            print(f"  {tag:<22} b={bb:<4} c={cc:<4} chi2={chi:6.1f}  p = {pv:.2g}")

        print("\nsample-level, paired on the shared random stream:")
        seq = {k: [c == "1" for r in rows for c in r[f"{k}_seq"]] for k in label}
        for k, ref in pairs:
            bb, cc, chi, pv = mcnemar(seq[k], seq[ref])
            tag = f"{label[k]} vs {label[ref]}"
            print(f"  {tag:<22} b={bb:<4} c={cc:<4} chi2={chi:6.1f}  p = {pv:.2g}")

        print("\nword-bounded marker matching (robustness, not the primary):")
        for k in order:
            hits = sum(r[f"{k}_hits_wb"] for r in rows)
            print(f"  {label[k]:<12}{hits}/{tot} = {100 * hits / tot:5.1f}%")


if __name__ == "__main__":
    main()
