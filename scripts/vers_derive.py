#!/usr/bin/env python3
"""VERS in both directions from N's 35 sites (EXPERIMENT-vers-doubt.md §2).

N (`data/pool/dir_A_1trace.pt`) is the continue edit averaged over the 35 doubt
boundaries of one `aime2025:1` rollout. This re-runs the AV once per site and
builds BOTH edits from the same explanation e0:

    Δ_i(edit) = AR(edit(e0_i, PARAGRAPH)) − AR(e0_i)
    N′ = direction(Δ(continue)),  P = direction(Δ(doubt))

with `direction` = N's recipe (`build_variants.direction(…, normalized=True)`):
normalize each Δ_i, average, normalize. The plain mean is reported for reference.
The rebuild check is cos(N′, N) ≥ 0.90.

    uv run python scripts/vers_derive.py --out-dir data/vers
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from build_variants import direction  # noqa: E402
from derive_direction import doubt_boundaries  # noqa: E402
from steer_demo import (  # noqa: E402
    CONTINUE_PARAGRAPH,
    DOUBT_PARAGRAPH,
    edit_explanation,
    load_ar,
    reconstruct,
)

from reasoning_attention.config import MODEL_ID, NLAConfig  # noqa: E402
from reasoning_attention.nla.arch import inner_transformer  # noqa: E402
from reasoning_attention.tokenview import _chat_header  # noqa: E402

EDITS = (("continue", CONTINUE_PARAGRAPH), ("doubt", DOUBT_PARAGRAPH))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/correct_sample.jsonl"))
    p.add_argument("--raw", type=Path, default=Path("data/pool/raw.pt"))
    p.add_argument("--n", type=Path, default=Path("data/pool/dir_A_1trace.pt"))
    p.add_argument("--question-id", default="aime2025:1")
    p.add_argument("--out-dir", type=Path, default=Path("data/vers"))
    p.add_argument("--av", type=Path, default=Path("checkpoints/av_rl_ep1"))
    p.add_argument("--ar", type=Path, default=Path("checkpoints/ar_rl_ep1"))
    p.add_argument("--av-max-new", type=int, default=300)
    p.add_argument("--max-prefix", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-cos", type=float, default=0.90)
    return p.parse_args()


def stats(stack: torch.Tensor) -> dict[str, float]:
    unit = direction(stack, True)
    mean = stack.mean(0)
    cos = torch.nn.functional.cosine_similarity(stack, unit[None], dim=1)
    return {
        "n": float(stack.shape[0]),
        "mean_norm": float(stack.norm(dim=1).mean()),
        "norm_of_mean": float(mean.norm()),
        "cos_to_mean_mean": float(cos.mean()),
        "cos_to_mean_min": float(cos.min()),
    }


def main() -> None:
    args = parse_args()
    cfg = NLAConfig()
    torch.manual_seed(args.seed)
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from reasoning_attention.nla.model import NLA

    # Sites: exactly the raw.pt rows N was averaged over, in order.
    raw = torch.load(args.raw, map_location="cpu", weights_only=False)
    rows = [m for m in raw["meta"] if m["question_id"] == args.question_id]
    rollouts = {str(m["rollout_index"]) for m in rows}
    assert len(rollouts) == 1, rollouts
    trace = next(
        t for t in map(json.loads, args.traces.open())
        if t["question_id"] == args.question_id and str(t["rollout_index"]) in rollouts
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    cuts = dict(doubt_boundaries(trace["response"], tokenizer, args.max_prefix))
    sites = [(m["token"], cuts[m["token"]]) for m in rows]
    print(f"{args.question_id} rollout {trace['rollout_index']}: {len(sites)} sites")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    trunk = inner_transformer(model)
    nla = NLA.av_only(str(args.av))
    nla.av.eval()
    ar_tok, ar_backbone, affine = load_ar(args.ar)

    header = _chat_header(tokenizer, trace["question"])
    n_header = len(tokenizer(header, add_special_tokens=False)["input_ids"])
    captured: dict[str, Any] = {"h": None, "site": -1}

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > captured["site"]:
            captured["h"] = hidden[0, captured["site"]].detach().float().cpu()
        return output

    handle = trunk.layers[cfg.extraction_layer].register_forward_hook(hook)
    deltas: dict[str, list[torch.Tensor]] = {name: [] for name, _ in EDITS}
    records: list[dict[str, Any]] = []
    try:
        for k, (ti, cut) in enumerate(sites, 1):
            enc = tokenizer(header + trace["response"][:cut], return_tensors="pt").to("cuda")
            assert enc["input_ids"].shape[1] <= args.max_prefix, (ti, enc["input_ids"].shape)
            captured["site"], captured["h"] = n_header + ti, None
            with torch.no_grad():
                trunk(**enc)
            h = captured["h"]
            assert h is not None, ti
            e0 = nla.verbalize(h, max_new_tokens=args.av_max_new, return_explanation=True).strip()
            # Both edits from the SAME e0; a site where e0 cannot be edited is dropped
            # from both sets, so N′ and P average over identical sites.
            try:
                edited = {name: edit_explanation(e0, para) for name, para in EDITS}
            except SystemExit:
                print(f"  [{k}/{len(sites)}] token {ti}: no prediction paragraph, skipped")
                records.append({"token": ti, "explanation": e0, "used": False})
                continue
            with torch.no_grad():
                a = reconstruct(ar_tok, ar_backbone, affine, e0)
                for name, ed in edited.items():
                    d = reconstruct(ar_tok, ar_backbone, affine, ed) - a
                    deltas[name].append(d.float().cpu())
            records.append({"token": ti, "explanation": e0, "used": True,
                            "h_norm": float(h.norm())})
            print(f"  [{k}/{len(sites)}] token {ti}", flush=True)
    finally:
        handle.remove()

    n_unit = torch.load(args.n, map_location="cpu", weights_only=False)["unit"].float()
    cos = torch.nn.functional.cosine_similarity
    out: dict[str, Any] = {}
    for name, vs in deltas.items():
        stack = torch.stack(vs)
        plain = stack.mean(0)
        out[name] = {
            "unit": direction(stack, True),
            "unit_plainmean": plain / plain.norm(),
            "stats": stats(stack),
            "deltas": stack,
        }
    np_, p = out["continue"]["unit"], out["doubt"]["unit"]

    def c(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(cos(a[None], b[None]))

    checks = {
        "n_sites": len(sites),
        "n_used": len(deltas["continue"]),
        "cos_Nprime_N": c(np_, n_unit),
        "cos_P_N": c(p, n_unit),
        "cos_P_negN": c(p, -n_unit),
        "cos_P_Nprime": c(p, np_),
        "plainmean_cos_Nprime_N": c(out["continue"]["unit_plainmean"], n_unit),
        "plainmean_cos_P_N": c(out["doubt"]["unit_plainmean"], n_unit),
        "cos_P_plain_vs_norm": c(out["doubt"]["unit_plainmean"], p),
        "continue": out["continue"]["stats"],
        "doubt": out["doubt"]["stats"],
        "rebuild_pass": c(np_, n_unit) >= args.min_cos,
        "negN_redundant": c(p, -n_unit) >= 0.95,
        "seed": args.seed,
    }
    print(json.dumps(checks, indent=2))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    source = {"question_id": args.question_id, "rollout_index": trace["rollout_index"],
              "tokens": [t for t, _ in sites]}
    for name, fname in (("doubt", "dir_P_doubt.pt"), ("continue", "dir_Nprime_continue.pt")):
        torch.save({**out[name], "edit": name, "source": source, "records": records,
                    "explanations": [r["explanation"] for r in records]},
                   args.out_dir / fname)
    torch.save({"unit": -n_unit, "stats": {"source": str(args.n)}, "edit": "negN"},
               args.out_dir / "dir_negN.pt")
    (args.out_dir / "derive.json").write_text(json.dumps(checks, indent=2))
    print(f"wrote {args.out_dir}/dir_P_doubt.pt, dir_Nprime_continue.pt, dir_negN.pt")
    if not checks["rebuild_pass"]:
        raise SystemExit(f"REBUILD CHECK FAILED: cos(N′,N)={checks['cos_Nprime_N']:.3f}")


if __name__ == "__main__":
    main()
