#!/usr/bin/env python3
"""How faithfully does each AV explanation reconstruct its own activation?

The AR is the NLA's other half and has been used for one thing only — producing
the steering `Δ` of Finding 3. Its actual job is this: read an explanation and
predict the activation it describes. The reconstruction error is therefore a
**per-row trustworthiness score** for an explanation, which is exactly what the
confabulation problem needs (D48: the AV invented "divisible by 3" for a 24-gon
chord argument, confidently and in the right semantic neighbourhood).

Two numbers per row:

  ar_cos   cos(AR(e), h) — the AR's loss is direction-only (both prediction and
           target are normalized to the injection scale), so this is the quantity
           it was actually trained on.
  ar_fve   1 − MSE(n(AR(e)), n(h)) / baseline, with the dataset-level `meannorm`
           baseline from `training.data.predict_mean_baselines` — the same
           denominator the SFT/RL runs reported, so a row's score is comparable
           to the checkpoint's 0.630.

Activations are re-extracted rather than loaded: `verbalize_blocks.py` dropped
them after verbalizing. One forward pass per trace covers all of its probes.

    uv run python scripts/ar_fidelity.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steer_demo import load_ar, reconstruct  # noqa: E402

from reasoning_attention.config import MODEL_ID, NLAConfig  # noqa: E402
from reasoning_attention.nla.injection import normalize_activation  # noqa: E402
from reasoning_attention.tokenview import StateCache, build_view  # noqa: E402
from reasoning_attention.training.data import predict_mean_baselines  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/correct_sample.jsonl"))
    p.add_argument("--explanations", type=Path, default=Path("data/block_explanations_3way.csv"))
    p.add_argument("--out", type=Path, default=Path("data/block_explanations_fve.csv"))
    p.add_argument("--base", default=MODEL_ID)
    p.add_argument("--ar", type=Path, default=Path("checkpoints/ar_rl_ep1"))
    p.add_argument("--limit", type=int, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NLAConfig()
    scale = cfg.resolve_injection_scale(2048)

    rows = list(csv.DictReader(args.explanations.open()))
    traces = {json.loads(line)["question_id"]: json.loads(line) for line in args.traces.open()}
    by_trace: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for r in rows:
        by_trace[r["question_id"]].append(r)
    keys = list(by_trace)[: args.limit] if args.limit else list(by_trace)
    print(f"{sum(len(by_trace[k]) for k in keys)} rows over {len(keys)} traces")

    # --- pass 1: the true activations -------------------------------------
    cache = StateCache(args.base, cfg.extraction_layer)
    golds: dict[tuple[str, str], torch.Tensor] = {}
    for n, qid in enumerate(keys, 1):
        trace = traces[qid]
        view = build_view(cache.tokenizer, trace["question"], trace["response"], trace["gold"])
        cache.fill(view)
        states = view.states
        assert states is not None
        for r in by_trace[qid]:
            idx = int(r["token"])
            if idx < states.shape[0]:
                golds[(qid, r["kind"])] = states[idx].float().cpu().clone()
        view.states = None
        del states
        torch.cuda.empty_cache()
        if n % 50 == 0 or n == len(keys):
            print(f"  extracted [{n}/{len(keys)}]")
    del cache
    torch.cuda.empty_cache()

    stacked = np.stack([v.numpy() for v in golds.values()])
    baselines = predict_mean_baselines(stacked, scale)
    print(
        f"\nbaselines over {len(golds)} states: meannorm {baselines.meannorm:.4f}  "
        f"rawvar {baselines.rawvar:.4f}"
    )

    # --- pass 2: reconstruct each explanation ------------------------------
    ar_tok, ar_backbone, affine = load_ar(args.ar)
    out: list[dict[str, Any]] = []
    for n, qid in enumerate(keys, 1):
        for r in by_trace[qid]:
            gold = golds.get((qid, r["kind"]))
            if gold is None:
                continue
            with torch.no_grad():
                pred = reconstruct(ar_tok, ar_backbone, affine, r["explanation"]).cpu()
            gn = normalize_activation(gold[None], scale)
            pn = normalize_activation(pred[None], scale)
            mse = float(((pn - gn) ** 2).mean())
            out.append(
                {
                    **r,
                    "ar_cos": round(
                        float(torch.nn.functional.cosine_similarity(pred[None], gold[None])), 4
                    ),
                    "ar_mse": round(mse, 4),
                    "ar_fve": round(1 - mse / baselines.meannorm, 4),
                }
            )
        if n % 50 == 0 or n == len(keys):
            print(f"  reconstructed [{n}/{len(keys)}]")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        w.writerows(out)
    print(f"\nwrote {args.out} ({len(out)} rows)\n")

    print(f"{'kind':<13}{'n':>5}{'ar_fve mean':>13}{'median':>9}{'ar_cos mean':>13}{'fve<0':>8}")
    for kind in ("doubt_wait", "doubt_other", "plain", "think_last"):
        sub = [r for r in out if r["kind"] == kind]
        if not sub:
            continue
        f = [r["ar_fve"] for r in sub]
        c = [r["ar_cos"] for r in sub]
        neg = sum(1 for x in f if x < 0)
        print(
            f"{kind:<13}{len(sub):>5}{st.mean(f):>13.3f}{st.median(f):>9.3f}"
            f"{st.mean(c):>13.3f}{neg:>8}"
        )


if __name__ == "__main__":
    main()
