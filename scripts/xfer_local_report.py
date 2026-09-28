#!/usr/bin/env python3
"""Summarize the single-site injection screen, 1.7B vs 8B on the same sites.

Per arm, split by what the model actually wrote next (doubt / plain site):
  P0 -> P1   mean P(doubt opener) without / with the injection
  rel        1 - P1/P0 (suppression > 0; induction < 0)
  KL         mean KL(base || steered) of the next-token distribution
  r(site)    correlation, across the shared sites, of this arm's log-ratio
             log(P1/P0) with the 1.7B N@alpha=1 log-ratio (does it move the SAME sites?)

    uv run python scripts/xfer_local_report.py
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def load(p: Path) -> dict:
    out = {}
    for line in p.open():
        r = json.loads(line)
        out[(r["origin"], r["qid"], r["pos"])] = r
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", type=Path, default=Path("data/xfer8b/local/17b.jsonl"))
    ap.add_argument("--big", type=Path, default=Path("data/xfer8b/local/8b.jsonl"))
    ap.add_argument("--ref", default="L20:N:1.0")
    args = ap.parse_args()
    S, B = load(args.small), load(args.big)
    keys = sorted(set(S) & set(B))
    print(f"{len(keys)} shared sites ({sum(S[k]['label'] for k in keys)} doubt)")

    def lr(rec, arm):
        return math.log(max(rec["arms"][arm]["p"], 1e-9) / max(rec["p_base"], 1e-9))

    ref = {k: lr(S[k], args.ref) for k in keys}

    def summarize(tag, D, arm):
        cells = []
        for lab in (1, 0):
            ks = [k for k in keys if D[k]["label"] == lab]
            p0 = np.mean([D[k]["p_base"] for k in ks])
            p1 = np.mean([D[k]["arms"][arm]["p"] for k in ks])
            kl = np.mean([D[k]["arms"][arm]["kl"] for k in ks])
            cells.append(f"{'doubt' if lab else 'plain'} {p0:.3f}->{p1:.3f} ({100*(1-p1/p0):+4.0f}%) KL {kl:.2f}")
        x = np.array([ref[k] for k in keys])
        y = np.array([lr(D[k], arm) for k in keys])
        r = np.corrcoef(x, y)[0, 1] if y.std() > 0 else float("nan")
        print(f"{tag:4s} {arm:14s} " + " | ".join(cells) + f" | r(site) {r:+.2f}")

    arms_s = list(next(iter(S.values()))["arms"])
    arms_b = list(next(iter(B.values()))["arms"])
    for a in arms_s:
        summarize("1.7B", S, a)
    for a in arms_b:
        summarize("8B", B, a)


if __name__ == "__main__":
    main()
