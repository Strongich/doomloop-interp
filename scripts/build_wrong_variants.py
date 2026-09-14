#!/usr/bin/env python3
"""Build suppression directions from the WRONG-trace pool, matched to arms A and E.

Two vectors, each constructed exactly like its correct-trace twin so the only
thing that differs is where the deltas came from:

    W   ~35 deltas, ONE wrong rollout, all depths, normalized mean   (twin of A)
    WP  all deltas, 30 wrong rollouts, all depths, normalized mean   (twin of E)

W alone cannot separate "derived from a failure" from "derived from that
particular rollout"; WP is what makes the comparison interpretable.

The cosines printed at the end are already a result. If cos(W, A) sits up with
the 0.93-0.999 that Finding 5b measured between construction methods, then the
model writes the same doubt direction whether or not it is getting the answer
right, and the behavioural arms should agree. A markedly lower cosine is the
first evidence in this study that the direction depends on the outcome.

    uv run python scripts/build_wrong_variants.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/pool_wrong/raw.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("data/pool_wrong"))
    p.add_argument("--matched-count", type=int, default=35)
    p.add_argument("--n-traces", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def direction(deltas: torch.Tensor, normalized: bool = True) -> torch.Tensor:
    if normalized:
        deltas = deltas / deltas.norm(dim=1, keepdim=True).clamp_min(1e-9)
    mean = deltas.mean(0)
    return mean / mean.norm()


def main() -> None:
    args = parse_args()
    blob = torch.load(args.raw, map_location="cpu", weights_only=False)
    deltas, meta = blob["deltas"], blob["meta"]

    traces = sorted({(m["question_id"], m["rollout_index"]) for m in meta})
    by_trace: dict[tuple[str, Any], list[int]] = {t: [] for t in traces}
    for i, m in enumerate(meta):
        by_trace[(m["question_id"], m["rollout_index"])].append(i)
    print(f"wrong pool: {len(deltas)} deltas over {len(traces)} traces\n")

    import random

    rng = random.Random(args.seed)
    K = args.matched_count

    rich = [t for t in traces if len(by_trace[t]) >= K]
    if not rich:
        rich = sorted(traces, key=lambda t: -len(by_trace[t]))[:1]
        print(f"note: no wrong trace has {K} boundaries; using the richest "
              f"({len(by_trace[rich[0]])})")
    w_trace = rng.choice(rich)
    w_idx = by_trace[w_trace][:K]

    shuffled = traces[:]
    rng.shuffle(shuffled)
    wp_traces = shuffled[: args.n_traces]
    wp_idx = [i for t in wp_traces for i in by_trace[t]]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out = {
        "dir_W_wrong1trace.pt": (
            w_idx,
            {"n": float(len(w_idx)), "traces": 1.0, "depths": "all", "source": "wrong"},
        ),
        "dir_WP_wrong30traces.pt": (
            wp_idx,
            {
                "n": float(len(wp_idx)),
                "traces": float(len(wp_traces)),
                "depths": "all",
                "source": "wrong",
            },
        ),
    }
    print(f"W source trace: {w_trace[0]} rollout {w_trace[1]}")
    print("building:")
    for name, (idx, stats) in out.items():
        torch.save(
            {"unit": direction(deltas[idx]), "stats": stats, "edit": "continue"},
            args.out_dir / name,
        )
        print(f"  {name:<26} {stats}")

    # Cosine against every correct-trace direction we already have. This is the
    # cheap half of the experiment: it says whether the two families of vectors
    # are even different before a single token is generated.
    others = {
        "A_correct1trace": Path("data/pool/dir_A_1trace.pt"),
        "E_correct30traces": Path("data/pool/dir_E_30traces.pt"),
        "G_global297": Path("data/delta_suppress_mean.pt"),
    }
    units = {n: torch.load(args.out_dir / n, weights_only=False)["unit"] for n in out}
    for n, p in others.items():
        if p.exists():
            units[n] = torch.load(p, weights_only=False)["unit"].float()

    names = list(units)
    print("\npairwise cosine (wrong-derived vs correct-derived):")
    print("                    " + "".join(f"{n[:11]:>13}" for n in names))
    for a in names:
        row = "".join(
            f"{torch.nn.functional.cosine_similarity(units[a][None].float(), units[b][None].float()).item():>13.3f}"
            for b in names
        )
        print(f"{a[:19]:<20}" + row)

    (args.out_dir / "variants.json").write_text(
        json.dumps(
            {
                "W": f"{len(w_idx)}d/1trace/all/wrong ({w_trace[0]} r{w_trace[1]})",
                "WP": f"{len(wp_idx)}d/{len(wp_traces)}traces/all/wrong",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
