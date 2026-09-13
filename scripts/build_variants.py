#!/usr/bin/env python3
"""Build the factorial of suppression directions from one raw delta pool.

Separates the three axes that every earlier direction varied together:

    A  40 deltas,  1 trace,   all depths   -- diversity floor
    B  40 deltas,  40 traces, all depths   -- vs A: does trace diversity matter?
    C  40 deltas,  40 traces, FIRST only   -- vs B: does depth bias matter?
    D  all deltas, 10 traces, all depths   -- count up, diversity fixed low
    E  all deltas, 30 traces, all depths   -- count up further
    F  same set as E, SIMPLE mean          -- vs E: does the averaging rule matter?

A-E use a NORMALIZED mean: each delta is scaled to unit length before averaging,
so every block gets an equal vote. The simple mean weights by magnitude, and
||delta|| varies about fourfold across probes, which lets a handful of large
blocks dominate. Finding 5's original vector used the simple mean, so F is the
like-for-like bridge back to it.

Written in the flat {unit, stats} layout `suppress_answer.py` expects.

    uv run python scripts/build_variants.py --raw data/pool/raw.pt --out-dir data/pool
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raw", type=Path, default=Path("data/pool/raw.pt"))
    p.add_argument("--out-dir", type=Path, default=Path("data/pool"))
    p.add_argument("--matched-count", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def direction(deltas: torch.Tensor, normalized: bool) -> torch.Tensor:
    if normalized:
        deltas = deltas / deltas.norm(dim=1, keepdim=True).clamp_min(1e-9)
    mean = deltas.mean(0)
    return mean / mean.norm()


def save(path: Path, unit: torch.Tensor, stats: dict[str, Any]) -> None:
    torch.save({"unit": unit, "stats": stats, "edit": "continue"}, path)
    print(f"  {path.name:<22} {stats}")


def main() -> None:
    args = parse_args()
    blob = torch.load(args.raw, map_location="cpu", weights_only=False)
    deltas, meta = blob["deltas"], blob["meta"]
    args.out_dir.mkdir(parents=True, exist_ok=True)

    traces = sorted({(m["question_id"], m["rollout_index"]) for m in meta})
    by_trace: dict[tuple[str, Any], list[int]] = {t: [] for t in traces}
    for i, m in enumerate(meta):
        by_trace[(m["question_id"], m["rollout_index"])].append(i)
    print(f"pool: {len(deltas)} deltas over {len(traces)} traces\n")

    rng = random.Random(args.seed)
    K = args.matched_count

    # A — matched count, ONE trace. Pick a trace with at least K boundaries.
    rich = [t for t in traces if len(by_trace[t]) >= K]
    if not rich:
        rich = sorted(traces, key=lambda t: -len(by_trace[t]))[:1]
        print(f"note: no trace has {K} boundaries; using the richest "
              f"({len(by_trace[rich[0]])})")
    a_trace = rng.choice(rich)
    a_idx = by_trace[a_trace][:K]

    # B — matched count, spread over as many DIFFERENT traces as possible.
    shuffled = traces[:]
    rng.shuffle(shuffled)
    b_idx = [rng.choice(by_trace[t]) for t in shuffled[:K]]

    # C — matched count, different traces, FIRST boundary only (Finding 5's bias).
    firsts = {t: min(by_trace[t], key=lambda i: meta[i]["depth"]) for t in traces}
    c_idx = [firsts[t] for t in shuffled[:K]]

    # D / E — count up, diversity fixed. All boundaries of n traces.
    d_traces, e_traces = shuffled[:10], shuffled[:30]
    d_idx = [i for t in d_traces for i in by_trace[t]]
    e_idx = [i for t in e_traces for i in by_trace[t]]

    def st(idx: list[int], traces_used: int, depths: str) -> dict[str, Any]:
        return {"n": float(len(idx)), "traces": float(traces_used), "depths": depths}

    print("building:")
    save(args.out_dir / "dir_A_1trace.pt", direction(deltas[a_idx], True),
         st(a_idx, 1, "all"))
    save(args.out_dir / "dir_B_spread.pt", direction(deltas[b_idx], True),
         st(b_idx, len(shuffled[:K]), "all"))
    save(args.out_dir / "dir_C_firstonly.pt", direction(deltas[c_idx], True),
         st(c_idx, len(shuffled[:K]), "first"))
    save(args.out_dir / "dir_D_10traces.pt", direction(deltas[d_idx], True),
         st(d_idx, len(d_traces), "all"))
    save(args.out_dir / "dir_E_30traces.pt", direction(deltas[e_idx], True),
         st(e_idx, len(e_traces), "all"))
    save(args.out_dir / "dir_F_30traces_simple.pt", direction(deltas[e_idx], False),
         st(e_idx, len(e_traces), "all-simple-mean"))

    # Cosines between every pair — if these are all ~0.95 the arms cannot differ
    # much behaviourally, and that is itself the headline.
    names = ["A_1trace", "B_spread", "C_firstonly", "D_10traces", "E_30traces",
             "F_30traces_simple"]
    units = [torch.load(args.out_dir / f"dir_{n}.pt", weights_only=False)["unit"] for n in names]
    print("\npairwise cosine:")
    print("        " + "".join(f"{n[:9]:>11}" for n in names))
    for i, n in enumerate(names):
        row = "".join(
            f"{torch.nn.functional.cosine_similarity(units[i][None], u[None]).item():>11.3f}"
            for u in units
        )
        print(f"{n[:8]:<8}" + row)

    (args.out_dir / "variants.json").write_text(json.dumps(
        {"A": "40d/1trace/all", "B": "40d/40traces/all", "C": "40d/40traces/first",
         "D": f"{len(d_idx)}d/10traces/all", "E": f"{len(e_idx)}d/30traces/all",
         "F": f"{len(e_idx)}d/30traces/all/simple-mean"}, indent=2))


if __name__ == "__main__":
    main()
