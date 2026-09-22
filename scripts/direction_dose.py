#!/usr/bin/env python3
"""How many doubt blocks must be averaged before the direction is good?

Finding 5a showed the rows sorted by `n` rather than by dataset, but those points
are confounded: each came from a different trace, a different dataset and a
different `n` at once. This isolates `n`.

One pool of per-probe Δ's is computed once. Pooled directions are then built at
`n = 1, 2, 4, ...` by sampling from a **build** split, and evaluated on a
**disjoint** eval split — sampling and evaluating on the same probes would let a
large `n` memorise its own test set. Several random draws per `n` give the spread,
which is the whole point: at small `n` the variance across draws is the quantity
of interest, not the mean.

Stage 1 (this script) writes the pooled vectors and needs no generation — only AR
forward passes over explanations that already exist on disk.
Stage 2 evaluates each with `steer_sweep.py --delta file:<path>`.

    uv run python scripts/direction_dose.py --out-dir data/dose
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steer_demo import (  # noqa: E402
    CONTINUE_PARAGRAPH,
    DOUBT_PARAGRAPH,
    edit_explanation,
    load_ar,
    reconstruct,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--explanations", type=Path, default=Path("data/block_explanations_3way.csv"))
    p.add_argument("--ar", type=Path, default=Path("checkpoints/ar_rl_ep1"))
    p.add_argument("--out-dir", type=Path, default=Path("data/dose"))
    p.add_argument("--kind", default="doubt_wait")
    p.add_argument("--eval-dataset", default="gsm8k", help="dataset the eval split is drawn from")
    p.add_argument("--n-eval", type=int, default=60)
    p.add_argument("--sizes", default="1,2,4,8,16,32,64,128,200")
    p.add_argument("--seeds", type=int, default=3, help="draws per size (1 for the largest)")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    probes = [
        r
        for r in csv.DictReader(args.explanations.open())
        if r["kind"] == args.kind and r["explanation"].count("\n\n") >= 1
    ]
    print(f"{len(probes)} {args.kind} probes")

    # The eval split is held out of every build pool. Drawn from one dataset so
    # the curve is not also measuring dataset mix as n grows.
    rng = random.Random(args.seed)
    pool = [r for r in probes if r["dataset"] == args.eval_dataset]
    rng.shuffle(pool)
    eval_ids = {r["question_id"] for r in pool[: args.n_eval]}
    build = [r for r in probes if r["question_id"] not in eval_ids]
    print(f"eval split: {len(eval_ids)} {args.eval_dataset} probes (held out)")
    print(f"build pool: {len(build)} probes")
    (args.out_dir / "eval_ids.json").write_text(json.dumps(sorted(eval_ids)))

    ar_tok, ar_backbone, affine = load_ar(args.ar)
    deltas: list[torch.Tensor] = []
    with torch.no_grad():
        for i, r in enumerate(build, 1):
            e0 = r["explanation"]
            try:
                ed = edit_explanation(e0, CONTINUE_PARAGRAPH)
            except SystemExit:
                continue
            a = reconstruct(ar_tok, ar_backbone, affine, e0)
            deltas.append((reconstruct(ar_tok, ar_backbone, affine, ed) - a).float().cpu())
            if i % 50 == 0:
                print(f"  [{i}/{len(build)}]", flush=True)
    stack = torch.stack(deltas)
    print(f"built {stack.shape[0]} deltas")

    # The opposite-sign direction is shared across sizes: it is only the control
    # arm, and re-estimating it per draw would add noise to the comparison.
    with torch.no_grad():
        dd = []
        for r in build[:64]:
            try:
                ed = edit_explanation(r["explanation"], DOUBT_PARAGRAPH)
            except SystemExit:
                continue
            a = reconstruct(ar_tok, ar_backbone, affine, r["explanation"])
            dd.append((reconstruct(ar_tok, ar_backbone, affine, ed) - a).float().cpu())
    doubt_unit = torch.stack(dd).mean(0)
    doubt_unit = doubt_unit / doubt_unit.norm()

    sizes = [int(x) for x in args.sizes.split(",")]
    index = []
    for n in sizes:
        if n > stack.shape[0]:
            continue
        n_draws = 1 if n >= stack.shape[0] else args.seeds
        for d in range(n_draws):
            idx = random.Random(1000 * n + d).sample(range(stack.shape[0]), n)
            mean = stack[idx].mean(0)
            unit = mean / mean.norm()
            path = args.out_dir / f"dir_n{n:04d}_s{d}.pt"
            torch.save(
                {
                    "continue": {"unit": unit, "stats": {"n": float(n), "draw": float(d)}},
                    "doubt": {"unit": doubt_unit, "stats": {"n": 64.0}},
                    "source": {"n_averaged": n, "draw": d, "dataset": "pooled"},
                },
                path,
            )
            index.append({"n": n, "draw": d, "path": str(path)})
    (args.out_dir / "index.json").write_text(json.dumps(index, indent=2))
    print(f"wrote {len(index)} pooled directions -> {args.out_dir}/index.json")


if __name__ == "__main__":
    main()
