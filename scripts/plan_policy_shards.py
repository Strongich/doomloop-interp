r"""Assign policies to GPUs so four data-parallel shards finish together.

Tensor parallelism is unavailable: the steering hook reads request IDs and
absolute token positions from the V1 model runner's batch state, which is not
meaningful once a model is split across devices, and it refuses
`tensor_parallel_size > 1` rather than steer the wrong rows. So scale-out is four
independent single-GPU engines, split BY POLICY -- which keeps each shard's
per-chunk concurrency at the full cohort size, where throughput was measured
best (2,220 tok/s at 200 vs 1,940 at 64).

Splitting by policy puts the baseline on one GPU and the steered arms on others,
so a systematic per-device difference would fall entirely inside the comparison.
Rather than assume identical A100s, **every shard also runs the baseline**. The
replicas cost ~9% overhead and turn that assumption into a measurement:
`merge_policy_shards.py` refuses to merge if they disagree.

The unit assigned to a shard is a **cell** -- the N policy and the D policy at the
same (alpha, delay) -- never a single policy. Packing policies individually put
every N on two GPUs and every D on the other two, because N and D have identical
modelled cost and the greedy alternated them. That confounds direction with
device on the one contrast the experiment exists to make. Keeping a cell together
puts each N-vs-D comparison on a single GPU, so no device difference can enter
it. The cell count divides evenly (16 cells over 4 shards), so nothing is lost.

Shards are balanced by expected token volume, not policy count. Stronger steering
finishes sooner, so an equal-count split would leave the weak-alpha shard running
long after the others idle. Cost is modelled as

    tokens ~ base_tokens * (1 - SAVING_AT_FULL * alpha * (1 - delay / DELAY_REF))

which only has to be monotone in the right directions to order the bin packing;
longest-processing-time-first is insensitive to the constant. Pass `--from-run`
to replace the model with measured means once any run has produced them.

    uv run python scripts/plan_policy_shards.py --shards 4
    uv run python scripts/plan_policy_shards.py --shards 4 --from-run data/reasoning_policy_v1/stage1
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path

SAVING_AT_FULL = 0.55
DELAY_REF = 2048.0
BASE_TOKENS = 4478.0


def cells() -> list[tuple[str, ...]]:
    """Each cell is the matched (N, D) pair at one operating point."""
    return [
        (f"N@a{a}d{dl}", f"D@a{a}d{dl}")
        for dl in (0, 256, 512, 1024)
        for a in (0.1, 0.25, 0.5, 1.0)
    ]


def modelled_cost(policy: str) -> float:
    if policy == "base" or "@" not in policy:
        return BASE_TOKENS
    _, rest = policy.split("@")
    alpha = float(rest.split("d")[0][1:])
    delay = int(rest.split("d")[1])
    return BASE_TOKENS * (1 - SAVING_AT_FULL * alpha * max(0.0, 1 - delay / DELAY_REF))


def measured_costs(run: Path) -> dict[str, float]:
    path = run / "rollouts.csv"
    if not path.exists():
        path = run / "rollouts.csv.partial"
    if not path.exists():
        return {}
    tok: dict[str, list[float]] = collections.defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            tok[r["policy"]].append(float(r["total_tokens"]))
    return {p: st.mean(v) for p, v in tok.items() if len(v) >= 20}


def plan(
    units: list[tuple[str, ...]], shards: int, cost: dict[str, float]
) -> list[list[str]]:
    """Longest-processing-time-first over CELLS: 4/3-optimal greedy for makespan."""
    bins: list[list[str]] = [[] for _ in range(shards)]
    load = [0.0] * shards
    for cell in sorted(units, key=lambda c: -sum(cost[p] for p in c)):
        i = min(range(shards), key=lambda i: load[i])
        bins[i].extend(cell)
        load[i] += sum(cost[p] for p in cell)
    return bins


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=4)
    ap.add_argument("--from-run", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("data/policy/shards.json"))
    ap.add_argument("--questions", type=int, default=200)
    args = ap.parse_args()

    units = cells()
    steered = [p for cell in units for p in cell]
    measured = measured_costs(args.from_run) if args.from_run else {}
    cost = {p: measured.get(p, modelled_cost(p)) for p in steered}
    if measured:
        print(f"using {len(set(measured) & set(steered))} measured means "
              f"from {args.from_run}, modelled for the rest\n")

    bins = plan(units, args.shards, cost)
    total = 0.0
    for i, b in enumerate(bins):
        # The baseline is replicated onto every shard as the cross-device control.
        load = sum(cost[p] for p in b) + BASE_TOKENS
        total += load
        print(f"shard {i}: {len(b) + 1:2d} policies, ~{load * args.questions / 1e6:.2f}M tokens")
        print(f"  base {' '.join(b)}")
    spread = max(
        sum(cost[p] for p in b) + BASE_TOKENS for b in bins
    ) / min(sum(cost[p] for p in b) + BASE_TOKENS for b in bins)
    print(f"\nload imbalance (max/min): {spread:.3f}   "
          f"total ~{total * args.questions / 1e6:.1f}M tokens")
    print(f"baseline replicas: {args.shards} "
          f"({100 * (args.shards - 1) * BASE_TOKENS / total:.0f}% overhead, "
          f"spent to measure cross-device agreement)")
    for i, b in enumerate(bins):
        n = sum(1 for p in b if p.startswith("N@"))
        d = sum(1 for p in b if p.startswith("D@"))
        if n != d:
            raise ValueError(f"shard {i} has {n} N and {d} D: direction is confounded with device")
    print("direction balance: every shard holds matched N/D pairs")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"shards": [["base", *b] for b in bins], "cost_source":
             str(args.from_run) if measured else "modelled"},
            indent=1,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
