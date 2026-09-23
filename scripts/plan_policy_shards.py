r"""Assign policies to GPUs so data-parallel shards finish together.

Tensor parallelism is unavailable: the steering hook reads request IDs and
absolute token positions from the V1 model runner's batch state, which is not
meaningful once a model is split across devices, and it refuses
`tensor_parallel_size > 1` rather than steer the wrong rows. So scale-out is N
independent single-GPU engines split BY POLICY, which keeps each shard's
per-chunk concurrency at the full cohort size where throughput measured best
(2,220 tok/s at 200 concurrent vs 1,940 at 64).

The unit assigned to a shard is a **cell** -- the N policy and the D policy at
the same (alpha, delay) -- never a single policy. Packing policies individually
put every N on two GPUs and every D on the other two, because N and D have
identical modelled cost and the greedy alternated them. That confounds direction
with device on the one contrast the experiment exists to make. Keeping a cell
together puts each N-vs-D comparison on a single GPU.

**Replicated policies** run on every shard. The baseline is always replicated;
`--replicate` adds steered policies, because agreement on an untreated baseline
says nothing about whether the *hook* behaves identically across devices. These
are reproducibility diagnostics, not extra independent samples -- see
`merge_policy_shards.py`, which declares one canonical shard for analysis.

Shards are balanced by expected token volume, not policy count: stronger steering
finishes sooner, so an equal-count split leaves the weak-alpha shard running
alone. The model only has to be monotone in the right directions to order the bin
packing. Pass `--from-run` to substitute measured means. The resulting imbalance
figure is a scheduling ESTIMATE; real completion times are reported by the
launcher and should be checked against it.

    uv run python scripts/plan_policy_shards.py --shards 4 --stage 1
    uv run python scripts/plan_policy_shards.py --shards 4 --stage 2 \
        --policies "N@a0.5d256 N@a1.0d512 ..." --controls brevityA brevityB
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


def default_cells() -> list[tuple[str, ...]]:
    """Stage-1 grid, as matched (N, D) pairs at each operating point."""
    return [
        (f"N@a{a}d{dl}", f"D@a{a}d{dl}")
        for dl in (0, 256, 512, 1024)
        for a in (0.1, 0.25, 0.5, 1.0)
    ]


def cells_from(policies: list[str]) -> list[tuple[str, ...]]:
    """Group an explicit policy list into cells that keep both directions together.

    Stage 1 sweeps a full grid, so every N has a D at the same operating point and
    a cell is that matched pair. A stage-2 shortlist has no such symmetry: N and D
    are selected independently and generally land on DIFFERENT alphas and delays,
    so grouping by operating point yields all singletons and the packer is free to
    put every N on one device.

    Where an exact partner exists it is used. The rest are paired ACROSS
    directions by cost rank -- cheapest N with cheapest D, and so on -- which is
    not a matched comparison but does guarantee each device carries both
    directions, so no device effect can align with the direction contrast. Any
    odd one out forms a cell of its own rather than being dropped.
    """
    by_point: dict[str, list[str]] = collections.defaultdict(list)
    other: list[tuple[str, ...]] = []
    for p in policies:
        direction, _, point = p.partition("@")
        if point and direction in ("N", "D"):
            by_point[point].append(p)
        else:
            other.append((p,))

    cells = [tuple(sorted(v)) for v in by_point.values() if len(v) > 1]
    left = {
        d: sorted((v[0] for v in by_point.values() if len(v) == 1 and v[0][0] == d),
                  key=modelled_cost)
        for d in ("N", "D")
    }
    for n, d in zip(left["N"], left["D"], strict=False):
        cells.append((n, d))
    paired = {p for cell in cells for p in cell}
    cells += [(p,) for v in left.values() for p in v if p not in paired]
    return cells + other


def modelled_cost(policy: str) -> float:
    if "@" not in policy:
        return BASE_TOKENS
    _, rest = policy.split("@", 1)
    alpha = float(rest.split("d")[0][1:])
    delay = int(rest.split("d")[1])
    return BASE_TOKENS * (1 - SAVING_AT_FULL * alpha * max(0.0, 1 - delay / DELAY_REF))


def measured_costs(run: Path) -> dict[str, float]:
    for name in ("rollouts.csv", "rollouts.csv.partial"):
        path = run / name
        if path.exists():
            break
    else:
        return {}
    tok: dict[str, list[float]] = collections.defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            tok[r["policy"]].append(float(r["total_tokens"]))
    return {p: st.mean(v) for p, v in tok.items() if len(v) >= 20}


def plan(
    units: list[tuple[str, ...]], shards: int, cost: dict[str, float]
) -> list[list[str]]:
    """Longest-processing-time-first over cells: 4/3-optimal greedy for makespan."""
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
    ap.add_argument("--stage", type=int, required=True, choices=(1, 2))
    ap.add_argument("--policies", default="", help="stage 2: the shortlisted policies")
    ap.add_argument("--controls", nargs="*", default=[], help="stage 2: prompt controls")
    ap.add_argument("--replicate", nargs="*", default=None,
                    help="policies run on EVERY shard; base is always included")
    ap.add_argument("--no-replicate", action="store_true",
                    help="run every policy exactly once, baseline included. For "
                         "evaluation runs where cross-device replicas are not wanted: "
                         "they cost a full extra baseline per shard.")
    ap.add_argument("--from-run", type=Path, nargs="+", default=None,
                    help="one or more run dirs whose measured mean tokens replace the model")
    ap.add_argument("--no-pair", action="store_true",
                    help="pack every policy on its own instead of keeping N and D together. "
                         "Pairing only guards against device effects on the N/D contrast; "
                         "use this when those are declared negligible (2026-09-23).")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--questions", type=int, default=None)
    args = ap.parse_args()

    if args.stage == 1:
        units = default_cells()
        questions = args.questions or 200
    else:
        listed = args.policies.split()
        if not listed:
            raise SystemExit("stage 2 needs --policies (the stage-1 shortlist)")
        units = (
            [(p,) for p in listed + list(args.controls)]
            if args.no_pair
            else cells_from(listed + list(args.controls))
        )
        questions = args.questions or 400

    steered = [p for cell in units for p in cell]
    if len(set(steered)) != len(steered):
        raise SystemExit(f"duplicate policies: {sorted(steered)}")

    # Replicated policies are removed from the packing -- they run everywhere.
    # With --no-replicate the baseline is packed like any other policy instead.
    if args.no_replicate:
        units = [*units, ("base",)]
        steered.append("base")
    replicate = [] if args.no_replicate else ["base"] + list(args.replicate or [])
    if args.replicate is None and args.stage == 1 and not args.no_replicate:
        # A steered replica is what actually tests the HOOK across devices;
        # an untreated baseline only tests the engine. Pick the strongest
        # intervention present, where a device difference would show up first.
        strongest = min(steered, key=modelled_cost, default=None)
        if strongest:
            replicate.append(strongest)
    replicate = list(dict.fromkeys(replicate))
    units = [tuple(p for p in cell if p not in replicate) for cell in units]
    units = [c for c in units if c]

    measured: dict[str, float] = {}
    for run in args.from_run or []:
        measured.update(measured_costs(run))
    cost = {p: measured.get(p, modelled_cost(p)) for p in steered + replicate}
    if measured:
        print(f"using {len(set(measured) & set(cost))} measured means from "
              f"{args.from_run}, modelled for the rest\n")

    bins = plan(units, args.shards, cost)
    shards = [[*replicate, *b] for b in bins]
    loads = [sum(cost[p] for p in s) for s in shards]
    for i, (s, load) in enumerate(zip(shards, loads, strict=True)):
        print(f"shard {i}: {len(s):2d} policies, ~{load * questions / 1e6:.2f}M tokens")
        print(f"  {' '.join(s)}")
    print(f"\nload imbalance (max/min): {max(loads) / min(loads):.3f}  "
          f"[ESTIMATE -- verify against measured shard wall times]")
    if replicate:
        print(f"replicated on every shard: {' '.join(replicate)}  "
              f"({100 * (args.shards - 1) * sum(cost[p] for p in replicate) / sum(loads):.0f}% "
              f"overhead, spent on cross-device diagnostics)")
    else:
        print("no replicated policies: every policy, baseline included, runs once")

    # A device effect must not align with the direction contrast. Perfect balance
    # is impossible with an odd shortlist, so the requirement is that no shard
    # carries one direction alone while another is available to it.
    worst = 0
    for i, s in enumerate(shards):
        n = sum(1 for p in s if p.startswith("N@"))
        d = sum(1 for p in s if p.startswith("D@"))
        worst = max(worst, abs(n - d))
        if (n == 0) != (d == 0) and min(n, d) == 0 and max(n, d) > 1:
            raise SystemExit(
                f"shard {i} has {n} N and {d} D: direction is confounded with device"
            )
    print(f"direction balance: max |N-D| per shard is {worst}")

    out = args.out or Path(f"data/policy/shards_stage{args.stage}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "stage": args.stage, "shards": shards, "replicated": replicate,
        "questions": questions,
        "cost_source": [str(r) for r in args.from_run] if measured else "modelled",
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
