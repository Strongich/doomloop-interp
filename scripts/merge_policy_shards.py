r"""Merge data-parallel shards, but only after proving they are comparable.

Concatenating directories is one line. The reason this is a script is everything
that must be checked first, because a silent merge of incomparable or incomplete
shards produces a results file that looks perfect and means nothing.

Four fatal gates:

1. **Exactly the planned shards.** Directories are taken from the shard plan, not
   discovered by wildcard: a stale directory from an earlier configuration would
   otherwise be swept in, and a missing one would pass unnoticed.
2. **Complete coverage.** Every (question_id, policy, seed) the plan and manifest
   promise must appear exactly once per shard. Equal question-ID sets prove
   nothing -- a shard holding only baseline seed 0 has the same question IDs as a
   complete one.
3. **Consistent definitions across ALL shards.** Direction name -> hash and
   brevity name -> text are validated as one global mapping. Comparing each shard
   against only the first lets two later shards disagree with each other whenever
   the first does not use that direction.
4. **Replica reproducibility.** Replicated policies are compared per
   (question, seed) across devices, on correctness AND token counts.

On gate 4, what this can and cannot establish: agreement is evidence the devices
behave alike; **non-significance is not proof that they do**, so no automatic
"interchangeable" verdict is issued. Equal aggregate accuracy can hide offsetting
failures, which is why the comparison is per-question-and-seed and includes
lengths. Replicas are therefore treated as a REPRODUCIBILITY DIAGNOSTIC, never as
extra independent samples, and primary analysis uses one declared canonical
shard's copy. Pooling them would understate the baseline's variance, since
replicas share questions and seeds and are not independent draws.

    uv run python scripts/merge_policy_shards.py --root data/reasoning_policy_v1 \
        --stage 1 --plan data/policy/shards_stage1.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import statistics as st
from pathlib import Path

# Only these may differ between shards; everything else defines the measurement.
SHARDABLE = {"policies", "max_model_len", "directions", "brevity_prompts"}


def check_manifests(dirs: list[Path]) -> tuple[dict, dict, dict]:
    """Returns (reference manifest, direction->hash, brevity->text), or raises."""
    manifests = {d.name: json.loads((d / "run_manifest.json").read_text()) for d in dirs}
    ref_name, ref = next(iter(manifests.items()))
    for name, m in manifests.items():
        diff = sorted(
            k for k in set(ref) | set(m) if k not in SHARDABLE and ref.get(k) != m.get(k)
        )
        if diff:
            raise ValueError(
                f"{name} and {ref_name} disagree on {diff}. Not comparable; refusing to merge."
            )
    # Global mappings, so two shards cannot disagree with each other merely
    # because the reference shard never used that direction or prompt.
    directions: dict[str, str] = {}
    prompts: dict[str, str] = {}
    for name, m in manifests.items():
        for key, store, label in (
            ("directions", directions, "direction"),
            ("brevity_prompts", prompts, "brevity prompt"),
        ):
            for k, v in (m.get(key) or {}).items():
                if store.setdefault(k, v) != v:
                    raise ValueError(
                        f"{name}: {label} {k!r} differs from another shard's definition"
                    )
    return ref, directions, prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--canonical-shard", type=int, default=0,
                    help="whose copy of a replicated policy enters primary analysis")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    plan = json.loads(args.plan.read_text())
    planned: list[list[str]] = plan["shards"]
    replicated = set(plan.get("replicated", ["base"]))
    dirs = [args.root / f"stage{args.stage}_shard{i}" for i in range(len(planned))]
    missing = [d.name for d in dirs if not (d / "rollouts.csv").exists()]
    if missing:
        raise ValueError(f"Incomplete shards (no rollouts.csv): {missing}")
    stale = sorted(
        p.name for p in args.root.glob(f"stage{args.stage}_shard*") if p not in dirs
    )
    if stale:
        raise ValueError(f"Unexpected shard directories not in the plan: {stale}")
    print(f"merging {len(dirs)} planned shards\n")

    ref, directions, prompts = check_manifests(dirs)
    print(f"[1] manifests agree; {len(directions)} direction(s) and "
          f"{len(prompts)} prompt(s) consistent across all shards")

    seeds = int(ref["seeds"])
    questions = set(ref["question_ids"])
    rows: list[dict] = []
    for i, d in enumerate(dirs):
        with (d / "rollouts.csv").open() as f:
            shard_rows = list(csv.DictReader(f))
        expect = {
            (q, p, str(s)) for q in questions for p in planned[i] for s in range(seeds)
        }
        seen = collections.Counter(
            (r["question_id"], r["policy"], r["seed"]) for r in shard_rows
        )
        dupes = [k for k, n in seen.items() if n > 1]
        absent = expect - set(seen)
        extra = set(seen) - expect
        if dupes or absent or extra:
            raise ValueError(
                f"{d.name} coverage is wrong: {len(absent)} missing, {len(extra)} "
                f"unexpected, {len(dupes)} duplicated (of {len(expect)} expected). "
                f"e.g. missing={sorted(absent)[:3]} unexpected={sorted(extra)[:3]}"
            )
        for r in shard_rows:
            r["shard"] = str(i)
        rows.extend(shard_rows)
    print(f"[2] coverage complete: {len(questions)} questions x {seeds} seed(s), "
          f"every (question, policy, seed) exactly once per shard")

    # Gate 4 -- replica reproducibility, keyed by question AND seed.
    print(f"[3] replica diagnostics across {len(dirs)} devices "
          f"(reproducibility only; not extra samples):")
    for pol in sorted(replicated):
        per_key: dict[tuple[str, str], dict[str, dict]] = collections.defaultdict(dict)
        for r in rows:
            if r["policy"] == pol:
                per_key[(r["question_id"], r["seed"])][r["shard"]] = r
        full = [v for v in per_key.values() if len(v) == len(dirs)]
        if not full:
            print(f"      {pol}: not present on every shard, skipped")
            continue
        same_correct = sum(1 for v in full if len({x["correct"] for x in v.values()}) == 1)
        same_tokens = sum(1 for v in full if len({x["total_tokens"] for x in v.values()}) == 1)
        spread = [
            max(float(x["total_tokens"]) for x in v.values())
            - min(float(x["total_tokens"]) for x in v.values())
            for v in full
        ]
        accs = {
            s: 100 * st.mean(float(r["correct"]) for r in rows
                             if r["policy"] == pol and r["shard"] == s)
            for s in sorted({r["shard"] for r in rows if r["policy"] == pol})
        }
        print(f"      {pol}:")
        print(f"        accuracy by shard: "
              f"{', '.join(f'{s}={a:.2f}%' for s, a in accs.items())} "
              f"(spread {max(accs.values()) - min(accs.values()):.2f} pp)")
        print(f"        identical answer: {same_correct}/{len(full)} "
              f"({100 * same_correct / len(full):.1f}%); "
              f"identical length: {same_tokens}/{len(full)} "
              f"({100 * same_tokens / len(full):.1f}%)")
        print(f"        token spread: mean {st.mean(spread):.0f}, max {max(spread):.0f}")

    # Primary analysis keeps ONE copy of each replicated policy.
    canon = str(args.canonical_shard)
    analysis = [r for r in rows if r["policy"] not in replicated or r["shard"] == canon]
    print(f"[4] primary analysis uses shard {canon}'s copy of "
          f"{sorted(replicated)}; other replicas retained in rollouts_all.csv")

    out = args.out or args.root / f"stage{args.stage}" / "rollouts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in rows[0] if k != "shard"] + ["shard"]
    for path, data in ((out, analysis), (out.parent / "rollouts_all.csv", rows)):
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(data)
    (out.parent / "merge_manifest.json").write_text(json.dumps({
        "plan": str(args.plan), "shards": [d.name for d in dirs],
        "canonical_shard": canon, "replicated": sorted(replicated),
        "n_rows_analysis": len(analysis), "n_rows_all": len(rows),
        "directions": directions, "brevity_prompts": prompts,
        "source_manifests": {
            d.name: json.loads((d / "run_manifest.json").read_text()) for d in dirs
        },
        "shared": {k: v for k, v in ref.items() if k not in SHARDABLE},
    }, indent=1))
    print(f"\nwrote {out}: {len(analysis)} rollouts, "
          f"{len({r['policy'] for r in analysis})} distinct policies")


if __name__ == "__main__":
    main()
