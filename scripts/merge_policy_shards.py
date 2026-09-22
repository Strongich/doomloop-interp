r"""Merge data-parallel shards, but only after proving they are comparable.

Concatenating four directories is one line. The reason this is a script is
everything that has to be checked first, because a silent merge of incomparable
shards produces a results file that looks perfect and means nothing.

Three gates, all fatal:

1. **Manifest agreement.** Every field that defines the measurement -- model,
   commit, decoding, budget, layer, direction hashes, code hashes, cohort hash,
   engine settings -- must be identical across shards. Only `policies` and the
   derived `max_model_len` may differ, because those are what sharding varies.
2. **Cohort agreement.** All shards must have run the same question IDs.
3. **Baseline homogeneity.** Every shard re-runs the untreated baseline on the
   same questions with the same seeds. This is the check the 10% replication
   overhead buys, and it is what makes the by-policy split defensible rather than
   merely convenient.

   The test is a chi-square test of homogeneity, NOT a fixed percentage-point
   threshold. That distinction matters: four baselines of 200 questions each
   spread about 3.5 pp from sampling noise alone, so any absolute limit near that
   value rejects identical hardware, and a looser one cannot detect a real device
   effect. The question is whether the spread exceeds what sampling explains.

   The per-question identical rate is reported alongside as a descriptive. If the
   engine is deterministic across processes it should be near 100%, and the
   chi-square is then trivially satisfied; if it is low, the replicas are behaving
   as independent samples and the chi-square is doing the real work.

The merged file keeps every baseline replica under the name `base`, so the report
averages them per question. That is legitimate extra precision on the baseline,
not double counting: the replicas are separate rollouts, and each steered policy
is still compared against the same pooled baseline.

    uv run python scripts/merge_policy_shards.py --root data/reasoning_policy_v1 --stage 1
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import statistics as st
from pathlib import Path

def chi2_sf(x: float, df: int) -> float:
    """Upper tail of chi-square via Wilson-Hilferty; ample for a go/no-go gate."""
    if x <= 0 or df <= 0:
        return 1.0
    t = (x / df) ** (1 / 3)
    mean = 1 - 2 / (9 * df)
    sd = math.sqrt(2 / (9 * df))
    z = (t - mean) / sd
    return 0.5 * math.erfc(z / math.sqrt(2))


def homogeneity(counts: dict[str, tuple[int, int]]) -> tuple[float, float]:
    """Chi-square across shards on (correct, total). Returns (chi2, p)."""
    tot_c = sum(c for c, _ in counts.values())
    tot_n = sum(n for _, n in counts.values())
    p = tot_c / tot_n
    chi2 = 0.0
    for c, n in counts.values():
        for obs, exp in ((c, n * p), (n - c, n * (1 - p))):
            if exp > 0:
                chi2 += (obs - exp) ** 2 / exp
    return chi2, chi2_sf(chi2, len(counts) - 1)


# Fields that sharding is allowed to change. Everything else must match exactly.
SHARDABLE = {"policies", "max_model_len", "brevity_prompts", "directions"}


def check_manifests(dirs: list[Path]) -> dict:
    manifests = {d: json.loads((d / "run_manifest.json").read_text()) for d in dirs}
    ref_dir, ref = next(iter(manifests.items()))
    for d, m in manifests.items():
        if d == ref_dir:
            continue
        diff = sorted(
            k for k in set(ref) | set(m) if k not in SHARDABLE and ref.get(k) != m.get(k)
        )
        if diff:
            raise ValueError(
                f"{d.name} and {ref_dir.name} disagree on {diff}. These shards are not "
                f"comparable and must not be merged."
            )
    # Direction hashes must match for directions that BOTH shards used.
    for d, m in manifests.items():
        for name, h in m.get("directions", {}).items():
            if ref.get("directions", {}).get(name, h) != h:
                raise ValueError(f"{d.name}: direction {name} has a different hash")
    return ref


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--stage", type=int, required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--base-alpha", type=float, default=0.001,
                    help="significance level for rejecting baseline homogeneity")
    args = ap.parse_args()

    dirs = sorted(args.root.glob(f"stage{args.stage}_shard*"))
    if len(dirs) < 2:
        raise ValueError(f"Found {len(dirs)} shard directories under {args.root}")
    print(f"merging {len(dirs)} shards: {', '.join(d.name for d in dirs)}\n")

    ref = check_manifests(dirs)
    print("[1] manifests agree on every non-shardable field")

    rows: list[dict] = []
    cohorts = {}
    for d in dirs:
        path = d / "rollouts.csv"
        if not path.exists():
            raise ValueError(f"{d.name} has no completed rollouts.csv")
        with path.open() as f:
            shard_rows = list(csv.DictReader(f))
        for r in shard_rows:
            r["shard"] = d.name.rsplit("shard", 1)[1]
        cohorts[d.name] = {r["question_id"] for r in shard_rows}
        rows.extend(shard_rows)
    sets = list(cohorts.values())
    if any(s != sets[0] for s in sets):
        raise ValueError(f"Shards ran different question sets: "
                         f"{ {k: len(v) for k, v in cohorts.items()} }")
    print(f"[2] all shards ran the same {len(sets[0])} questions")

    # Baseline replicas.
    base = [r for r in rows if r["policy"] == "base"]
    by_shard = collections.defaultdict(list)
    for r in base:
        by_shard[r["shard"]].append(r)
    accs = {s: 100 * st.mean(float(r["correct"]) for r in v) for s, v in by_shard.items()}
    toks = {s: st.mean(float(r["total_tokens"]) for r in v) for s, v in by_shard.items()}
    print(f"[3] baseline replicas on {len(by_shard)} devices:")
    for s in sorted(by_shard):
        print(f"      shard {s}: n={len(by_shard[s]):3d}  acc={accs[s]:5.2f}%  "
              f"mean tokens={toks[s]:6.0f}")
    gap = max(accs.values()) - min(accs.values())
    counts = {
        s: (sum(int(float(r["correct"])) for r in v), len(v)) for s, v in by_shard.items()
    }
    chi2, pval = homogeneity(counts)
    # Per-question agreement is the sharper test: identical devices should give
    # identical answers, not merely the same average.
    per_q = collections.defaultdict(dict)
    for r in base:
        per_q[r["question_id"]][r["shard"]] = r["correct"]
    full = [v for v in per_q.values() if len(v) == len(by_shard)]
    identical = sum(1 for v in full if len(set(v.values())) == 1)
    print(f"      accuracy spread {gap:.2f} pp; "
          f"per-question identical on {identical}/{len(full)} "
          f"({100 * identical / len(full):.1f}%)")
    print(f"      homogeneity chi2={chi2:.2f} (df={len(by_shard) - 1}), p={pval:.3f}")
    if pval < args.base_alpha:
        raise ValueError(
            f"Baseline replicas are not homogeneous (chi2={chi2:.2f}, p={pval:.4g} < "
            f"{args.base_alpha}). The spread of {gap:.2f} pp exceeds sampling noise, so the "
            f"devices are not interchangeable and a by-policy split confounds direction "
            f"with hardware. Re-split by question or investigate before merging."
        )
    print(f"      consistent with one process (p >= {args.base_alpha}) -- "
          f"devices interchangeable")

    out = args.out or args.root / f"stage{args.stage}" / "rollouts.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = [k for k in rows[0] if k != "shard"] + ["shard"]
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (out.parent / "merge_manifest.json").write_text(
        json.dumps(
            {"shards": [d.name for d in dirs], "n_rows": len(rows),
             "baseline_accuracy_by_shard": accs, "baseline_accuracy_spread_pp": gap,
             "baseline_per_question_identical": f"{identical}/{len(full)}",
             "shared": {k: v for k, v in ref.items() if k not in SHARDABLE}},
            indent=1,
        )
    )
    n_pol = len({r["policy"] for r in rows})
    print(f"\nwrote {out}: {len(rows)} rollouts, {n_pol} distinct policies")
    print(f"      (baseline appears {len(by_shard)}x by design; the report pools it)")


if __name__ == "__main__":
    main()
