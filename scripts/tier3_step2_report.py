#!/usr/bin/env python3
"""Score Tier 3 step 2: k rollouts per question on the held-out correct-only set.

Finding 5 measured each question ONCE, so every disagreement between conditions
mixed the intervention with a coin flip at T=0.6. Only 7 of 120 pairs disagreed
and the McNemar p was 1.00 -- uninformative. Averaging k rollouts per question
turns each question into a RATE rather than a bit, which is both more stable and
a stronger test.

Reported:
  * per-question accuracy rate under each condition, averaged over k seeds
  * paired mean difference with a bootstrap 95% CI (the number to quote)
  * a sign test over questions whose rate actually moved
  * median thinking tokens, pooled over seeds

Note this set CANNOT confirm a gain: most of its questions are solved every time,
so accuracy has no headroom. A tight interval around zero is the result.

    uv run python scripts/tier3_step2_report.py
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import statistics as st
from pathlib import Path

SEED0 = {"A": Path("data/pool/arms/A_1trace.csv"), "G": Path("data/pool/arms/G_global297.csv")}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("data/tier3"))
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    return p.parse_args()


def load(path: Path) -> list[dict[str, str]]:
    return list(csv.DictReader(path.open())) if path.exists() else []


def boot_ci(d: list[float], n: int = 20000, seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    m = len(d)
    means = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n))
    return means[int(0.025 * n)], means[int(0.975 * n)]


def main() -> None:
    args = parse_args()
    base0 = load(SEED0["A"])
    if not base0:
        raise SystemExit(f"{SEED0['A']} missing")
    ids = [r["question_id"] for r in base0]

    # accumulate per-question hit counts and think tokens
    none_hits: dict[str, list[int]] = {q: [] for q in ids}
    sup: dict[str, dict[str, list[int]]] = {v: {q: [] for q in ids} for v in ("A", "G")}
    think: dict[str, list[int]] = {"none": [], "A": [], "G": []}

    for r in base0:                                   # seed 0: baseline + A
        none_hits[r["question_id"]].append(int(r["none_correct"]))
        sup["A"][r["question_id"]].append(int(r["suppress_correct"]))
        think["none"].append(int(r["none_think_tokens"]))
        think["A"].append(int(r["suppress_think_tokens"]))
    for r in load(SEED0["G"]):                        # seed 0: G (suppress only)
        sup["G"][r["question_id"]].append(int(r["suppress_correct"]))
        think["G"].append(int(r["suppress_think_tokens"]))

    used = [0]
    for s in args.seeds:
        a = load(args.dir / f"step2_A_seed{s}.csv")
        g = load(args.dir / f"step2_G_seed{s}.csv")
        if not a:
            continue
        used.append(s)
        for r in a:
            none_hits[r["question_id"]].append(int(r["none_correct"]))
            sup["A"][r["question_id"]].append(int(r["suppress_correct"]))
            think["none"].append(int(r["none_think_tokens"]))
            think["A"].append(int(r["suppress_think_tokens"]))
        for r in g:
            sup["G"][r["question_id"]].append(int(r["suppress_correct"]))
            think["G"].append(int(r["suppress_think_tokens"]))

    k = len(used)
    print(f"Tier 3 step 2 — {len(ids)} held-out questions the model solves, "
          f"k={k} rollouts each (seeds {used})\n")

    def rate(d: dict[str, list[int]]) -> dict[str, float]:
        return {q: (sum(v) / len(v) if v else float("nan")) for q, v in d.items()}

    r_none = rate(none_hits)
    print(f"{'condition':<26}{'accuracy':>10}{'med think':>11}{'n rollouts':>12}")
    print("-" * 59)
    print(f"{'baseline (no push)':<26}{100*st.mean(r_none.values()):>9.1f}%"
          f"{st.median(think['none']):>11.0f}{len(think['none']):>12}")
    for v in ("A", "G"):
        rv = rate(sup[v])
        if not think[v]:
            continue
        lab = f"suppress {v}"
        print(f"{lab:<26}{100*st.mean(rv.values()):>9.1f}%"
              f"{st.median(think[v]):>11.0f}{len(think[v]):>12}")

    print("\npaired vs baseline — per-question accuracy rate")
    for v in ("A", "G"):
        rv = rate(sup[v])
        if not think[v]:
            continue
        d = [rv[q] - r_none[q] for q in ids if not math.isnan(rv[q])]
        lo, hi = boot_ci(d)
        moved = [x for x in d if x != 0]
        up = sum(1 for x in moved if x > 0)
        m = len(moved)
        chi = (abs(up - (m - up)) - 1) ** 2 / m if m else 0.0
        p = math.erfc(math.sqrt(chi / 2)) if chi > 0 else 1.0
        print(f"  suppress {v}:  {100*st.mean(d):+5.2f}pp   95% CI "
              f"[{100*lo:+5.2f}, {100*hi:+5.2f}]   "
              f"{up}/{m} questions improved, sign test p={p:.3f}")

    print("\nHow to read this:")
    print("  The CI is the result, not the p-value. A tight interval containing zero")
    print("     means 'no accuracy cost larger than the interval', which is the claim")
    print("     a length-reduction method needs to support.")
    print("  A gain cannot show up here: most questions are solved on every rollout.")


if __name__ == "__main__":
    main()
