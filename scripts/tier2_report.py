#!/usr/bin/env python3
"""Score the Tier 2 arms: does suppression survive on questions the model fails?

Reports accuracy, length and doubt removal overall and **per difficulty band**,
because the hypothesis under test is that self-doubt earns its keep exactly where
the problem is hard. A single pooled average would hide that: the sample is a
third all-wrong and a third nearly-solved, and an effect confined to the former
washes out against the latter.

Paired McNemar throughout, with a 95% CI on the accuracy difference. The CI is the
number to quote, not the p-value: a large p at this sample size means "no effect
we could see", and only the interval says how big an effect could have hidden.

    uv run python scripts/tier2_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

ARMS = {
    "A_1trace": "A: 35 deltas, 1 CORRECT rollout",
    "G_global297": "G: 297 deltas, 297 CORRECT rollouts",
    "W_wrong1": "W: 35 deltas, 1 WRONG rollout",
    "WP_wrong30": "WP: all deltas, 30 WRONG rollouts",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("data/tier2"))
    p.add_argument("--traces", type=Path, default=Path("data/hard_sample_heldout.jsonl"))
    return p.parse_args()


def mcnemar(pairs: list[tuple[int, int]]) -> tuple[int, int, float, float, float]:
    """(b, c, diff, ci_lo, ci_hi) for paired binary outcomes (base, treat)."""
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = len(pairs)
    if n == 0:
        return 0, 0, float("nan"), float("nan"), float("nan")
    diff = (c - b) / n
    se = math.sqrt(b + c) / n
    return b, c, diff, diff - 1.96 * se, diff + 1.96 * se


def main() -> None:
    args = parse_args()
    band = {}
    if args.traces.exists():
        for line in args.traces.open():
            r = json.loads(line)
            band[r["question_id"]] = r.get("band", "?")

    base_path = args.dir / "A_1trace.csv"
    if not base_path.exists():
        raise SystemExit(f"{base_path} missing — run scripts/run_tier2.sh first")
    base = list(csv.DictReader(base_path.open()))
    n = len(base)
    print(f"Tier 2 — {n} questions the model does NOT reliably solve\n")

    def block(rows: list[dict[str, str]], cond: str, label: str) -> None:
        m = len(rows)
        acc = sum(int(r[f"{cond}_correct"]) for r in rows)
        th = sorted(int(r[f"{cond}_think_tokens"]) for r in rows)
        db = st.mean(int(r[f"{cond}_doubt_blocks"]) for r in rows)
        an = sum(int(r[f"{cond}_has_answer"]) for r in rows)
        cp = sum(int(r[f"{cond}_capped"]) for r in rows)
        print(
            f"{label:<30} {100*acc/m:>7.1f}% {th[m//2]:>10} {db:>10.2f} "
            f"{100*an/m:>8.0f}% {100*cp/m:>7.0f}%"
        )

    hdr = (
        f"{'condition':<30} {'acc':>8} {'med think':>10} "
        f"{'doubt blk':>10} {'answered':>9} {'capped':>8}"
    )
    print(hdr)
    print("-" * len(hdr))
    block(base, "none", "baseline (no push)")
    block(base, "random", "random push, matched norm")
    for tag, desc in ARMS.items():
        path = args.dir / f"{tag}.csv"
        if path.exists():
            block(list(csv.DictReader(path.open())), "suppress", desc)

    print("\npaired vs baseline — accuracy")
    for tag, desc in ARMS.items():
        path = args.dir / f"{tag}.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        pairs = [
            (int(x["none_correct"]), int(y["suppress_correct"]))
            for x, y in zip(base, rows, strict=False)
        ]
        b, c, d, lo, hi = mcnemar(pairs)
        print(
            f"  {desc:<30} b={b:>3} c={c:>3}  {100*d:+6.1f}pp  "
            f"95% CI [{100*lo:+5.1f}, {100*hi:+5.1f}]"
        )
    pairs = [(int(r["none_correct"]), int(r["random_correct"])) for r in base]
    b, c, d, lo, hi = mcnemar(pairs)
    print(
        f"  {'random control':<30} b={b:>3} c={c:>3}  {100*d:+6.1f}pp  "
        f"95% CI [{100*lo:+5.1f}, {100*hi:+5.1f}]"
    )

    if not band:
        return
    print("\nBY DIFFICULTY BAND — the comparison this experiment exists for")
    print("(all_wrong = the model never solved it; mixed = it usually does)\n")
    for tag, desc in ARMS.items():
        path = args.dir / f"{tag}.csv"
        if not path.exists():
            continue
        rows = list(csv.DictReader(path.open()))
        print(f"  {desc}")
        for bname in ("all_wrong", "mostly_wrong", "mixed"):
            idx = [i for i, r in enumerate(base) if band.get(r["question_id"]) == bname]
            if not idx:
                continue
            bb = [base[i] for i in idx]
            rr = [rows[i] for i in idx]
            a0 = sum(int(r["none_correct"]) for r in bb) / len(bb)
            a1 = sum(int(r["suppress_correct"]) for r in rr) / len(rr)
            t0 = st.median(int(r["none_think_tokens"]) for r in bb)
            t1 = st.median(int(r["suppress_think_tokens"]) for r in rr)
            b_, c_, d, lo, hi = mcnemar(
                [
                    (int(x["none_correct"]), int(y["suppress_correct"]))
                    for x, y in zip(bb, rr, strict=False)
                ]
            )
            print(
                f"    {bname:<14} n={len(bb):>3}  acc {100*a0:>5.1f}% -> {100*a1:>5.1f}% "
                f"({100*d:+5.1f}pp, CI [{100*lo:+5.1f},{100*hi:+5.1f}])   "
                f"think {t0:>5.0f} -> {t1:<5.0f}"
            )
        print()

    print("How to read this:")
    print("  accuracy holds in every band -> doubt is decorative; distillation is safe.")
    print("  accuracy drops in all_wrong only -> doubt is load-bearing where it is hard,")
    print("     and the goal becomes CONDITIONAL suppression, not blanket suppression.")
    print("  accuracy rises -> the doubt actively hurts. Strongest outcome.")
    print("  Always check `capped`: if suppression caps far less than baseline, part of")
    print("     any gain is budget, not reasoning.")
    print("  W/WP vs A/G: the vectors differ only in whether the traces they were read")
    print("     off were answered correctly. Same numbers -> the doubt direction does not")
    print("     depend on the outcome, and every earlier finding generalises. Different")
    print("     numbers -> it does, and `correct-only derivation` becomes a real caveat.")


if __name__ == "__main__":
    main()
