#!/usr/bin/env python3
"""Score Tier 3 step 1 — the confirmatory run on 789 untested hard questions.

Primary endpoint, pre-specified: accuracy change under suppression, pooled across
all questions with a Cochran-Mantel-Haenszel test stratified by difficulty band.
CMH is used rather than a naive pooled McNemar because the bands have genuinely
different effects (the pilot put `mostly_wrong` near zero and `mixed` highest),
and pooling unstratified would let a null stratum dilute a real one.

Secondary: per-band McNemar with 95% CIs, and the length sign test.

    uv run python scripts/tier3_step1_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

BANDS = ("all_wrong", "mostly_wrong", "mixed")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", type=Path, default=Path("data/tier3/step1_A.csv"))
    p.add_argument("--traces", type=Path, default=Path("data/hard832.jsonl"))
    return p.parse_args()


def mcnemar(pairs: list[tuple[int, int]]) -> tuple[int, int, float, float, float, float]:
    b = sum(1 for x, y in pairs if x and not y)
    c = sum(1 for x, y in pairs if y and not x)
    n = len(pairs) or 1
    d = (c - b) / n
    se = math.sqrt(b + c) / n
    chi = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0.0
    p = math.erfc(math.sqrt(chi / 2)) if chi > 0 else 1.0
    return b, c, 100 * d, 100 * (d - 1.96 * se), 100 * (d + 1.96 * se), p


def main() -> None:
    args = parse_args()
    if not args.csv.exists():
        part = Path(str(args.csv) + ".partial")
        if part.exists():
            print(f"note: {args.csv} not finished; scoring {part.name} (partial)\n")
            args.csv = part
        else:
            raise SystemExit(f"{args.csv} missing — run scripts/run_tier3_step1.sh")
    band = {json.loads(line)["question_id"]: json.loads(line)["band"]
            for line in args.traces.open()}
    rows = list(csv.DictReader(args.csv.open()))
    n = len(rows)
    print(f"Tier 3 step 1 — vector A on {n} untested hard questions\n")

    print(f"{'condition':<24}{'accuracy':>10}{'med think':>11}{'doubt blk':>11}"
          f"{'answered':>10}{'capped':>8}")
    print("-" * 74)
    for cond, lab in (("none", "baseline (no push)"), ("suppress", "suppress (vector A)")):
        acc = sum(int(r[f"{cond}_correct"]) for r in rows)
        th = st.median(int(r[f"{cond}_think_tokens"]) for r in rows)
        db = st.mean(int(r[f"{cond}_doubt_blocks"]) for r in rows)
        an = sum(int(r[f"{cond}_has_answer"]) for r in rows)
        cp = sum(int(r[f"{cond}_capped"]) for r in rows)
        print(f"{lab:<24}{100*acc/n:>9.1f}%{th:>11.0f}{db:>11.2f}"
              f"{100*an/n:>9.0f}%{100*cp/n:>7.0f}%")

    # ---- primary endpoint: CMH stratified by band -------------------------
    num = den = 0.0
    print("\nPRIMARY — accuracy, CMH stratified by band")
    print(f"{'band':<14}{'n':>5}{'b':>5}{'c':>5}{'diff':>9}{'95% CI':>19}{'p':>9}")
    for bn in BANDS:
        pr = [(int(r["none_correct"]), int(r["suppress_correct"]))
              for r in rows if band.get(r["question_id"]) == bn]
        if not pr:
            continue
        b, c, d, lo, hi, p = mcnemar(pr)
        num += (c - b) / 2.0
        den += (b + c) / 4.0
        print(f"{bn:<14}{len(pr):>5}{b:>5}{c:>5}{d:>+8.1f}pp  [{lo:+5.1f},{hi:+5.1f}]{p:>9.3f}")
    chi_cmh = (abs(num) - 0.5) ** 2 / den if den > 0 else 0.0
    p_cmh = math.erfc(math.sqrt(chi_cmh / 2)) if chi_cmh > 0 else 1.0
    pooled = mcnemar([(int(r["none_correct"]), int(r["suppress_correct"])) for r in rows])
    print(f"\n  CMH chi2 = {chi_cmh:.2f} (1 df), p = {p_cmh:.4g}")
    print(f"  pooled difference {pooled[2]:+.1f}pp, 95% CI [{pooled[3]:+.1f}, {pooled[4]:+.1f}]"
          f"  (b={pooled[0]}, c={pooled[1]})")

    # ---- secondary: length ------------------------------------------------
    sh = sum(1 for r in rows
             if int(r["suppress_think_tokens"]) < int(r["none_think_tokens"]))
    lg = sum(1 for r in rows
             if int(r["suppress_think_tokens"]) > int(r["none_think_tokens"]))
    m = sh + lg
    chi = (abs(sh - lg) - 1) ** 2 / m if m else 0.0
    t0 = st.median(int(r["none_think_tokens"]) for r in rows)
    t1 = st.median(int(r["suppress_think_tokens"]) for r in rows)
    print(f"\nSECONDARY — length: median {t0:.0f} -> {t1:.0f} "
          f"({100*(t1-t0)/t0:+.0f}%), shorter in {sh}/{m}, "
          f"sign test p = {math.erfc(math.sqrt(chi/2)):.3g}")
    print("\nNote: the CI is the number to quote. A p below .05 with a CI reaching")
    print("near zero still means the effect size is poorly pinned down.")


if __name__ == "__main__":
    main()
