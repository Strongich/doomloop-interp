#!/usr/bin/env python3
"""Score the delay x alpha sweep, and apply the pre-registered shortlist rule.

Every statistic is clustered by QUESTION: seeds are averaged within a question
first, and the question is the unit of analysis. Seeds reduce within-question
sampling noise; they do not increase n. A per-rollout interval over 400x2 rows
would be roughly sqrt(2) too narrow.

Comparisons are PAIRED against the shared baseline on the questions both arms
completed, so a config that is missing rows is compared fairly rather than
flattered by a different question mix.

`--shortlist` applies the rule fixed in EXPERIMENT-reasoning-policy.md before any
generation ran:

  1. Pareto frontier over (higher accuracy delta, lower mean total tokens).
  2. Frontier points with delta >= -5 pp, cheapest first, up to 3. The gate is
     looser than the -2 pp selection margin on purpose: one seed on 200
     questions cannot resolve 2 points, and a config cut here never returns.
  3. Plus the weakest intervention still saving >=10% tokens (smallest alpha,
     then longest delay), even if dominated -- otherwise screening can only ever
     nominate strong steering.
  4. Fill to 4 by highest accuracy delta.

    uv run python scripts/report_reasoning_policy.py --run data/reasoning_policy_v1/stage1
    uv run python scripts/report_reasoning_policy.py --run ... --shortlist
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import math
from pathlib import Path


def mean(xs: list[float]) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def paired(
    per_q: dict[str, dict[str, float]], arm: str, base: str
) -> tuple[float, float, float, int]:
    """Mean paired difference, half-width, two-sided p, n -- over shared questions."""
    diffs = [
        per_q[q][arm] - per_q[q][base]
        for q in per_q
        if arm in per_q[q] and base in per_q[q]
    ]
    n = len(diffs)
    if n < 2:
        return float("nan"), float("nan"), float("nan"), n
    m = mean(diffs)
    var = sum((d - m) ** 2 for d in diffs) / (n - 1)
    se = math.sqrt(var / n)
    if se == 0:
        return m, 0.0, 1.0, n
    return m, 1.96 * se, math.erfc(abs(m / se) / math.sqrt(2)), n


def load(run: Path) -> list[dict]:
    rows = list(csv.DictReader((run / "rollouts.csv").open()))
    if not rows:
        raise ValueError(f"No rows in {run}/rollouts.csv")
    return rows


def collapse(rows: list[dict], field: str) -> dict[str, dict[str, float]]:
    """question -> policy -> value, averaged over seeds."""
    acc: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    for r in rows:
        acc[r["question_id"]][r["policy"]].append(float(r[field]))
    return {q: {p: mean(v) for p, v in per.items()} for q, per in acc.items()}


def shortlist(table: list[dict], direction: str) -> list[str]:
    cands = [t for t in table if t["direction"] == direction]
    if not cands:
        return []
    # 1. Pareto frontier: not dominated on both accuracy and cost.
    front = [
        c
        for c in cands
        if not any(
            o is not c and o["dacc"] >= c["dacc"] and o["tok"] <= c["tok"]
            and (o["dacc"] > c["dacc"] or o["tok"] < c["tok"])
            for o in cands
        )
    ]
    picked: list[dict] = []
    for c in sorted(front, key=lambda c: c["tok"]):
        if c["dacc"] >= -5.0 and len(picked) < 3:
            picked.append(c)
    # 3. Keep one low-dose arm alive.
    base_tok = cands[0]["base_tok"]
    weak = [c for c in cands if c["tok"] <= 0.90 * base_tok and c not in picked]
    if weak:
        weak.sort(key=lambda c: (c["alpha"], -c["delay"]))
        picked.append(weak[0])
    # 4. Fill by accuracy.
    for c in sorted(cands, key=lambda c: -c["dacc"]):
        if len(picked) >= 4:
            break
        if c not in picked:
            picked.append(c)
    return [c["policy"] for c in picked[:4]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--base", default="base")
    ap.add_argument("--shortlist", action="store_true")
    ap.add_argument("--shortlist-out", type=Path, default=None,
                    help="write the nominations as JSON, for an unattended chain")
    ap.add_argument("--gate-min-base-acc", type=float, default=35.0)
    ap.add_argument("--gate-max-base-acc", type=float, default=70.0)
    ap.add_argument("--gate-max-capped", type=float, default=5.0)
    args = ap.parse_args()

    rows = load(args.run)
    acc_q = collapse(rows, "correct")
    tok_q = collapse(rows, "total_tokens")
    think_q = collapse(rows, "think_tokens")
    policies = sorted({r["policy"] for r in rows})
    if args.base not in policies:
        raise ValueError(f"Baseline {args.base!r} not present; have {policies}")

    meta = {r["policy"]: r for r in rows}
    n_seeds = len({r["seed"] for r in rows})
    n_q = len(acc_q)
    print(f"{args.run}: {len(rows)} rollouts, {n_q} questions, {n_seeds} seed(s), "
          f"{len(policies)} policies")
    capped = sum(int(r["capped"]) for r in rows)
    unans = sum(1 for r in rows if not int(r["has_answer"]))
    print(f"capped {capped}/{len(rows)} ({100 * capped / len(rows):.2f}%), "
          f"unanswered {unans} ({100 * unans / len(rows):.2f}%)")
    print()

    base_tok = mean([tok_q[q][args.base] for q in tok_q if args.base in tok_q[q]])
    base_acc = mean([acc_q[q][args.base] for q in acc_q if args.base in acc_q[q]])
    print(f"{'policy':16s}{'n':>5s}{'acc':>7s}{'dacc':>7s}{'95% CI':>17s}{'p':>7s}"
          f"{'tokens':>9s}{'save':>7s}{'think':>8s}{'inj':>6s}")
    table = []
    for p in policies:
        d_acc, hw, pv, n = paired(acc_q, p, args.base)
        tok = mean([tok_q[q][p] for q in tok_q if p in tok_q[q]])
        think = mean([think_q[q][p] for q in think_q if p in think_q[q]])
        a = mean([acc_q[q][p] for q in acc_q if p in acc_q[q]])
        inj = mean([float(r["injections"]) for r in rows if r["policy"] == p])
        save = 100 * (1 - tok / base_tok) if base_tok else 0.0
        ci = "" if p == args.base else f"[{100*(d_acc-hw):+.1f},{100*(d_acc+hw):+.1f}]"
        print(f"{p:16s}{n:5d}{100*a:7.1f}{100*d_acc:+7.1f}{ci:>17s}"
              f"{pv:7.3f}{tok:9.0f}{save:+7.1f}{think:8.0f}{inj:6.1f}")
        if p != args.base:
            table.append({
                "policy": p, "direction": meta[p]["direction"],
                "alpha": float(meta[p]["alpha"]), "delay": int(meta[p]["delay"]),
                "dacc": 100 * d_acc, "tok": tok, "base_tok": base_tok,
            })
    print(f"\nbaseline: {100*base_acc:.1f}% accuracy, {base_tok:.0f} mean total tokens")

    if args.shortlist:
        print("\n--- pre-registered shortlist (nomination only, not selection) ---")
        for d in ("N", "D"):
            picked = shortlist(table, d)
            if picked:
                print(f"{d}: {' '.join(picked)}")
        print("\nStage 2:  STAGE=2 SHORTLIST=\"<the 8 above>\" "
              "bash scripts/run_reasoning_policy_sweep.sh")

        if args.shortlist_out:
            # Sanity gates for an UNATTENDED chain. The pre-registration makes
            # stage 2 a human checkpoint so a degenerate screen cannot promote
            # itself; when nobody is watching, these stand in for that judgement.
            # They are deliberately loose -- they catch a broken run (grader
            # failing, engine truncating everything, injection not firing), not a
            # disappointing one. A disappointing screen is a result; a broken one
            # must not spend another two GPU-hours.
            picks = {d: shortlist(table, d) for d in ("N", "D")}
            pct_capped = 100 * capped / len(rows)
            problems = []
            if not (args.gate_min_base_acc <= 100 * base_acc <= args.gate_max_base_acc):
                problems.append(
                    f"baseline accuracy {100 * base_acc:.1f}% outside "
                    f"[{args.gate_min_base_acc}, {args.gate_max_base_acc}] -- "
                    f"suspect grading or generation, not policy quality"
                )
            if pct_capped > args.gate_max_capped:
                problems.append(f"capped {pct_capped:.1f}% > {args.gate_max_capped}%")
            for d in ("N", "D"):
                if len(picks[d]) < 2:
                    problems.append(f"only {len(picks[d])} {d} nomination(s)")
            no_inject = [
                p for p in policies
                if p != args.base
                and mean([float(r["injections"]) for r in rows if r["policy"] == p]) == 0
            ]
            if no_inject:
                problems.append(f"steered policies with zero injections: {no_inject[:3]}")

            args.shortlist_out.write_text(json.dumps({
                "shortlist": [p for d in ("N", "D") for p in picks[d]],
                "by_direction": picks, "baseline_accuracy": 100 * base_acc,
                "pct_capped": pct_capped, "problems": problems,
                "ok": not problems,
            }, indent=1))
            if problems:
                print("\nGATE FAILED -- not proceeding unattended:")
                for x in problems:
                    print(f"  - {x}")
            else:
                print(f"\ngates passed; wrote {args.shortlist_out}")


if __name__ == "__main__":
    main()
