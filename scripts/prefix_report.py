#!/usr/bin/env python3
"""Score the fixed-prefix branching experiment.

Every arm continues the SAME frozen prefix, so the comparison is within-question
and the only thing that differs is the intervention. Uncertainty is clustered by
question: k seeds per prefix measure how much of a flip is sampling noise, and
they do NOT count as k independent questions.

The headline numbers are stratified, because the two effects that matter cancel
in an aggregate:

    candidate CORRECT  ->  `break` rate  = damage the continuation does
    candidate WRONG    ->  `repair` rate = rescue the continuation performs

An intervention that shortens reasoning is only safe if it keeps `break` low
AND keeps `repair` near the baseline's.

    uv run python scripts/prefix_report.py
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

BANDS = ["all_right", "mixed", "mostly_wrong", "all_wrong"]


def med(xs: list[float]) -> float:
    return st.median(xs) if xs else float("nan")


def clustered(diffs: list[float]) -> tuple[float, float, float]:
    """Mean per-question difference, its 95% half-width, and a two-sided p.

    One observation per QUESTION (the mean over its seeds), so seeds reduce the
    noise inside a cluster without inflating n.
    """
    n = len(diffs)
    if n < 2:
        return (float("nan"),) * 3
    m = sum(diffs) / n
    sd = st.stdev(diffs)
    se = sd / math.sqrt(n)
    if se == 0:
        return m, 0.0, 1.0
    z = abs(m / se)
    return m, 1.96 * se, math.erfc(z / math.sqrt(2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", type=Path, default=Path("data/prefix/branches.csv"))
    ap.add_argument("--base", default="base")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="override which seeds to score (default: the complete ones)")
    args = ap.parse_args()

    path = args.csv if args.csv.exists() else Path(str(args.csv) + ".partial")
    rows = list(csv.DictReader(path.open()))
    for r in rows:
        for k in ("seed", "cand_correct", "freeze_at", "correct", "has_answer",
                  "cont_tokens", "cont_think_tokens", "capped", "closed",
                  "boundaries", "injections", "markers", "doubt_blocks"):
            r[k] = int(r[k])
        r["doubt_rate"] = float(r["doubt_rate"])
    arms = sorted({r["arm"] for r in rows}, key=lambda a: (a != args.base, a))
    all_seeds = sorted({r["seed"] for r in rows})
    # A run in progress always has one half-finished seed. Scoring it would
    # silently drop every question that seed has not reached yet, so only seeds
    # that every arm finished are used, and which ones is printed.
    have: dict[tuple[str, int], set[str]] = defaultdict(set)
    for r in rows:
        have[(r["arm"], r["seed"])].add(r["question_id"])
    widest = max((len(v) for v in have.values()), default=0)
    seeds = ([s for s in all_seeds if all(len(have[(a, s)]) >= widest for a in arms)]
             if args.seeds is None else args.seeds)
    if not seeds:
        print(f"{path}: no seed is complete across all arms yet ({len(rows)} branches)")
        return
    rows = [r for r in rows if r["seed"] in seeds]
    print(f"{path}: {len(rows)} branches, arms {arms}, seeds {seeds} of {all_seeds}")

    # Only questions every arm has finished at every seed are comparable.
    per: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        per[(r["question_id"], r["arm"])].append(r)
    qs = sorted({q for q, _ in per})
    full = [q for q in qs if all(len(per[(q, a)]) == len(seeds) for a in arms)]
    print(f"{len(qs)} questions seen, {len(full)} complete across every arm x seed\n")
    if not full:
        return
    info = {q: per[(q, arms[0])][0] for q in full}

    def cell(q: str, a: str, f: str) -> float:
        v = [x[f] for x in per[(q, a)]]
        return sum(v) / len(v)

    print("## per arm (complete questions only)")
    hdr = (f"{'arm':6s} {'acc':>7s} {'answered':>9s} {'cont tok':>9s} {'think tok':>10s} "
           f"{'capped':>7s} {'doubt/brk':>10s} {'boundaries':>11s}")
    print(hdr)
    for a in arms:
        rs = [x for q in full for x in per[(q, a)]]
        acc = sum(x["correct"] for x in rs) / len(rs)
        ans = sum(x["has_answer"] for x in rs) / len(rs)
        dr = [x["doubt_rate"] for x in rs if x["doubt_rate"] >= 0]
        print(f"{a:6s} {acc:6.1%} {ans:8.1%} {med([x['cont_tokens'] for x in rs]):9.0f} "
              f"{med([x['cont_think_tokens'] for x in rs if x['cont_think_tokens'] >= 0]):10.0f} "
              f"{sum(x['capped'] for x in rs) / len(rs):6.1%} {med(dr):10.3f} "
              f"{med([float(x['boundaries']) for x in rs]):11.0f}")
    commit = sum(info[q]["cand_correct"] for q in full) / len(full)
    print(f"{'commit':6s} {commit:6.1%}    (deterministic: return the detected candidate)\n")

    print("## paired vs " + args.base + ", clustered by question")
    print(f"{'arm':6s} {'stratum':16s} {'n':>4s} {'Δacc':>8s} {'95% CI':>18s} {'p':>7s} "
          f"{'Δthink':>8s} {'Δtokens':>9s}")
    strata: list[tuple[str, Any]] = [
        ("all", lambda q: True),
        ("cand correct", lambda q: info[q]["cand_correct"] == 1),
        ("cand wrong", lambda q: info[q]["cand_correct"] == 0),
    ]
    strata += [(b, (lambda q, b=b: info[q]["band"] == b)) for b in BANDS]
    for a in arms:
        if a == args.base:
            continue
        for name, keep in strata:
            sel = [q for q in full if keep(q)]
            if len(sel) < 2:
                continue
            d = [cell(q, a, "correct") - cell(q, args.base, "correct") for q in sel]
            m, hw, p = clustered(d)
            # `exit` never re-enters <think>, so its think length is n/a (-1);
            # the total-token column is the comparable one for that arm.
            dt = [cell(q, a, "cont_think_tokens") - cell(q, args.base, "cont_think_tokens")
                  for q in sel] if a != "exit" else []
            dk = [cell(q, a, "cont_tokens") - cell(q, args.base, "cont_tokens") for q in sel]
            tstr = f"{sum(dt) / len(dt):+8.0f}" if dt else f"{'n/a':>8s}"
            print(f"{a:6s} {name:16s} {len(sel):4d} {m:+7.1%} "
                  f"[{m - hw:+6.1%},{m + hw:+6.1%}] {p:7.3f} {tstr} "
                  f"{sum(dk) / len(dk):+9.0f}")
        print()

    print("## what the continuation did to the candidate (rate over question x seed)")
    print(f"{'arm':6s} {'keep':>7s} {'break':>7s} | {'repair':>8s} {'stuck':>7s} "
          f"{'no answer':>10s}")
    for a in arms:
        good = [x for q in full if info[q]["cand_correct"] for x in per[(q, a)]]
        bad = [x for q in full if not info[q]["cand_correct"] for x in per[(q, a)]]
        kb = sum(x["flip"] == "break" for x in good) / max(len(good), 1)
        rp = sum(x["flip"] == "repair" for x in bad) / max(len(bad), 1)
        na = [x for q in full for x in per[(q, a)]]
        print(f"{a:6s} {1 - kb:6.1%} {kb:6.1%} | {rp:7.1%} {1 - rp:6.1%} "
              f"{1 - sum(x['has_answer'] for x in na) / len(na):9.1%}")
    print("\n`break` = candidate was right and the continuation lost it.")
    print("`repair` = candidate was wrong and the continuation fixed it.")
    print("Commit never breaks and never repairs, by construction.")


if __name__ == "__main__":
    main()
