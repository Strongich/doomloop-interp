#!/usr/bin/env python3
"""Score Stage A: do the cheap directions do what the NLA direction does?

Primary endpoint is DOUBT BLOCKS REMOVED. Vector A removes 92% of them; the
strongest system prompt removes 32% (Finding 7). That gap is what discriminates,
and it discriminates at n=50.

Length is secondary. Accuracy at n=150 resolves only ~11.5pp, so it is printed
as a bound and must not be read as a result.

Baseline is Finding 6's `none` column, paired by question_id.

    uv run python scripts/dom_report.py
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as st
from pathlib import Path

ARMS = {
    "A_ref": "N: NLA direction (35 deltas, 1 rollout)",
    "D_diffmeans": "D: difference of means",
    "P_probe": "P: linear probe weights",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("data/dom"))
    p.add_argument("--baseline", type=Path, default=Path("data/tier3/step1_A.csv"))
    p.add_argument("--traces", type=Path, default=Path("data/dom_testset150.jsonl"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    band = {json.loads(x)["question_id"]: json.loads(x)["band"] for x in args.traces.open()}
    base = {r["question_id"]: r for r in csv.DictReader(args.baseline.open())}

    got = {t: list(csv.DictReader((args.dir / f"{t}.csv").open()))
           for t in ARMS if (args.dir / f"{t}.csv").exists()}
    if not got:
        raise SystemExit("no arms yet — run scripts/run_dom_stageA.sh")
    ids = [r["question_id"] for r in next(iter(got.values())) if r["question_id"] in base]
    n = len(ids)
    b_doubt = st.mean(float(base[q]["none_doubt_blocks"]) for q in ids)
    b_think = st.median(int(base[q]["none_think_tokens"]) for q in ids)
    b_acc = sum(int(base[q]["none_correct"]) for q in ids) / n
    print(f"Stage A — {n} questions (50 per band), baseline from Finding 6\n")
    print(f"{'direction':<38}{'doubt blk':>11}{'removed':>9}{'med think':>11}{'length':>9}{'acc':>8}")
    print("-" * 86)
    print(f"{'baseline (no push)':<38}{b_doubt:>11.1f}{'':>9}{b_think:>11.0f}{'':>9}{100*b_acc:>7.1f}%")
    for tag, lab in ARMS.items():
        if tag not in got:
            continue
        rows = [r for r in got[tag] if r["question_id"] in base]
        db = st.mean(float(r["suppress_doubt_blocks"]) for r in rows)
        th = st.median(int(r["suppress_think_tokens"]) for r in rows)
        ac = sum(int(r["suppress_correct"]) for r in rows) / len(rows)
        print(f"{lab:<38}{db:>11.2f}{100*(1-db/b_doubt):>8.0f}%{th:>11.0f}"
              f"{100*(th-b_think)/b_think:>8.0f}%{100*ac:>7.1f}%")

    print("\nby band — doubt blocks removed (the endpoint this test exists for)")
    hdr = f"{'direction':<38}" + "".join(f"{b[:12]:>14}" for b in
                                         ("all_wrong", "mostly_wrong", "mixed"))
    print(hdr)
    for tag, lab in ARMS.items():
        if tag not in got:
            continue
        rows = {r["question_id"]: r for r in got[tag]}
        cells = []
        for bn in ("all_wrong", "mostly_wrong", "mixed"):
            qs = [q for q in ids if band.get(q) == bn]
            if not qs:
                cells.append(f"{'-':>14}")
                continue
            b0 = st.mean(float(base[q]["none_doubt_blocks"]) for q in qs)
            b1 = st.mean(float(rows[q]["suppress_doubt_blocks"]) for q in qs)
            cells.append(f"{100*(1-b1/b0):>13.0f}%")
        print(f"{lab:<38}" + "".join(cells))

    print("\naccuracy vs baseline — a BOUND, not a result (n=150 resolves ~11.5pp)")
    for tag, lab in ARMS.items():
        if tag not in got:
            continue
        rows = {r["question_id"]: r for r in got[tag]}
        pairs = [(int(base[q]["none_correct"]), int(rows[q]["suppress_correct"])) for q in ids]
        bb = sum(1 for x, y in pairs if x and not y)
        cc = sum(1 for x, y in pairs if y and not x)
        d = (cc - bb) / n
        se = math.sqrt(bb + cc) / n
        print(f"  {lab:<36} {100*d:+6.1f}pp  95% CI [{100*(d-1.96*se):+5.1f}, "
              f"{100*(d+1.96*se):+5.1f}]  (b={bb}, c={cc})")

    print("\nHow to read this:")
    print("  D or P removes ~90% of doubt  -> the NLA is not needed; the method")
    print("     scales to any model for free and the write-up is a finding, not a method.")
    print("  D and P remove much less (<= ~50%) -> editing a natural-language")
    print("     explanation finds a handle the standard contrastive tool misses.")
    print("  Cosines say they are different directions (D vs N 0.43, P vs N 0.07),")
    print("     so equal behaviour would mean doubt is a subspace, not a single line.")


if __name__ == "__main__":
    main()
