#!/usr/bin/env python3
"""Score post-answer suppression.

Primary endpoint: **post-answer share** -- tokens generated after the gold value
first appears, as a fraction of the trace. Baseline wastes 69% of its reasoning
there (loop_metrics.py); every-boundary suppression only reaches 53%.

The metric that carries the risk, and the interesting one: **talked itself out of
it** -- of the traces that DID write the gold value during reasoning, how many
ended up answering something else. Suppressing doubt after the answer should
lower that rate if the doubt was destroying correct answers, and raise it if
post-answer reasoning catches real errors.

Traces that never state the gold get no injection and are excluded from the
paired comparisons; their count is reported instead.

    uv run python scripts/after_answer_report.py
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics as st
from pathlib import Path

ARMS = {"baseline": "baseline (no push)",
        "N_after": "N: NLA, after answer",
        "D_after": "D: diff-means, after answer"}
BANDS = ("all_right", "mixed", "mostly_wrong", "all_wrong")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dir", type=Path, default=Path("data/afteranswer"))
    return p.parse_args()


def load(path: Path) -> list[dict[str, str]]:
    if path.exists():
        return list(csv.DictReader(path.open()))
    part = Path(str(path) + ".partial")
    return list(csv.DictReader(part.open())) if part.exists() else []


def main() -> None:
    args = parse_args()
    got = {t: {r["question_id"]: r for r in load(args.dir / f"{t}.csv")} for t in ARMS}
    got = {k: v for k, v in got.items() if v}
    if "baseline" not in got:
        raise SystemExit("baseline arm not started yet")
    ids = sorted(set.intersection(*(set(v) for v in got.values())))
    print(f"Post-answer suppression — {len(ids)} questions scored "
          f"({', '.join(f'{k}:{len(v)}' for k, v in got.items())})\n")

    print(f"{'arm':<30}{'triggered':>11}{'post-ans':>10}{'med think':>11}"
          f"{'doubt blk':>11}{'accuracy':>10}")
    print("-" * 83)
    for tag, lab in ARMS.items():
        if tag not in got:
            continue
        rs = [got[tag][q] for q in ids]
        tr = [r for r in rs if r["triggered"] == "1"]
        post = st.mean(float(r["post_share"]) for r in tr) if tr else float("nan")
        print(f"{lab:<30}{100*len(tr)/len(rs):>10.0f}%{100*post:>9.0f}%"
              f"{st.median(int(r['think_tokens']) for r in rs):>11.0f}"
              f"{st.mean(int(r['doubt_blocks']) for r in rs):>11.2f}"
              f"{100*sum(int(r['correct']) for r in rs)/len(rs):>9.1f}%")

    print("\ntrigger rate by band — did the model ever write the gold value?")
    print(f"{'band':<16}" + "".join(f"{t[:13]:>15}" for t in got))
    for b in BANDS:
        qs = [q for q in ids if got["baseline"][q]["band"] == b]
        if not qs:
            continue
        cells = "".join(
            f"{100*sum(1 for q in qs if got[t][q]['triggered']=='1')/len(qs):>14.0f}%"
            for t in got)
        print(f"{b:<16}" + cells)

    print("\npost-answer share, triggered traces only (primary endpoint)")
    print(f"{'band':<16}" + "".join(f"{t[:13]:>15}" for t in got))
    for b in BANDS:
        cells = []
        for t in got:
            v = [float(got[t][q]["post_share"]) for q in ids
                 if got["baseline"][q]["band"] == b and got[t][q]["triggered"] == "1"]
            cells.append(f"{100*st.mean(v):>14.0f}%" if v else f"{'-':>15}")
        print(f"{b:<16}" + "".join(cells))

    print("\n'talked itself out of it' — wrote the gold value, then answered otherwise")
    for tag, lab in ARMS.items():
        if tag not in got:
            continue
        tr = [got[tag][q] for q in ids if got[tag][q]["triggered"] == "1"]
        lost = sum(1 for r in tr if r["correct"] == "0")
        print(f"  {lab:<30}{lost:>4}/{len(tr):<4} = {100*lost/max(len(tr),1):>5.1f}%")

    print("\npaired accuracy vs baseline (all questions)")
    for tag, lab in ARMS.items():
        if tag == "baseline" or tag not in got:
            continue
        pairs = [(int(got["baseline"][q]["correct"]), int(got[tag][q]["correct"])) for q in ids]
        b = sum(1 for x, y in pairs if x and not y)
        c = sum(1 for x, y in pairs if y and not x)
        n = len(pairs)
        d = (c - b) / n
        se = math.sqrt(b + c) / n if b + c else 0
        chi = (abs(b - c) - 1) ** 2 / (b + c) if b + c else 0
        p = math.erfc(math.sqrt(chi / 2)) if chi > 0 else 1.0
        print(f"  {lab:<30}b={b:>3} c={c:>3}  {100*d:+6.1f}pp  "
              f"95% CI [{100*(d-1.96*se):+5.1f}, {100*(d+1.96*se):+5.1f}]  p={p:.3f}")


if __name__ == "__main__":
    main()
