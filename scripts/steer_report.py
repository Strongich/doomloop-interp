#!/usr/bin/env python3
"""Aggregate a set of steer_sweep CSVs into one alpha-by-condition table.

The sweep prints its own numbers per run; this joins several runs so the
dose-response and the point where the random control starts doing damage are
visible side by side.

    uv run python scripts/steer_report.py data/steer_suppress_a*.csv
"""

from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

CONDS = ("none", "suppress", "doubt", "random")
COL = {"none": "none", "suppress": "continue", "doubt": "doubt", "random": "random"}


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float, float]:
    bb = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    cc = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    chi = (abs(bb - cc) - 1) ** 2 / (bb + cc) if bb + cc else 0.0
    p = math.erfc(math.sqrt(chi / 2)) if chi > 0 else 1.0
    return bb, cc, chi, p


def main() -> None:
    paths = sorted(Path(p) for p in sys.argv[1:])
    print(f"{'alpha':>6}{'n':>5}{'samples':>9}" + "".join(f"{c:>16}" for c in CONDS))
    per_alpha = {}
    for path in paths:
        rows = list(csv.DictReader(path.open()))
        alpha = float(re.search(r"_a([0-9.]+)\.csv", path.name).group(1))
        samples = len(rows[0]["none_seq"])
        tot = len(rows) * samples
        cells = []
        for c in CONDS:
            hits = sum(int(r[f"{COL[c]}_hits"]) for r in rows)
            cells.append(f"{hits}/{tot} = {100 * hits / tot:4.1f}%")
        print(f"{alpha:>6}{len(rows):>5}{samples:>9}" + "".join(f"{x:>16}" for x in cells))
        per_alpha[alpha] = rows

    for alpha, rows in per_alpha.items():
        samples = len(rows[0]["none_seq"])
        print(f"\nalpha = {alpha}  ({len(rows)} probes x {samples} samples)")
        seq = {c: [ch == "1" for r in rows for ch in r[f"{COL[c]}_seq"]] for c in CONDS}
        anti = {
            c: [int(r[f"{COL[c]}_hits"]) < samples for r in rows] for c in CONDS
        }
        for a, b in (
            ("suppress", "none"),
            ("suppress", "doubt"),
            ("suppress", "random"),
            ("random", "none"),
            ("doubt", "none"),
        ):
            bb, cc, chi, p = mcnemar(seq[a], seq[b])
            bb2, cc2, chi2, p2 = mcnemar(anti[a], anti[b])
            print(
                f"  {a:>8} vs {b:<8} sample-level b={bb:<4} c={cc:<4} "
                f"chi2={chi:7.1f} p={p:9.3g}   |   probe-level(>=1 non-doubt) "
                f"b={bb2:<4} c={cc2:<4} chi2={chi2:6.1f} p={p2:.3g}"
            )


if __name__ == "__main__":
    main()
