#!/usr/bin/env python3
r"""EXPERIMENT-probe-gated.md §G5: match each random arm's unsteered share to its probe arm.

  target   u_G = realized unsteered share of G5 / G10 / G20 on mathtest_gate400, pooled:
           sum(unsteered_sites) / sum(eligible_sites) over all rollouts of the arm
  grid     p0 = 1 - (1 - u_G)^(1/4) (a firing covers 4 sites), and p0 x {0.85, 1, 1.15}
  calib    data/gated/calib50.jsonl: 50 MATH-train questions of the probe-training pool
           (source1000), seed 20261003; 2 seeds; the realized u(p) per grid point
  choice   p_q by linear interpolation of u(p) over the arm's 3 points (extrapolated at
           the ends), rounded to 4 decimals -> data/gated/calibration.json

    uv run python scripts/gate_calibrate.py --targets        # write targets + grid policies
    uv run python scripts/gate_calibrate.py --grid-policies  # print the grid policy names
    uv run python scripts/gate_calibrate.py --choose         # fit p_q from the calib run
    uv run python scripts/gate_calibrate.py --policies [Q..] # print R policy names (p_q)
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/gated"
SEED = 20261003
QS = ("5", "10", "20")
POL = "N@a1.0d256"


def share(rows: list[dict]) -> tuple[float, int, int]:
    u = sum(int(r["unsteered_sites"]) for r in rows)
    e = sum(int(r["eligible_sites"]) for r in rows)
    return (u / e if e else float("nan")), u, e


def build_calib50() -> None:
    path = OUT / "calib50.jsonl"
    if path.exists():
        return
    src = sorted({json.loads(x)["question_id"] for x in open(ROOT / "data/selective/source1000.jsonl")})
    pick = set(random.Random(SEED).sample(src, 50))
    pool = [json.loads(x) for x in open(ROOT / "data/distill/train.jsonl")]
    rows = [r for r in pool if r["question_id"] in pick]
    assert len(rows) == 50
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", action="store_true")
    ap.add_argument("--grid-policies", action="store_true")
    ap.add_argument("--choose", action="store_true")
    ap.add_argument("--policies", nargs="*", default=None)
    args = ap.parse_args()
    tpath, cpath = OUT / "calib_targets.json", OUT / "calibration.json"
    if args.targets:
        rows = [r for f in sorted(glob.glob(str(OUT / "eval/gate400_s*/rollouts.csv")))
                for r in csv.DictReader(open(f))]
        t = {}
        for q in QS:
            u, nu, ne = share([r for r in rows if r["policy"] == f"{POL}G{q}"])
            p0 = 1 - (1 - u) ** 0.25
            t[q] = {"u_target": round(u, 5), "unsteered": nu, "eligible": ne, "p0": round(p0, 4),
                    "grid": [round(p0 * m, 4) for m in (0.85, 1.0, 1.15)]}
        build_calib50()
        tpath.write_text(json.dumps(t, indent=1) + "\n")
        print(json.dumps(t, indent=1))
    elif args.grid_policies:
        t = json.loads(tpath.read_text())
        print(" ".join(sorted({f"{POL}R{p:.4f}" for q in QS for p in t[q]["grid"]})))
    elif args.choose:
        t = json.loads(tpath.read_text())
        rows = [r for f in sorted(glob.glob(str(OUT / "calib/s*/rollouts.csv")))
                for r in csv.DictReader(open(f))]
        out = {}
        for q in QS:
            ps = t[q]["grid"]
            us = [share([r for r in rows if r["policy"] == f"{POL}R{p:.4f}"])[0] for p in ps]
            a, b = np.polyfit(ps, us, 1) if len(set(us)) > 1 else (0.0, us[0])
            # piecewise-linear interpolation; linear fit through the 3 points outside them
            order = np.argsort(us)
            lo, hi = us[order[0]], us[order[-1]]
            tgt = t[q]["u_target"]
            if lo <= tgt <= hi:
                p = float(np.interp(tgt, np.array(us)[order], np.array(ps)[order]))
            else:
                p = float((tgt - b) / a)
            out[q] = {"u_target": tgt, "grid": ps, "u_calib": [round(u, 5) for u in us],
                      "p": round(p, 4), "policy": f"{POL}R{round(p, 4):.4f}",
                      "calib_rollouts_per_point": [sum(r["policy"] == f"{POL}R{x:.4f}" for r in rows)
                                                   for x in ps]}
        cpath.write_text(json.dumps(out, indent=1) + "\n")
        print(json.dumps(out, indent=1))
    elif args.policies is not None:
        c = json.loads(cpath.read_text())
        print(" ".join(c[q]["policy"] for q in (args.policies or QS)))
    else:
        sys.exit("nothing to do")


if __name__ == "__main__":
    main()
