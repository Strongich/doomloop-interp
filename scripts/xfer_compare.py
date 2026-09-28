#!/usr/bin/env python3
"""Arm-vs-base table for transfer runs, question-clustered (Finding 12 conventions).

Each --run is a rollouts.csv (or its directory); rows are pooled and tagged with
--tag so the 1.7B and 8B runs can be reported side by side. Every arm is compared
to the `base` policy OF ITS OWN TAG on the (question, seed) pairs both have.

  dacc     accuracy difference, pp; seeds averaged within question, 95% bootstrap
           over questions
  saved    1 - mean(tokens_arm)/mean(tokens_base), %, same bootstrap
  doubt    1 - mean(doubt_blocks_arm)/mean(doubt_blocks_base), % removed

    uv run python scripts/xfer_compare.py --run 17=data/xfer8b/r17_dev 8b=data/xfer8b/r8b_dev_*
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd


def load(spec: str) -> pd.DataFrame:
    tag, _, pat = spec.partition("=")
    frames = []
    for p in sorted(glob.glob(pat)):
        p = Path(p)
        f = p / "rollouts.csv" if p.is_dir() else p
        if not f.exists():
            f = f.with_name("rollouts.csv.partial")
        if f.exists():
            frames.append(pd.read_csv(f))
    df = pd.concat(frames).drop_duplicates(["question_id", "policy", "seed"])
    df["tag"] = tag
    return df


def boot(fn, qs: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    est = fn(np.arange(len(qs)))
    bs = [fn(rng.integers(0, len(qs), len(qs))) for _ in range(n)]
    return est, float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def compare(base: pd.DataFrame, arm: pd.DataFrame) -> dict:
    j = base.merge(arm, on=["question_id", "seed"], suffixes=("_b", "_a"))
    if j.empty:
        return {}
    g = j.groupby("question_id").agg(
        cb=("correct_b", "mean"), ca=("correct_a", "mean"),
        tb=("total_tokens_b", "mean"), ta=("total_tokens_a", "mean"),
        db=("doubt_blocks_b", "mean"), da=("doubt_blocks_a", "mean"),
        capb=("capped_b", "mean"), capa=("capped_a", "mean"))
    A = g.to_numpy()
    cols = {c: i for i, c in enumerate(g.columns)}

    def dacc(ix):
        return 100 * (A[ix, cols["ca"]] - A[ix, cols["cb"]]).mean()

    def saved(ix):
        return 100 * (1 - A[ix, cols["ta"]].sum() / A[ix, cols["tb"]].sum())

    def doubt(ix):
        return 100 * (1 - A[ix, cols["da"]].sum() / max(A[ix, cols["db"]].sum(), 1e-9))

    qs = g.index.to_numpy()
    return {"n_q": len(qs), "n_pairs": len(j), "acc_base": 100 * g.cb.mean(), "acc_arm": 100 * g.ca.mean(),
            "dacc": boot(dacc, qs), "tok_base": g.tb.mean(), "tok_arm": g.ta.mean(),
            "saved": boot(saved, qs), "doubt_removed": doubt(np.arange(len(qs))),
            "cap_base": 100 * g.capb.mean(), "cap_arm": 100 * g.capa.mean()}


def fmt(r: dict) -> str:
    if not r:
        return "(no overlap)"
    d, s = r["dacc"], r["saved"]
    return (f"n={r['n_q']:4d} acc {r['acc_base']:5.1f}->{r['acc_arm']:5.1f}  "
            f"dacc {d[0]:+5.1f} [{d[1]:+5.1f},{d[2]:+5.1f}]  tok {r['tok_base']:6.0f}->{r['tok_arm']:6.0f}  "
            f"saved {s[0]:+5.1f}% [{s[1]:+5.1f},{s[2]:+5.1f}]  doubt-rm {r['doubt_removed']:+5.1f}%  "
            f"cap {r['cap_base']:.1f}/{r['cap_arm']:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", nargs="+", required=True, help="TAG=glob")
    ap.add_argument("--cohort", nargs="*", default=["data/xfer8b/dev_math250.jsonl",
                                                    "data/policy/math500.jsonl"])
    ap.add_argument("--by-level", action="store_true")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    df = pd.concat([load(s) for s in args.run])
    lv = {}
    for c in args.cohort:
        if Path(c).exists():
            for line in open(c):
                r = json.loads(line)
                if "level" in r:
                    lv[r["question_id"]] = r["level"]
    df["level"] = df.question_id.map(lv).fillna(0).astype(int)
    out = {}
    for tag, d in df.groupby("tag", sort=False):
        base = d[d.policy == "base"]
        print(f"== {tag}: base n={base.question_id.nunique()} q, acc {100*base.correct.mean():.1f}%, "
              f"tok {base.total_tokens.mean():.0f}, doubt/rollout {base.doubt_blocks.mean():.1f}, "
              f"cap {100*base.capped.mean():.1f}%")
        for pol in sorted(p for p in d.policy.unique() if p != "base"):
            r = compare(base, d[d.policy == pol])
            out[f"{tag}/{pol}"] = r
            print(f"  {pol:22s} {fmt(r)}")
            if args.by_level:
                for L in sorted(d.level.unique()):
                    rl = compare(base[base.level == L], d[(d.policy == pol) & (d.level == L)])
                    out[f"{tag}/{pol}/L{L}"] = rl
                    print(f"    L{L} {fmt(rl)}")
    if args.json:
        args.json.write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
