#!/usr/bin/env python3
"""Pre-stated alpha selection (transfer logs): among arms with dAcc >= -1pp and cap rate
<= 2x base, pick the saving closest to --target-saved (the 1.7B's 48.8%); ties -> smaller
alpha. Paired on (question, seed) against the base's same seeds. Prints the chosen policy
name, or NONE."""
import argparse, glob
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="glob of base rollouts.csv dirs")
ap.add_argument("--arms", required=True, help="glob of arm rollouts.csv dirs")
ap.add_argument("--target-saved", type=float, default=48.8)
a = ap.parse_args()
load = lambda pat: pd.concat([pd.read_csv(f) for p in glob.glob(pat) for f in glob.glob(p + "/rollouts.csv")])
B, A = load(a.base), load(a.arms)
B = B[B.policy == "base"]
rows = []
for pol, g in A[A.policy != "base"].groupby("policy"):
    m = g.merge(B, on=["question_id", "seed"], suffixes=("", "_b"))
    q = m.groupby("question_id")[["correct", "correct_b", "total_tokens", "total_tokens_b"]].mean()
    dacc = 100 * (q.correct - q.correct_b).mean()
    saved = 100 * (1 - q.total_tokens.mean() / q.total_tokens_b.mean())
    cap, capb = 100 * m.capped.mean(), 100 * m.capped_b.mean()
    alpha = float(g.alpha.iloc[0])
    ok = dacc >= -1 and cap <= max(2 * capb, 2 * 0.25)  # 2x base (a 0-cap base allows 0.5%)
    rows.append((pol, alpha, dacc, saved, cap, capb, ok))
    print(f"{pol:18s} a={alpha:<5} dacc {dacc:+.2f}  saved {saved:5.1f}%  cap {cap:.2f}% (base {capb:.2f}%)  eligible {ok}")
el = sorted([r for r in rows if r[6]], key=lambda r: (abs(r[3] - a.target_saved), r[1]))
print("SELECTED", el[0][0] if el else "NONE")
