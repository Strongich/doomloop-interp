"""Exploratory alpha dose-response on MATH-500 levels 4-5 (EXPERIMENT §11).

Joins the new sweep (base + N at alpha 0.25/0.5/0.75) to Finding 12's alpha=1.0
point on the same questions and seeds. Each arm is compared with the baseline
from ITS OWN run: new arms against the new base, Finding 12's arms against
Finding 12's base. The two bases are compared first; the curve is only joined
if they agree.

Statistics are question-clustered: seeds are averaged within a question, then
questions are resampled.

    uv run python scripts/alpha_report.py \
        --new data/eval_math500_alpha/stage2/rollouts_all.csv \
        --old data/eval_math500/stage2/rollouts_all.csv \
        --cohort data/policy/math500_L45.jsonl
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

B = 10_000
RNG = np.random.default_rng(0)


def per_question(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("question_id")[col].mean()


def paired(base: pd.DataFrame, arm: pd.DataFrame) -> dict:
    """Δaccuracy (pp) and token saving (%) with question-bootstrap 95% CIs."""
    ab, aa = per_question(base, "correct"), per_question(arm, "correct")
    tb, ta = per_question(base, "total_tokens"), per_question(arm, "total_tokens")
    q = ab.index.intersection(aa.index)
    ab, aa, tb, ta = (s.loc[q].to_numpy() for s in (ab, aa, tb, ta))
    idx = RNG.integers(0, len(q), size=(B, len(q)))
    d = (aa - ab) * 100
    dboot = d[idx].mean(1)
    sav = 1 - ta.sum() / tb.sum()
    sboot = 1 - ta[idx].sum(1) / tb[idx].sum(1)
    return {
        "n_q": len(q),
        "base_acc": ab.mean() * 100,
        "arm_acc": aa.mean() * 100,
        "d_pp": d.mean(),
        "d_lo": np.percentile(dboot, 2.5),
        "d_hi": np.percentile(dboot, 97.5),
        "save_pct": sav * 100,
        "s_lo": np.percentile(sboot, 2.5) * 100,
        "s_hi": np.percentile(sboot, 97.5) * 100,
    }


def pipeline(base: pd.DataFrame, arm: pd.DataFrame | None, k_arm: int = 2) -> dict:
    """Gold-verified generation: seeds 0..k_arm-1 of the arm, then the LAST 4-k_arm
    base seeds only for questions with no correct arm rollout. Accepted = shortest
    correct. arm=base is the staged base-only pipeline (seeds 0-1, then 2-3);
    arm=None spends all 4 base seeds."""
    spend = covered = 0
    acc_len = []
    for qid, gb in base.groupby("question_id"):
        gb = gb.sort_values("seed")
        if arm is None:
            pool = gb
        else:
            ga = arm[arm.question_id == qid].sort_values("seed").head(k_arm)
            pool = ga
            if not ga.correct.any():
                pool = pd.concat([ga, gb.tail(4 - k_arm)])
        spend += pool.total_tokens.sum()
        ok = pool[pool.correct == 1]
        if len(ok):
            covered += 1
            acc_len.append(ok.total_tokens.min())
    n = base.question_id.nunique()
    return {
        "coverage": covered / n * 100,
        "accepted_len": float(np.mean(acc_len)) if acc_len else float("nan"),
        "tok_per_accepted": spend / max(covered, 1),
    }


def fmt(r: dict) -> str:
    return (
        f"n={r['n_q']:3d}  base {r['base_acc']:5.1f}  arm {r['arm_acc']:5.1f}  "
        f"Δ {r['d_pp']:+5.1f} [{r['d_lo']:+5.1f}, {r['d_hi']:+5.1f}]  "
        f"saved {r['save_pct']:5.1f}% [{r['s_lo']:5.1f}, {r['s_hi']:5.1f}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--cohort", required=True)
    ap.add_argument("--out", default=None, help="optional JSON dump of all numbers")
    args = ap.parse_args()

    levels = {
        (r := json.loads(line))["question_id"]: r["level"] for line in open(args.cohort)
    }
    new = pd.read_csv(args.new)
    old = pd.read_csv(args.old)
    new = new[new.question_id.isin(levels)]
    old = old[old.question_id.isin(levels)]
    for df in (new, old):
        df["level"] = df.question_id.map(levels)
    seeds = sorted(new.seed.unique())
    old = old[old.seed.isin(seeds)]

    out: dict = {}
    nb, ob = new[new.policy == "base"], old[old.policy == "base"]

    print("== baseline agreement (Finding 12 base vs new base, same questions/seeds)")
    agree = paired(ob, nb)
    out["base_agreement"] = agree
    print("  old->new", fmt(agree))
    j = ob.set_index(["question_id", "seed"]).correct.to_frame("o").join(
        nb.set_index(["question_id", "seed"]).correct.rename("n"), how="inner"
    )
    same = (j.o == j.n).mean()
    print(f"  rollout-level agreement on correctness: {same * 100:.1f}% of {len(j)} pairs")

    arms = [(p, new, nb) for p in sorted(new.policy.unique()) if p != "base"]
    arms += [(p, old, ob) for p in ("N@a1.0d256", "D@a0.5d512") if p in set(old.policy)]
    for label, sel in (("all L4-5", None), ("L4", 4), ("L5", 5)):
        print(f"\n== {label}")
        for pol, df, base in arms:
            b = base if sel is None else base[base.level == sel]
            a = df[df.policy == pol]
            a = a if sel is None else a[a.level == sel]
            src = "F12" if df is old else "new"
            r = paired(b, a)
            out.setdefault(label, {})[pol] = r
            print(f"  {pol:12s} ({src})  {fmt(r)}")

    print("\n== pipeline (gold check; arm x2 then base x2 fallback; accepted = shortest correct)")
    for src, base in (("new", nb), ("F12", ob)):
        for name, arm in ((f"base x4 ({src})", None), (f"base x2 + x2 ({src})", base)):
            p = pipeline(base, arm)
            out.setdefault("pipeline", {})[name] = p
            print(
                f"  {name:30s}coverage {p['coverage']:5.1f}%  "
                f"accepted len {p['accepted_len']:7.0f}  tok/accepted {p['tok_per_accepted']:7.0f}"
            )
    for pol, df, base in arms:
        p = pipeline(base, df[df.policy == pol])
        out["pipeline"][pol] = p
        print(
            f"  {pol + ' x2 + base x2':30s}coverage {p['coverage']:5.1f}%  "
            f"accepted len {p['accepted_len']:7.0f}  tok/accepted {p['tok_per_accepted']:7.0f}"
        )

    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2, default=float)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
