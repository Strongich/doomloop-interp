#!/usr/bin/env python3
"""EXPERIMENT-negN-and-gsm8k-full.md: readouts for both runs.

All rows are regraded with the D69 grader from rollouts.jsonl. CIs are a question-clustered
bootstrap (2,000 reps, seed 20261005), one resample per set, shared by every quantity so
differences between arms are paired.

    python scripts/negn_gsm8k_report.py negn          # Run 1: -N vs base (and vs P) on dev400
    python scripts/negn_gsm8k_report.py gsm8k         # Run 2: full GSM8K test, by group
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPS, SEED = 2000, 20261005
ALPHAS = ("0.25", "0.5", "1.0")


def _load(path: str) -> pd.DataFrame:
    from reasoning_attention.grading import grade

    rows = []
    for line in open(path):
        r = json.loads(line)
        g = grade(r["text"], r["gold"])
        rows.append({
            "question_id": r["question_id"], "policy": r["policy"], "seed": int(r["seed"]),
            "correct": int(g.is_correct), "total_tokens": int(r["total_tokens"]),
            "capped": int(r["capped"]), "doubt_blocks": int(r["doubt_blocks"]),
            "markers": int(r["markers"]),
        })
    return pd.DataFrame(rows)


def load(runs: list[Path]) -> pd.DataFrame:
    with ProcessPoolExecutor(8) as ex:
        df = pd.concat(ex.map(_load, [str(d / "rollouts.jsonl") for d in runs]))
    dup = df.duplicated(["question_id", "policy", "seed"])
    assert not dup.any(), f"{dup.sum()} duplicate (question, policy, seed) rows across runs"
    return df


def paired(df: pd.DataFrame, arm: str, ref: str, qs: list[str], idx: np.ndarray) -> dict:
    a, b = df[df.policy == arm], df[df.policy == ref]
    m = a.merge(b, on=["question_id", "seed"], suffixes=("", "_r"))
    assert len(m) == len(a) == len(b), (arm, ref, len(m), len(a), len(b))
    qm = m.groupby("question_id").mean(numeric_only=True).reindex(qs)
    qs_ = m.groupby("question_id").sum(numeric_only=True).reindex(qs)

    def ci(boot: np.ndarray) -> list[float]:
        return [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]

    out = {"n": len(m), "acc": m.correct.mean() * 100, "acc_ref": m.correct_r.mean() * 100,
           "tokens": m.total_tokens.mean(), "tokens_ref": m.total_tokens_r.mean(),
           "capped": m.capped.mean() * 100, "capped_ref": m.capped_r.mean() * 100,
           "doubt": m.doubt_blocks.mean(), "doubt_ref": m.doubt_blocks_r.mean()}
    for key, col in (("dacc", "correct"), ("ddoubt", "doubt_blocks"), ("dmarkers", "markers")):
        d = (qm[col] - qm[f"{col}_r"]).to_numpy() * (100 if col == "correct" else 1)
        out[key], out[f"{key}_ci"] = float(d.mean()), ci(d[idx].mean(1))
    t, tr = qs_.total_tokens.to_numpy(), qs_.total_tokens_r.to_numpy()
    out["length"] = float(t.sum() / tr.sum() - 1)
    out["length_ci"] = ci(t[idx].sum(1) / tr[idx].sum(1) - 1)
    return out


def fmt(name: str, r: dict) -> str:
    def c(k: str, f: str = "+.2f") -> str:
        return f"{r[k]:{f}} [{r[k + '_ci'][0]:{f}}, {r[k + '_ci'][1]:{f}}]"

    return (f"| {name} | {r['n']} | {r['acc']:.1f} vs {r['acc_ref']:.1f} | {c('dacc')} | "
            f"{r['tokens']:,.0f} vs {r['tokens_ref']:,.0f} | {r['length']:+.1%} "
            f"[{r['length_ci'][0]:+.1%}, {r['length_ci'][1]:+.1%}] | "
            f"{r['doubt']:.2f} vs {r['doubt_ref']:.2f} | {c('ddoubt')} | {c('dmarkers')} | "
            f"{r['capped']:.2f}% vs {r['capped_ref']:.2f}% |")


HEAD = ("| arm vs ref | pairs | acc % | Δacc pp [CI] | tokens | length [CI] | doubt/rollout | "
        "Δdoubt [CI] | Δmarkers [CI] | capped |\n|" + "---|" * 10)


def negn(args: argparse.Namespace) -> None:
    # -N has its own base run; P is paired with the base of the run it came from.
    ref_run = load([args.base_run])
    ref_run["policy"] = ref_run.policy.replace("base", "base_ref")
    df = pd.concat([load([args.run]), ref_run])
    qs = sorted(df[df.policy == "base"].question_id.unique())
    idx = np.random.default_rng(SEED).integers(0, len(qs), (REPS, len(qs)))
    report, lines = {}, ["## Run 1: -N on dev400 x 2 (P and base_ref from " f"{args.base_run})",
                         "", HEAD]
    report["base vs base_ref"] = paired(df, "base", "base_ref", qs, idx)
    lines.append(fmt("base vs base_ref", report["base vs base_ref"]))
    for a in ALPHAS:
        for arm, ref in ((f"negN@a{a}d256", "base"), (f"P@a{a}d256", "base_ref"),
                         (f"negN@a{a}d256", f"P@a{a}d256")):
            r = paired(df, arm, ref, qs, idx)
            report[f"{arm} vs {ref}"] = r
            lines.append(fmt(f"{arm} vs {ref}", r))
    lines.append("")
    for a in ALPHAS:
        r = report[f"negN@a{a}d256 vs base"]
        more = r["ddoubt_ci"][0] > 0
        longer = r["length_ci"][0] > 0
        lines.append(f"α {a}: -N adds doubt (CI > 0): {more}; -N lengthens output (CI > 0): "
                     f"{longer}; Δacc {r['dacc']:+.2f}pp")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float))
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def gsm8k(args: argparse.Namespace) -> None:
    df = load(args.run)
    groups = {json.loads(x)["question_id"]: json.loads(x)["group"]
              for x in args.cohort.read_text().splitlines() if x.strip()}
    report, lines = {}, ["## Run 2: full GSM8K test x 4 seeds", "", HEAD]
    for g in ("all", "fresh997", "touched322"):
        qs = sorted(q for q, v in groups.items() if g == "all" or v == g)
        sub = df[df.question_id.isin(qs)]
        missing = set(qs) - set(sub.question_id)
        assert not missing, f"{g}: {len(missing)} cohort questions have no rollouts"
        idx = np.random.default_rng(SEED).integers(0, len(qs), (REPS, len(qs)))
        lines.append(f"| **{g} ({len(qs)})** |" + " |" * 9)
        for arm in ("N@a1.0d256", "brevityA"):
            r = paired(sub, arm, "base", qs, idx)
            report[f"{g}: {arm} vs base"] = r
            lines.append(fmt(f"{arm} vs base", r))
        r = paired(sub, "N@a1.0d256", "brevityA", qs, idx)
        report[f"{g}: N vs brevityA"] = r
        lines.append(fmt("N@a1.0d256 vs brevityA", r))
    lines.append("\nTokens saved = -(length). Δacc in percentage points.")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=float))
    args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("negn")
    a.add_argument("--base-run", type=Path, default=Path("data/vers/dev"))
    a.add_argument("--run", type=Path, default=Path("data/vers/negN_dev"))
    a.add_argument("--out", type=Path, default=Path("data/vers/negN_report"))
    b = sub.add_parser("gsm8k")
    b.add_argument("--run", type=Path, nargs="+", default=[Path("data/eval_gsm8k_full/run")],
                   help="one run dir, or one per seed when seeds were split across GPUs")
    b.add_argument("--cohort", type=Path, default=Path("data/policy/gsm8k_test_1319.jsonl"))
    b.add_argument("--out", type=Path, default=Path("data/eval_reports/gsm8k_full"))
    args = ap.parse_args()
    negn(args) if args.cmd == "negn" else gsm8k(args)


if __name__ == "__main__":
    main()
