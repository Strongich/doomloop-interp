#!/usr/bin/env python3
r"""Student readout, LOCKED PROTOCOL v2 §L7-§L8.

Per model and set: pass@1 (seeds averaged per question) with a question-clustered 95% CI,
mean total tokens, median thinking tokens, capped rate, no-answer rate, doubt paragraphs
per rollout; MATH-500 by level. Paired contrasts bootstrap question-level differences
(same questions, same seeds). The §L8 win rule on each primary MATH set:
  tokens: CI of (NLA - natural) mean total tokens excludes 0 on the low side
  acc:    CI lower bound of (NLA - natural) pass@1 >= -2pp
val is a breakage monitor: a pass@1 drop > 3pp against Base flags the run.

    uv run python scripts/distill_report.py            # -> data/distill/report.{json,md}
"""

from __future__ import annotations

import collections
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "data/distill/eval"
B = 4000
PRIMARY_SETS = ("math500", "confirm400")
PRIMARY = [("sft_steered_lora", "sft_short_lora"), ("sft_steered_full", "sft_short_full"),
           ("dpo_steered_lora", "dpo_natural_lora"), ("dpo_steered_full", "dpo_natural_full")]
OPTIONAL = [("sft_steered_short_lora", "sft_short_lora"),
            ("sft_steered_short_full", "sft_short_full")]


def load(name: str) -> dict[str, dict[str, list[dict]]]:
    """set -> question_id -> rollouts."""
    out: dict = collections.defaultdict(lambda: collections.defaultdict(list))
    p = EVAL / name / "rollouts.csv"
    import csv

    for r in csv.DictReader(p.open()):
        out[r["set"]][r["question_id"]].append(r)
    return out


def qmeans(d: dict[str, list[dict]], key: str) -> dict[str, float]:
    return {q: float(np.mean([float(r[key]) for r in rs])) for q, rs in d.items()}


def boot_ci(x: np.ndarray, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarize(d: dict[str, list[dict]]) -> dict:
    rs = [r for v in d.values() for r in v]
    acc = np.array(list(qmeans(d, "correct").values()))
    a, lo, hi = boot_ci(acc)
    return {"n_q": len(d), "n": len(rs), "acc": a, "acc_lo": lo, "acc_hi": hi,
            "mean_total_tokens": float(np.mean([float(r["total_tokens"]) for r in rs])),
            "median_think_tokens": float(np.median([float(r["think_tokens"]) for r in rs])),
            "capped": float(np.mean([int(r["capped"]) for r in rs])),
            "no_answer": float(np.mean([1 - int(r["has_answer"]) for r in rs])),
            "doubt_per_rollout": float(np.mean([int(r["doubt_blocks"]) for r in rs]))}


def paired(a: dict[str, list[dict]], b: dict[str, list[dict]]) -> dict:
    qs = sorted(set(a) & set(b))
    if len(qs) != len(a) or len(qs) != len(b):
        raise ValueError("unpaired question sets")
    out = {}
    for key in ("correct", "total_tokens"):
        ma, mb = qmeans(a, key), qmeans(b, key)
        m, lo, hi = boot_ci(np.array([ma[q] - mb[q] for q in qs]))
        out[key] = {"delta": m, "lo": lo, "hi": hi}
    out["tokens_rel"] = out["total_tokens"]["delta"] / float(np.mean(list(qmeans(b, "total_tokens").values())))
    out["win_tokens"] = out["total_tokens"]["hi"] < 0
    out["noninferior_acc"] = out["correct"]["lo"] >= -0.02
    out["win"] = bool(out["win_tokens"] and out["noninferior_acc"])
    return out


def main() -> None:
    models = sorted(p.name for p in EVAL.iterdir() if (p / "rollouts.csv").exists())
    data = {m: load(m) for m in models}
    rep: dict = {"models": {}, "primary": {}, "optional": {}, "vs_base": {}, "flags": []}
    for m in models:
        rep["models"][m] = {s: summarize(d) for s, d in data[m].items()}
        if "math500" in data[m]:
            by_lv: dict = collections.defaultdict(dict)
            for q, rs in data[m]["math500"].items():
                by_lv[rs[0]["level"]][q] = rs
            rep["models"][m]["math500_by_level"] = {lv: summarize(d) for lv, d in sorted(by_lv.items())}
    for group, pairs in (("primary", PRIMARY), ("optional", OPTIONAL)):
        for x, y in pairs:
            if x in data and y in data:
                rep[group][f"{x} vs {y}"] = {s: paired(data[x][s], data[y][s])
                                             for s in data[x] if s in data[y]}
    if "base" in data:
        for m in models:
            if m == "base":
                continue
            rep["vs_base"][m] = {s: paired(data[m][s], data["base"][s])
                                 for s in data[m] if s in data["base"]}
            v = rep["vs_base"][m].get("val")
            if v and v["correct"]["delta"] < -0.03:
                rep["flags"].append(f"{m}: val pass@1 {100 * v['correct']['delta']:+.1f}pp vs base")
    (ROOT / "data/distill/report.json").write_text(json.dumps(rep, indent=1) + "\n")

    L = ["# Distillation readout (LOCKED PROTOCOL v2)", ""]
    sets = sorted({s for m in models for s in data[m]})
    for s in sets:
        L += [f"## {s}", "", "| model | pass@1 [95% CI] | mean tokens | median think | capped | "
              "no answer | doubt/rollout |", "|---|---|---:|---:|---:|---:|---:|"]
        for m in models:
            v = rep["models"][m].get(s)
            if v:
                L.append(f"| {m} | {100 * v['acc']:.1f} [{100 * v['acc_lo']:.1f}, "
                         f"{100 * v['acc_hi']:.1f}] | {v['mean_total_tokens']:.0f} | "
                         f"{v['median_think_tokens']:.0f} | {100 * v['capped']:.1f}% | "
                         f"{100 * v['no_answer']:.1f}% | {v['doubt_per_rollout']:.2f} |")
        L.append("")
    for group in ("primary", "optional", "vs_base"):
        L += [f"## Paired: {group}", "", "| contrast | set | Δacc pp [CI] | Δtokens [CI] (rel) | "
              "win |", "|---|---|---|---|---|"]
        items = rep[group].items() if group != "vs_base" else \
            ((f"{m} vs base", v) for m, v in rep[group].items())
        for name, per in items:
            for s, v in per.items():
                c, t = v["correct"], v["total_tokens"]
                L.append(f"| {name} | {s} | {100 * c['delta']:+.1f} [{100 * c['lo']:+.1f}, "
                         f"{100 * c['hi']:+.1f}] | {t['delta']:+.0f} [{t['lo']:+.0f}, {t['hi']:+.0f}] "
                         f"({100 * v['tokens_rel']:+.1f}%) | "
                         f"{'WIN' if v['win'] else ('tok' if v['win_tokens'] else '-')} |")
        L.append("")
    L += ["## Flags", ""] + ([f"- {f}" for f in rep["flags"]] or ["- none"])
    (ROOT / "data/distill/report.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
