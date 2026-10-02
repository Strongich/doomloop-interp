#!/usr/bin/env python3
r"""PROTOCOL v3 readout (§V7): primary on mathfresh, per-level breakdowns, SFT->DPO diagnostic.

    uv run python scripts/distill_v3_report.py      # -> data/distill_v3/report.md
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import distill_report as R  # noqa: E402

R.EVAL = ROOT / "data/distill_v3/eval"
LEVELS = {json.loads(l)["question_id"]: int(json.loads(l)["level"])
          for f in ("data/distill_v3/mathtest_fresh1000.jsonl", "data/policy/math500.jsonl")
          for l in open(ROOT / f)}
CONTRASTS = [("dpo_nla", "dpo_natural", "PRIMARY"), ("dpo_nla", "base", "vs Base"),
             ("dpo_natural", "base", "vs Base"), ("sft_nla", "sft_natural", "SFT stage"),
             ("sft_nla", "base", "vs Base"), ("sft_natural", "base", "vs Base"),
             ("dpo_nla", "sft_nla", "DPO effect"), ("dpo_natural", "sft_natural", "DPO effect")]


def main() -> None:
    models = [m for m in ("base", "sft_natural", "sft_nla", "dpo_natural", "dpo_nla")
              if (R.EVAL / m / "rollouts.csv").exists()]
    data = {m: R.load(m) for m in models}
    L = ["# PROTOCOL v3 readout (1.7B)", ""]
    for s in ("mathfresh", "math500", "gsm8k_test"):
        L += [f"## {s}", "", "| model | pass@1 [CI] | mean tokens | median think | capped | doubt |",
              "|---|---|---:|---:|---:|---:|"]
        for m in models:
            if s in data[m]:
                v = R.summarize(data[m][s])
                L.append(f"| {m} | {100*v['acc']:.1f} [{100*v['acc_lo']:.1f}, {100*v['acc_hi']:.1f}] | "
                         f"{v['mean_total_tokens']:.0f} | {v['median_think_tokens']:.0f} | "
                         f"{100*v['capped']:.1f}% | {v['doubt_per_rollout']:.1f} |")
        L.append("")
    for s in ("mathfresh", "math500"):
        L += [f"## Paired contrasts, {s} (Δ = first − second)", "",
              "| contrast | role | level | Δacc pp [CI] | Δtokens (rel) [CI] | win |",
              "|---|---|---|---|---|---|"]
        for a, b, role in CONTRASTS:
            if a not in data or b not in data or s not in data[a] or s not in data[b]:
                continue
            groups = [("all", None)] + [(f"L{k}", k) for k in range(1, 6)]
            for gname, lv in groups:
                da = {q: v for q, v in data[a][s].items() if lv is None or LEVELS[q] == lv}
                db = {q: v for q, v in data[b][s].items() if lv is None or LEVELS[q] == lv}
                p = R.paired(da, db)
                c, t = p["correct"], p["total_tokens"]
                base = t["delta"] / p["tokens_rel"] if p["tokens_rel"] else 1
                L.append(f"| {a} vs {b} | {role} | {gname} | {100*c['delta']:+.1f} [{100*c['lo']:+.1f}, "
                         f"{100*c['hi']:+.1f}] | {100*p['tokens_rel']:+.1f}% [{100*t['lo']/base:+.1f}, "
                         f"{100*t['hi']/base:+.1f}] | {'WIN' if p['win'] and gname == 'all' else ''} |")
        L.append("")
    out = ROOT / "data/distill_v3/report.md"
    out.write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
