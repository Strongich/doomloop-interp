#!/usr/bin/env python3
"""EXPERIMENT-vers-doubt.md §5: P (and −N) vs base on hard128 × 8 and gate400 × 4.

Base and N@a1.0d256 are reused from Part C / Part A runs (data/baselines/c17_s*,
a17_gate_s*). Everything is regraded with the D69 grader from rollouts.jsonl. CIs are a
question-clustered bootstrap (2,000 reps, seed 20261005), one resample per set, shared by
every quantity so that differences between arms are paired.

    python scripts/vers_report.py        # -> data/vers/report.{md,json}
"""

from __future__ import annotations

import glob
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

REPS, SEED = 2000, 20261005
CJK = re.compile(r"[一-鿿]")
SUBSET = {"aime2024", "hmmt_feb2025"}
V, B = "data/vers", "data/baselines"
N_POL = "N@a1.0d256"


def _load(path: str) -> pd.DataFrame:
    from reasoning_attention.grading import grade

    rows = []
    for line in open(path):
        r = json.loads(line)
        think = r["text"].split("</think>")[0]
        rows.append({
            "question_id": r["question_id"], "dataset": r["dataset"], "policy": r["policy"],
            "seed": int(r["seed"]), "correct": int(grade(r["text"], r["gold"]).is_correct),
            "total_tokens": int(r["total_tokens"]), "capped": int(r["capped"]),
            "doubt_blocks": int(r["doubt_blocks"]),
            "lang": int(len(CJK.findall(think)) >= 20),
        })
    return pd.DataFrame(rows)


def load(pattern: str, policy: str | None) -> pd.DataFrame | None:
    paths = [str(Path(d) / "rollouts.jsonl") for d in sorted(glob.glob(str(ROOT / pattern)))
             if (Path(d) / "rollouts.csv").exists()]
    if not paths:
        return None
    with ProcessPoolExecutor(8) as ex:
        df = pd.concat(ex.map(_load, paths))
    if policy:
        df = df[df.policy == policy]
    elif df.policy.nunique() != 1:
        raise SystemExit(f"{pattern}: several policies {df.policy.unique()}")
    if df.duplicated(["question_id", "seed"]).any():
        raise SystemExit(f"{pattern}: duplicate (question, seed)")
    return df


def ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


class Set:
    def __init__(self, base: pd.DataFrame, arms: dict[str, pd.DataFrame]):
        self.base = base
        self.q = sorted(base.question_id.unique())
        self.idx = np.random.default_rng(SEED).integers(0, len(self.q), (REPS, len(self.q)))
        self.arms = arms

    def qm(self, arm: pd.DataFrame) -> pd.DataFrame:
        m = arm.merge(self.base, on=["question_id", "seed"], suffixes=("", "_b"))
        if len(m) != len(arm):
            raise SystemExit("arm rows without a base pair")
        return m.groupby("question_id").mean(numeric_only=True).reindex(self.q), len(m)

    def stats(self, name: str) -> dict:
        arm = self.arms[name]
        qm, n = self.qm(arm)
        if qm.isna().any().any():
            raise SystemExit(f"{name}: questions without rows")
        d = (qm.correct - qm.correct_b).to_numpy() * 100
        ta, tb = qm.total_tokens.to_numpy(), qm.total_tokens_b.to_numpy()
        dd = (qm.doubt_blocks - qm.doubt_blocks_b).to_numpy()
        ds = self.base.drop_duplicates("question_id").set_index("question_id").dataset
        sub = ds.reindex(self.q).isin(SUBSET).to_numpy()
        out = {
            "n_pairs": n, "n_q": len(self.q),
            "acc": qm.correct.mean() * 100, "base_acc": qm.correct_b.mean() * 100,
            "dacc": d.mean(), "dacc_ci": ci(d[self.idx].mean(1)),
            "length": ta.sum() / tb.sum() - 1,
            "length_ci": ci(ta[self.idx].sum(1) / tb[self.idx].sum(1) - 1),
            "tokens": ta.mean(), "base_tokens": tb.mean(),
            "capped": qm.capped.mean() * 100, "base_capped": qm.capped_b.mean() * 100,
            "doubt": qm.doubt_blocks.mean(), "base_doubt": qm.doubt_blocks_b.mean(),
            "ddoubt": dd.mean(), "ddoubt_ci": ci(dd[self.idx].mean(1)),
            "lang": qm.lang.mean() * 100, "base_lang": qm.lang_b.mean() * 100,
            "_d": d[self.idx].mean(1), "_dd": dd[self.idx].mean(1),
            "_len": ta[self.idx].sum(1) / tb[self.idx].sum(1) - 1,
        }
        if sub.any():
            # resample whole questions, keep only the subset's within each draw
            ws = sub[self.idx]
            boot = (d[self.idx] * ws).sum(1) / np.maximum(ws.sum(1), 1)
            out.update(sub_n=int(sub.sum()), sub_dacc=float(d[sub].mean()), sub_dacc_ci=ci(boot),
                       sub_acc=float(qm.correct.to_numpy()[sub].mean() * 100),
                       sub_base_acc=float(qm.correct_b.to_numpy()[sub].mean() * 100))
        # rescue (hard128, 8 seeds): base right on <= 2 seeds and arm on >= 4, vs the reverse
        k = arm.groupby("question_id").correct.sum().reindex(self.q)
        kb = self.base.groupby("question_id").correct.sum().reindex(self.q)
        if arm.seed.nunique() == 8:
            out["rescued"] = int(((kb <= 2) & (k >= 4)).sum())
            out["lost"] = int(((k <= 2) & (kb >= 4)).sum())
        return out


def diff(x: dict, y: dict) -> dict:
    return {"dacc": x["dacc"] - y["dacc"], "dacc_ci": ci(x["_d"] - y["_d"]),
            "ddoubt": x["ddoubt"] - y["ddoubt"], "ddoubt_ci": ci(x["_dd"] - y["_dd"]),
            "length": x["length"] - y["length"], "length_ci": ci(x["_len"] - y["_len"])}


def fmt(name: str, s: dict) -> str:
    sub = (f" | AIME24+HMMT Δ {s['sub_dacc']:+.1f} [{s['sub_dacc_ci'][0]:+.1f},"
           f"{s['sub_dacc_ci'][1]:+.1f}]" if "sub_dacc" in s else "")
    res = f" | rescued {s['rescued']} / lost {s['lost']}" if "rescued" in s else ""
    return (f"| {name} | {s['n_pairs']} | {s['acc']:.1f} vs {s['base_acc']:.1f} | "
            f"{s['dacc']:+.1f} [{s['dacc_ci'][0]:+.1f}, {s['dacc_ci'][1]:+.1f}] | "
            f"{s['length']:+.1%} [{s['length_ci'][0]:+.1%}, {s['length_ci'][1]:+.1%}] | "
            f"{s['capped']:.1f} vs {s['base_capped']:.1f} | "
            f"{s['doubt']:.2f} vs {s['base_doubt']:.2f} ({s['ddoubt']:+.2f} "
            f"[{s['ddoubt_ci'][0]:+.2f}, {s['ddoubt_ci'][1]:+.2f}]) | "
            f"{s['lang']:.1f} vs {s['base_lang']:.1f} |{sub}{res}")


HEAD = ("| arm | pairs | acc % (arm vs base) | Δacc pp [95% CI] | length change [CI] | capped % |"
        " doubt/rollout (Δ [CI]) | lang switch % | extra |\n|---|---|---|---|---|---|---|---|---|")


def main() -> None:
    sel = (ROOT / V / "selected.txt").read_text().strip()
    lines = ["# VERS +doubt — report", "",
             f"Selected policy: `{sel}`. D69 grading; question-clustered bootstrap "
             f"({REPS} reps, seed {SEED}). Length change = tokens(arm)/tokens(base) − 1.", ""]
    res: dict = {"selected": sel}
    sj = ROOT / V / "select.json"
    if sj.exists():
        lines += ["## Dev (dev400 × 2, 16k cap)", "", "```",
                  json.dumps(json.loads(sj.read_text()), indent=1), "```", ""]
    sets = {
        "hard128 × 8": (f"{B}/c17_s*", {"N@a1.0 (ref)": (f"{B}/c17_s*", N_POL),
                                        "P": (f"{V}/h128_s*", None),
                                        "−N": (f"{V}/negN_s*", None)}),
        "gate400 × 4": (f"{B}/a17_gate_s*", {"N@a1.0 (ref)": (f"{B}/a17_gate_s*", N_POL),
                                            "P": (f"{V}/gate_s*", None)}),
    }
    for title, (bpat, arms) in sets.items():
        base = load(bpat, "base")
        loaded = {k: load(p, pol) for k, (p, pol) in arms.items()}
        loaded = {k: v for k, v in loaded.items() if v is not None}
        if "P" in loaded:  # restrict base to the arm's seeds (partial runs)
            base = base[base.seed.isin(loaded["P"].seed.unique())]
        s = Set(base, loaded)
        st = {k: s.stats(k) for k in loaded}
        lines += [f"## {title}", "", HEAD, *[fmt(k, v) for k, v in st.items()], ""]
        if "P" in st and "−N" in st:
            dd = diff(st["P"], st["−N"])
            lines += [f"P − (−N): Δacc {dd['dacc']:+.1f}pp [{dd['dacc_ci'][0]:+.1f}, "
                      f"{dd['dacc_ci'][1]:+.1f}]; doubt {dd['ddoubt']:+.2f} "
                      f"[{dd['ddoubt_ci'][0]:+.2f}, {dd['ddoubt_ci'][1]:+.2f}]; length "
                      f"{dd['length']:+.1%} [{dd['length_ci'][0]:+.1%}, {dd['length_ci'][1]:+.1%}]",
                      ""]
            st["P_minus_negN"] = dd
        res[title] = {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                      for k, v in st.items()}
    (ROOT / V / "report.md").write_text("\n".join(lines) + "\n")
    (ROOT / V / "report.json").write_text(json.dumps(res, indent=2, default=float))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
