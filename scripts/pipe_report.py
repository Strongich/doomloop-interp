#!/usr/bin/env python3
r"""EXPERIMENT-pipeline-sft.md §P5-§P6 readout -> data/pipeline_sft/report.{md,json} + figure.

Question-clustered bootstrap (2,000 reps, seed 20261002): questions resampled, each question
carrying its per-model mean over seeds, so every contrast is paired.
  P6.1  Pipeline (and Gated) vs Base: dtokens CI < 0 and dacc CI lower >= -2pp
  P6.2  Pipeline vs Mix on L4-L5 pooled: dacc CI lower > 0 (tokens reported)
  P6.3  r(L) = tokens Pipeline / tokens Steered at level L; r(L5) - r(L1) >= 0.05, CI > 0
  P6.4  vs Steered-SFT and Short-SFT: dacc, dtokens (no pass/fail)
val (breakage monitor, §P7): a drop > 3pp vs Base flags the run.

    uv run python scripts/pipe_report.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PE, VE = ROOT / "data/pipeline_sft/eval", ROOT / "data/distill/eval"
SEED, REPS = 20261002, 2000
NEW = [f"sft_{a}_{p}" for p in ("lora", "full") for a in ("pipeline", "mix", "gated")]
COMP = ["sft_short_lora", "sft_short_full", "sft_steered_lora", "sft_steered_full"]
NAMES = {"pipeline": "Pipeline-SFT", "mix": "Mix-SFT", "gated": "Gated-SFT",
         "short": "Short-SFT", "steered": "Steered-SFT"}


def rows_of(model: str, s: str) -> list[dict]:
    for root in (PE, VE):
        p = root / model / "rollouts.csv"
        if p.exists():
            rs = [r for r in csv.DictReader(open(p)) if r["set"] == s]
            if rs:
                return rs
    return []


def table(model: str, s: str) -> dict[str, dict]:
    """question -> {correct, tokens, think, capped, noans, doubt, level} means over seeds."""
    by: dict[str, list[dict]] = {}
    for r in rows_of(model, s):
        by.setdefault(r["question_id"], []).append(r)
    out = {}
    for q, rs in by.items():
        f = lambda k: float(np.mean([float(r[k]) for r in rs]))  # noqa: E731
        out[q] = {"correct": f("correct"), "tokens": f("total_tokens"), "capped": f("capped"),
                  "noans": float(np.mean([1 - int(r["has_answer"]) for r in rs])),
                  "doubt": f("doubt_blocks"), "level": rs[0]["level"], "n": len(rs),
                  "think": [float(r["think_tokens"]) for r in rs]}
    return out


def boot(qs: list[str], fn, rng_seed: int = SEED) -> tuple[float, list[float]]:
    rng = np.random.default_rng(rng_seed)
    idx = np.arange(len(qs))
    pt = fn(idx)
    bs = [fn(rng.integers(0, len(qs), len(qs))) for _ in range(REPS)]
    return float(pt), [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def arr(t: dict, qs: list[str], k: str) -> np.ndarray:
    return np.array([t[q][k] for q in qs])


def main() -> None:
    models = ["base", *COMP, *NEW]
    T = {m: table(m, "mathpipe") for m in models}
    have = [m for m in models if T[m]]
    rep: dict = {"models_with_mathpipe": have, "summary": {}, "readings": {}}
    qs_all = sorted(T["base"])
    lv = {q: T["base"][q]["level"] for q in qs_all}
    rep["n_questions"] = len(qs_all)

    def summ(m: str, qs: list[str]) -> dict:
        t = T[m]
        a, ci = boot(qs, lambda i: 100 * arr(t, qs, "correct")[i].mean())
        think = [x for q in qs for x in t[q]["think"]]
        return {"acc": round(a, 2), "acc_ci": [round(x, 2) for x in ci],
                "tokens": round(float(arr(t, qs, "tokens").mean()), 1),
                "median_think": float(np.median(think)),
                "capped": round(float(arr(t, qs, "capped").mean()), 4),
                "noans": round(float(arr(t, qs, "noans").mean()), 4),
                "doubt": round(float(arr(t, qs, "doubt").mean()), 2)}

    levels = sorted({lv[q] for q in qs_all})
    for m in have:
        assert set(T[m]) == set(qs_all), f"{m}: question set differs"
        rep["summary"][m] = {"all": summ(m, qs_all),
                             **{f"L{L}": summ(m, [q for q in qs_all if lv[q] == L]) for L in levels}}

    def contrast(a: str, b: str, qs: list[str]) -> dict:
        da, dci = boot(qs, lambda i: 100 * (arr(T[a], qs, "correct")[i].mean() - arr(T[b], qs, "correct")[i].mean()))
        dt, tci = boot(qs, lambda i: 100 * (arr(T[a], qs, "tokens")[i].mean() / arr(T[b], qs, "tokens")[i].mean() - 1))
        return {"dacc_pp": round(da, 2), "dacc_ci": [round(x, 2) for x in dci],
                "dtokens_pct": round(dt, 2), "dtokens_ci": [round(x, 2) for x in tci]}

    hard = [q for q in qs_all if lv[q] in ("4", "5")]
    for p in ("lora", "full"):
        R: dict = {}
        for arm in ("pipeline", "gated"):
            m = f"sft_{arm}_{p}"
            if m in have:
                c = contrast(m, "base", qs_all)
                c["PASS"] = c["dtokens_ci"][1] < 0 and c["dacc_ci"][0] >= -2
                R[f"P6.1_{arm}_vs_base"] = c
        pm, mm, sm, sh = (f"sft_{x}_{p}" for x in ("pipeline", "mix", "steered", "short"))
        if pm in have and mm in have:
            c = contrast(pm, mm, hard)
            c["PASS"] = c["dacc_ci"][0] > 0
            R["P6.2_pipeline_vs_mix_L45"] = c
        if pm in have and sm in have:
            q1 = [q for q in qs_all if lv[q] == "1"]
            q5 = [q for q in qs_all if lv[q] == "5"]
            rng = np.random.default_rng(SEED)

            def ratio(qs: list[str], i: np.ndarray) -> float:
                return arr(T[pm], qs, "tokens")[i].mean() / arr(T[sm], qs, "tokens")[i].mean()

            pt = ratio(q5, np.arange(len(q5))) - ratio(q1, np.arange(len(q1)))
            bs = [ratio(q5, rng.integers(0, len(q5), len(q5))) - ratio(q1, rng.integers(0, len(q1), len(q1)))
                  for _ in range(REPS)]
            ci = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
            R["P6.3_ratio_L5_minus_L1"] = {"r_L1": round(ratio(q1, np.arange(len(q1))), 4),
                                           "r_L5": round(ratio(q5, np.arange(len(q5))), 4),
                                           "diff": round(pt, 4), "ci": [round(x, 4) for x in ci],
                                           "PASS": pt >= 0.05 and ci[0] > 0}
        for ref in (sm, sh):
            for arm in ("pipeline", "mix", "gated"):
                m = f"sft_{arm}_{p}"
                if m in have and ref in have:
                    R[f"P6.4_{arm}_vs_{ref.split('_')[1]}"] = contrast(m, ref, qs_all)
        rep["readings"][p] = R

    # secondary sets for the new models (+ v2 numbers of comparators / base, same engine)
    sec = {}
    for s in ("math500", "confirm400", "gsm8k_test", "aime_amc", "val"):
        sec[s] = {}
        for m in ["base", *COMP, *NEW]:
            t = table(m, s)
            if t:
                qs = sorted(t)
                sec[s][m] = {"acc": round(100 * float(arr(t, qs, "correct").mean()), 2),
                             "tokens": round(float(arr(t, qs, "tokens").mean()), 1),
                             "capped": round(float(arr(t, qs, "capped").mean()), 4), "n_q": len(qs)}
    rep["secondary"] = sec
    if "base" in sec["val"]:
        rep["val_flags"] = {m: round(v["acc"] - sec["val"]["base"]["acc"], 2) < -3
                            for m, v in sec["val"].items() if m in NEW}
    (ROOT / "data/pipeline_sft/report.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)) + "\n")

    # figure: tokens vs accuracy on mathpipe (+ v2 DPO on MATH-500)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    for m in have:
        s = rep["summary"][m]["all"]
        mk = "s" if m.endswith("full") else ("o" if m.endswith("lora") else "*")
        ax[0].scatter(s["tokens"], s["acc"], marker=mk, s=60)
        ax[0].annotate(m.replace("sft_", ""), (s["tokens"], s["acc"]), fontsize=7)
    ax[0].set(xlabel="mean total tokens", ylabel="pass@1 (%)", title="mathtest_pipe1000")
    for m in ["base", *COMP, "dpo_natural_lora", "dpo_steered_lora", "dpo_natural_full", "dpo_steered_full", *NEW]:
        t = table(m, "math500")
        if t:
            qs = sorted(t)
            x, y = float(arr(t, qs, "tokens").mean()), 100 * float(arr(t, qs, "correct").mean())
            ax[1].scatter(x, y, marker="s" if m.endswith("full") else "o", s=50)
            ax[1].annotate(m.replace("sft_", ""), (x, y), fontsize=7)
    ax[1].set(xlabel="mean total tokens", ylabel="pass@1 (%)", title="MATH-500 (incl. v2 DPO)")
    fig.tight_layout()
    fig.savefig(ROOT / "data/pipeline_sft/frontier.png", dpi=130)

    md = ["# Pipeline-SFT readout (mathtest_pipe1000)", "",
          "| model | acc | 95% CI | tokens | median think | capped | no-ans | doubt |", "|---|---|---|---|---|---|---|---|"]
    for m in have:
        s = rep["summary"][m]["all"]
        md.append(f"| {m} | {s['acc']} | {s['acc_ci']} | {s['tokens']} | {s['median_think']} | "
                  f"{s['capped']} | {s['noans']} | {s['doubt']} |")
    md += ["", "## Per level (acc / tokens)", "", "| model | " + " | ".join(f"L{L}" for L in levels) + " |",
           "|---|" + "---|" * len(levels)]
    for m in have:
        md.append(f"| {m} | " + " | ".join(f"{rep['summary'][m][f'L{L}']['acc']} / {rep['summary'][m][f'L{L}']['tokens']:.0f}"
                                           for L in levels) + " |")
    for p, R in rep["readings"].items():
        md += ["", f"## Readings ({p})", ""]
        md += [f"- **{k}**: {json.dumps(v, default=lambda o: o.item())}" for k, v in R.items()]
    md += ["", "## Secondary sets (acc / tokens)", ""]
    for s, d in sec.items():
        md.append(f"- **{s}**: " + "; ".join(f"{m} {v['acc']} / {v['tokens']:.0f}" for m, v in d.items()))
    if "val_flags" in rep:
        md += ["", f"val breakage flags (>3pp below Base): {rep['val_flags']}"]
    (ROOT / "data/pipeline_sft/report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
