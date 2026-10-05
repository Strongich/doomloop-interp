#!/usr/bin/env python3
r"""PROTOCOL v4 (8B) readout, §M6-§M7 -> data/distill_8b/report.{md,json}.

Contrasts are paired over (question, seed): every model has the same request seeds, so
each question carries its per-model seed mean; question-clustered bootstrap, 2,000 reps,
seed 20261003. Δtokens = relative difference of mean total tokens (%).
  win(X vs Y) on a set  = Δtokens CI upper < 0  and  Δacc CI lower >= -2pp
  a win must hold on BOTH primary sets (mathfresh, mathgate)
Readings 1-5 and the outcome wording as pre-stated in §M7, per method (SFT, DPO).
The online teacher (base + NU26@a0.75d256 through the steering runner) is the reference of
reading 4: its Δacc is against the runner's own base.

    uv run python scripts/distill8b_report.py
"""

from __future__ import annotations

import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/distill_8b"
SEED, REPS = 20261003, 2000
PRIMARY = ("mathfresh", "mathgate")
SETS = ("mathfresh", "mathgate", "aime_amc", "math500", "gsm8k_test", "val_math")
MODELS = ["base", "sft_short_lora", "sft_steered_lora", "dpo_natural_lora", "dpo_steered_lora"]
NAMES = {"base": "Base", "sft_short_lora": "Short-SFT", "sft_steered_lora": "Steered-SFT",
         "dpo_natural_lora": "DPO-natural", "dpo_steered_lora": "DPO-steered",
         "online_base": "online Base (runner)", "online_teacher": "online teacher NU26@a0.75"}
TEACHER = "NU26@a0.75d256"


def levels() -> dict[str, int]:
    out = {}
    for f in ("data/distill_v3/mathtest_fresh1000.jsonl", "data/gated/mathtest_gate400.jsonl",
              "data/policy/math500.jsonl", "data/distill_8b/val_math150.jsonl"):
        for x in open(ROOT / f):
            r = json.loads(x)
            out[r["question_id"]] = int(r["level"])
    return out


def load_eval(name: str) -> dict[str, list[dict]]:
    by: dict[str, list[dict]] = {}
    seen: set = set()
    for f in sorted(glob.glob(str(D / f"eval/{name}__*/rollouts.csv"))
                    + glob.glob(str(D / f"eval_split/{name}__*/rollouts.csv"))):
        for r in csv.DictReader(open(f)):
            k = (r["set"], r["question_id"], r["seed"])
            assert k not in seen, f"{name}: duplicate rollout {k}"
            seen.add(k)
            by.setdefault(r["set"], []).append(r)
    return by


def load_online() -> dict[str, dict[str, list[dict]]]:
    out: dict[str, dict[str, list[dict]]] = {"online_base": {}, "online_teacher": {}}
    for f in sorted(glob.glob(str(D / "online/s*/rollouts.csv"))):
        for r in csv.DictReader(open(f)):
            m = "online_base" if r["policy"] == "base" else "online_teacher"
            s = "mathfresh" if r["question_id"].startswith("mathfresh:") else "mathgate"
            out[m].setdefault(s, []).append(r)
    return out


def table(rows: list[dict]) -> dict[str, dict]:
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["question_id"], []).append(r)
    return {q: {"correct": np.mean([float(r["correct"]) for r in rs]),
                "tokens": np.mean([float(r["total_tokens"]) for r in rs]),
                "capped": np.mean([float(r["capped"]) for r in rs]),
                "noans": np.mean([1 - int(r["has_answer"]) for r in rs]),
                "doubt": np.mean([float(r["doubt_blocks"]) for r in rs]),
                "think": [float(r["think_tokens"]) for r in rs], "n": len(rs)} for q, rs in by.items()}


def ci(v) -> list[float]:
    return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]


def boots(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    return [rng.integers(0, n, n) for _ in range(REPS)]


def summary(t: dict, qs: list[str]) -> dict:
    c = np.array([t[q]["correct"] for q in qs])
    B = boots(len(qs))
    return {"acc": round(100 * c.mean(), 2), "acc_ci": ci([100 * c[i].mean() for i in B]),
            "tokens": round(float(np.mean([t[q]["tokens"] for q in qs])), 1),
            "median_think": float(np.median([x for q in qs for x in t[q]["think"]])),
            "capped": round(100 * float(np.mean([t[q]["capped"] for q in qs])), 2),
            "noans": round(100 * float(np.mean([t[q]["noans"] for q in qs])), 2),
            "doubt": round(float(np.mean([t[q]["doubt"] for q in qs])), 2),
            "questions": len(qs), "rollouts": int(sum(t[q]["n"] for q in qs))}


def contrast(ta: dict, tb: dict, qs: list[str]) -> dict:
    a_c = np.array([ta[q]["correct"] for q in qs]); b_c = np.array([tb[q]["correct"] for q in qs])
    a_t = np.array([ta[q]["tokens"] for q in qs]); b_t = np.array([tb[q]["tokens"] for q in qs])
    B = boots(len(qs))
    da = lambda i: 100 * (a_c[i].mean() - b_c[i].mean())  # noqa: E731
    dt = lambda i: 100 * (a_t[i].mean() / b_t[i].mean() - 1)  # noqa: E731
    al = np.arange(len(qs))
    out = {"dacc_pp": round(da(al), 2), "dacc_ci": ci([da(i) for i in B]),
           "dtokens_pct": round(dt(al), 2), "dtokens_ci": ci([dt(i) for i in B])}
    out["tokens_lower"] = out["dtokens_ci"][1] < 0
    out["noninferior"] = out["dacc_ci"][0] >= -2
    out["WIN"] = out["tokens_lower"] and out["noninferior"]
    return out


def main() -> None:
    lv = levels()
    T = {m: {s: table(rs) for s, rs in load_eval(m).items()} for m in MODELS}
    T.update({m: {s: table(rs) for s, rs in d.items()} for m, d in load_online().items()})
    have = [m for m in T if T[m]]
    rep: dict = {"models": have, "summary": {}, "by_level": {}}
    for m in have:
        rep["summary"][m] = {}
        for s, t in T[m].items():
            qs = sorted(t)
            rep["summary"][m][s] = summary(t, qs)
            if s in ("mathfresh", "mathgate", "math500"):
                rep["by_level"].setdefault(s, {})[m] = {
                    f"L{L}": summary(t, [q for q in qs if lv[q] == L])
                    for L in sorted({lv[q] for q in qs})}

    def c(a: str, b: str, s: str) -> dict | None:
        if s not in T.get(a, {}) or s not in T.get(b, {}):
            return None
        assert set(T[a][s]) == set(T[b][s]), f"{a}/{b} {s}: question sets differ"
        return contrast(T[a][s], T[b][s], sorted(T[a][s]))

    def both(a: str, b: str) -> dict:
        r = {s: c(a, b, s) for s in PRIMARY}
        r["WIN_both"] = all(r[s] and r[s]["WIN"] for s in PRIMARY)
        return r

    R: dict = {}
    pairs = {"sft": ("sft_steered_lora", "sft_short_lora"), "dpo": ("dpo_steered_lora", "dpo_natural_lora")}
    for m, (nla, nat) in pairs.items():
        R[f"1_{m}_weak_{nla}_vs_base"] = both(nla, "base")
        R[f"2_{m}_strong_{nla}_vs_{nat}"] = both(nla, nat)
    # 3: the shortest natural arm that is non-inferior to Base on both primary sets
    nat_ok = []
    for nat in ("sft_short_lora", "dpo_natural_lora"):
        vb = both(nat, "base")
        if all(vb[s] and vb[s]["noninferior"] for s in PRIMARY):
            nat_ok.append((np.mean([rep["summary"][nat][s]["tokens"] for s in PRIMARY]), nat))
    if nat_ok:
        best = min(nat_ok)[1]
        R["3_best_natural"] = {"arm": best, **{f"{nla}_vs_{best}": both(nla, best)
                                               for nla in ("sft_steered_lora", "dpo_steered_lora")}}
    else:
        R["3_best_natural"] = {"arm": None, "void": "no natural arm is non-inferior to Base on both"}
    # 4: diagnostic
    diag = {}
    for s in PRIMARY:
        diag[s] = {"online_teacher_vs_online_base": c("online_teacher", "online_base", s),
                   **{f"{m}_vs_base": c(m, "base", s) for m in MODELS[1:]}}
    R["4_diagnostic"] = diag
    # 5: AIME/AMC
    R["5_aime_amc"] = {m: {**rep["summary"][m].get("aime_amc", {}),
                           "vs_base": c(m, "base", "aime_amc")} for m in MODELS if "aime_amc" in T[m]}
    for m, v in R["5_aime_amc"].items():
        if v.get("vs_base"):
            v["drop_gt_5pp"] = v["vs_base"]["dacc_pp"] < -5
    # secondary contrasts on MATH-500 / GSM8K
    rep["secondary"] = {s: {f"{a}_vs_{b}": c(a, b, s) for a, b in
                            [("sft_steered_lora", "base"), ("dpo_steered_lora", "base"),
                             ("sft_steered_lora", "sft_short_lora"), ("dpo_steered_lora", "dpo_natural_lora"),
                             ("sft_short_lora", "base"), ("dpo_natural_lora", "base")]}
                        for s in ("math500", "gsm8k_test")}
    if "val_math" in T["base"]:
        rep["val_flags"] = {m: (rep["summary"][m]["val_math"]["acc"] - rep["summary"]["base"]["val_math"]["acc"]) < -3
                            for m in MODELS[1:] if "val_math" in T[m]}

    words = {}
    for m, (nla, nat) in pairs.items():
        w = R[f"1_{m}_weak_{nla}_vs_base"]
        s2 = R[f"2_{m}_strong_{nla}_vs_{nat}"]
        if w["WIN_both"]:
            pct = round(-np.mean([w[s]["dtokens_pct"] for s in PRIMARY]), 1)
            if s2["WIN_both"]:
                words[m] = (f"Offline NLA-steered traces distil into the 8B: a hook-free student reasons "
                            f"{pct}% shorter than Base at non-inferior accuracy, and shorter than a student "
                            "trained on its best natural traces.")
            elif not all(s2[s]["tokens_lower"] for s in PRIMARY):
                words[m] = "The benefit is yield: one steered rollout gives what best-of-4 natural sampling gives."
            else:
                y = min(s2[s]["dacc_pp"] for s in PRIMARY)
                words[m] = (f"NLA data shortens the student more than natural data, at {y} pp; both are "
                            "non-inferior to Base.")
        elif w["mathfresh"] and w["mathfresh"]["WIN"]:
            words[m] = (f"Distillation works on easy and mid problems; on hard ones the student loses "
                        f"{w['mathgate']['dacc_pp']} pp, unlike its teacher.")
        else:
            words[m] = ("Offline distillation does not carry the steering effect into the 8B at base "
                        f"accuracy. Size: mathfresh {w['mathfresh']}, mathgate {w['mathgate']}.")
    rep["readings"] = R
    rep["outcome"] = words
    (D / "report.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.item()) + "\n")

    md = ["# PROTOCOL v4 readout (Qwen3-8B)", ""]
    for m, w in words.items():
        md.append(f"**Outcome ({m.upper()}):** {w}")
    for s in SETS:
        md += ["", f"## {s}", "", "| model | pass@1 [CI] | mean tokens | median think | capped % | no-ans % | doubt |",
               "|---|---|---:|---:|---:|---:|---:|"]
        for m in have:
            v = rep["summary"][m].get(s)
            if v:
                md.append(f"| {NAMES[m]} | {v['acc']} {v['acc_ci']} | {v['tokens']:.0f} | {v['median_think']:.0f} | "
                          f"{v['capped']} | {v['noans']} | {v['doubt']} |")
    for s, d in rep["by_level"].items():
        md += ["", f"## {s} by level (acc / tokens)", "", "| model | " + " | ".join(next(iter(d.values()))) + " |",
               "|---|" + "---|" * len(next(iter(d.values())))]
        for m, lvls in d.items():
            md.append(f"| {NAMES[m]} | " + " | ".join(f"{v['acc']} / {v['tokens']:.0f}" for v in lvls.values()) + " |")
    md += ["", "## Readings (§M7)", ""]
    md += [f"- **{k}**: `{json.dumps(v, default=lambda o: o.item())}`" for k, v in R.items()]
    md += ["", "## Secondary contrasts", ""]
    md += [f"- **{s} {k}**: `{json.dumps(v, default=lambda o: o.item())}`" for s, d in rep["secondary"].items()
           for k, v in d.items()]
    if "val_flags" in rep:
        md += ["", f"val breakage flags (> 3pp below Base): {rep['val_flags']}"]
    (D / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
