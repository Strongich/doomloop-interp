#!/usr/bin/env python3
r"""EXPERIMENT-probe-gated.md §G8 readout -> data/gated/report.{json,md}.

On mathtest_gate400 (L4 + L5 pooled), question-clustered paired bootstrap (2,000 reps,
seed 20261003), each question carrying its mean over 4 seeds:
  saving(X)   = 100 * (1 - tokens_X / tokens_base)                      (points)
  dacc(X, Y)  = 100 * (acc_X - acc_Y)                                   (pp)
  dtok(X, Y)  = saving(X) - saving(Y)                                   (points of base)
  primary     G10 vs R10: dacc CI lower > 0 AND |dtok| <= 3
  claim       G10 vs base: dacc CI lower >= -2 AND saving CI > 0
  dose        G5/R5, G20/R20 the same way (no pass/fail)
  match       realized unsteered share (pooled) of each G / R pair; |diff| > 2 points flagged
Mechanism: productive / redundant episodes per rollout (episodes_mech.jsonl).
Easy levels: MATH-500 L1-3 and GSM8K saving of each arm next to N's, and the share unsteered.
AIME2025 + AMC23: base, N, G10, R10.

    uv run python scripts/gate_report.py
"""

from __future__ import annotations

import collections
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
G = ROOT / "data/gated"
SEED, REPS = 20261003, 2000
N = "N@a1.0d256"


def load(*patterns: str) -> list[dict]:
    return [r for p in patterns for f in sorted(glob.glob(str(G / p))) for r in csv.DictReader(open(f))]


def cube(rows: list[dict], arms: list[str]):
    """question x arm arrays of per-question seed means (correct, tokens)."""
    qs = sorted({r["question_id"] for r in rows})
    qi = {q: i for i, q in enumerate(qs)}
    acc = {a: collections.defaultdict(list) for a in arms}
    tok = {a: collections.defaultdict(list) for a in arms}
    for r in rows:
        if r["policy"] in acc:
            acc[r["policy"]][r["question_id"]].append(float(r["correct"]))
            tok[r["policy"]][r["question_id"]].append(float(r["total_tokens"]))
    A = np.full((len(qs), len(arms)), np.nan)
    T = np.full((len(qs), len(arms)), np.nan)
    n = {}
    for j, a in enumerate(arms):
        n[a] = sum(len(v) for v in acc[a].values())
        for q, v in acc[a].items():
            A[qi[q], j] = np.mean(v)
            T[qi[q], j] = np.mean(tok[a][q])
    return qs, A, T, n


def share(rows: list[dict], arm: str) -> float | None:
    xs = [r for r in rows if r["policy"] == arm and r.get("unsteered_sites", "") != ""]
    e = sum(int(r["eligible_sites"]) for r in xs)
    return round(100 * sum(int(r["unsteered_sites"]) for r in xs) / e, 2) if e else None


def ci(v) -> list[float]:
    return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]


def main() -> None:
    cal = json.loads((G / "calibration.json").read_text())
    R = {q: cal[q]["policy"] for q in ("5", "10", "20")}
    arms = ["base", N, "N@a0.5d256", *(f"{N}G{q}" for q in ("5", "10", "20")), *R.values()]
    rows = load("eval/gate400_s*/rollouts.csv", "eval/gate400R_s*/rollouts.csv")
    qs, A, T, nroll = cube(rows, arms)
    assert not np.isnan(A).any(), "missing cells"
    ix = {a: j for j, a in enumerate(arms)}
    rng = np.random.default_rng(SEED)
    boots = [rng.integers(0, len(qs), len(qs)) for _ in range(REPS)]
    allq = np.arange(len(qs))
    b0 = ix["base"]

    def acc(i, a):
        return 100 * A[i, ix[a]].mean()

    def sav(i, a):
        return 100 * (1 - T[i, ix[a]].mean() / T[i, b0].mean())

    rep: dict = {"questions": len(qs), "rollouts": nroll, "random_arms": R}
    rep["arms"] = {a: {"acc": round(acc(allq, a), 2), "acc_ci": ci([acc(i, a) for i in boots]),
                       "tokens": round(float(T[:, ix[a]].mean()), 1),
                       "saving": round(sav(allq, a), 2), "saving_ci": ci([sav(i, a) for i in boots]),
                       "unsteered_share": share(rows, a)} for a in arms}

    def contrast(x: str, y: str) -> dict:
        da = lambda i: acc(i, x) - acc(i, y)  # noqa: E731
        dt = lambda i: sav(i, x) - sav(i, y)  # noqa: E731
        return {"dacc_pp": round(da(allq), 2), "dacc_ci": ci([da(i) for i in boots]),
                "dsaving_pts": round(dt(allq), 2), "dsaving_ci": ci([dt(i) for i in boots])}

    pairs = {}
    for q in ("5", "10", "20"):
        c = contrast(f"{N}G{q}", R[q])
        su, sr = share(rows, f"{N}G{q}"), share(rows, R[q])
        c["unsteered_G_R"] = [su, sr]
        c["share_mismatch_gt2"] = abs(su - sr) > 2
        if q == "10":
            c["PASS"] = c["dacc_ci"][0] > 0 and abs(c["dsaving_pts"]) <= 3
        pairs[f"G{q}_vs_R{q}"] = c
    rep["probe_vs_random"] = pairs
    vb = contrast(f"{N}G10", "base")
    g10 = rep["arms"][f"{N}G10"]
    vb["saving_ci"] = g10["saving_ci"]
    vb["NONINFERIOR"] = vb["dacc_ci"][0] >= -2 and g10["saving_ci"][0] > 0
    rep["G10_vs_base"] = vb
    rep["vs_N"] = {a: contrast(a, N) for a in arms if a not in ("base", N)}
    lv = {json.loads(x)["question_id"]: json.loads(x)["level"] for x in open(G / "mathtest_gate400.jsonl")}
    rep["per_level"] = {}
    for L in (4, 5):
        ii = np.array([j for j, q in enumerate(qs) if lv[q] == L])
        rep["per_level"][f"L{L}"] = {a: {"acc": round(acc(ii, a), 2), "saving": round(sav(ii, a), 2)}
                                     for a in arms}
    prim = pairs["G10_vs_R10"]
    if prim["PASS"] and vb["NONINFERIOR"]:
        rep["outcome"] = (f"Gating N by a linear probe of the same state keeps error-catching doubt: at "
                          f"the same steering amount it is {prim['dacc_pp']} pp more accurate than random "
                          f"gating on hard problems, and non-inferior to unsteered reasoning at "
                          f"{g10['saving']}% fewer tokens.")
    elif prim["PASS"]:
        rep["outcome"] = (f"The probe selects useful doubt better than chance (+{prim['dacc_pp']} pp at "
                          "matched steering), but not enough to close the gap to unsteered accuracy.")
    else:
        auc = json.loads((G / "probe_report.json").read_text())["heldout_auc"]
        rep["outcome"] = ("Probe gating is no better than steering less at random. The state signal "
                          f"(AUC {auc}) is too weak at deployment to select checks.")

    # secondary sets
    sec = {}
    for name, pats in {"math500_L1-3": ["eval/math500L13_s*/rollouts.csv"],
                       "gsm8k_test": ["eval/gsm8k_s*/rollouts.csv"],
                       "aime_amc": ["eval/aime_s*/rollouts.csv", "eval/aimeR_s*/rollouts.csv"]}.items():
        rs = load(*pats)
        if not rs:
            continue
        sarms = [a for a in arms if any(r["policy"] == a for r in rs)]
        q2, A2, T2, n2 = cube(rs, sarms)
        base_t = T2[:, sarms.index("base")].mean()
        sec[name] = {a: {"acc": round(100 * float(np.nanmean(A2[:, j])), 2),
                         "tokens": round(float(np.nanmean(T2[:, j])), 1),
                         "saving": round(100 * (1 - float(np.nanmean(T2[:, j])) / base_t), 2),
                         "unsteered_share": share(rs, a), "rollouts": n2[a]}
                     for j, a in enumerate(sarms)}
    rep["secondary"] = sec

    mech = G / "episodes_mech.jsonl"
    if mech.exists():
        nr = collections.Counter(json.loads(x)["policy"] for x in open(G / "mech_select.jsonl"))
        cnt: dict = collections.defaultdict(collections.Counter)
        for x in open(mech):
            e = json.loads(x)
            cnt[e["policy"]][e["label"]] += 1
        rep["mechanism"] = {p: {"rollouts": nr[p], **{f"{lab}_per_rollout": round(cnt[p][lab] / nr[p], 3)
                                                     for lab in ("productive", "redundant", "harmful", "failed", "other")},
                                "doubt_episodes_per_rollout": round(sum(cnt[p].values()) / nr[p], 2)}
                            for p in nr}
    (G / "report.json").write_text(json.dumps(rep, indent=1, default=lambda o: o.item()) + "\n")

    md = ["# Probe-gated steering readout (mathtest_gate400, L4+L5)", "", f"**Outcome:** {rep['outcome']}", "",
          "| arm | acc | 95% CI | tokens | saving | 95% CI | unsteered % |", "|---|---|---|---|---|---|---|"]
    for a, s in rep["arms"].items():
        md.append(f"| {a} | {s['acc']} | {s['acc_ci']} | {s['tokens']} | {s['saving']} | {s['saving_ci']} | "
                  f"{s['unsteered_share']} |")
    md += ["", "## Probe vs random (matched)", ""]
    md += [f"- **{k}**: {json.dumps(v, default=lambda o: o.item())}" for k, v in pairs.items()]
    md += ["", f"- **G10 vs base**: {json.dumps(vb, default=lambda o: o.item())}", ""]
    md += ["## Secondary", ""] + [f"- **{k}**: " + "; ".join(
        f"{a} {v['acc']} / {v['tokens']:.0f} (sav {v['saving']}, unsteered {v['unsteered_share']})"
        for a, v in d.items()) for k, d in sec.items()]
    if "mechanism" in rep:
        md += ["", "## Mechanism (per rollout)", ""] + [f"- **{p}**: {json.dumps(v)}" for p, v in rep["mechanism"].items()]
    (G / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
