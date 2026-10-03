#!/usr/bin/env python3
r"""EXPERIMENT-selective-doubt.md §S6-§S7 readout.

On mathtest_sel400 (L4-L5 pooled), per arm: acc, mean total tokens,
  saving(X) = 100 * (1 - tokens_X / tokens_base)          (points)
  dacc(X)   = 100 * (acc_X - acc_base)                     (pp)
  N curve   = the N alpha points (a0.5, a0.75, a1.0), sorted by saving, linear interpolation
              between the two bracketing X's saving; outside their range -> "outside N's curve"
              (secondary variant: base added as the alpha = 0 point)
  excess(X) = dacc(X) - dacc_N(saving(X))
All of it inside one question-clustered bootstrap (2,000 reps, seed 20261002); a rep in
which X falls outside the curve is dropped and the dropped share reported.
Primary (§S7): excess(N_red) - excess(N_all). Win: its CI > 0, excess(N_red) CI > 0, and
saving(N_red) >= saving(N@a1.0) - 10.
Mechanism check (if episodes_mech.jsonl exists): productive / redundant episodes per rollout.

    uv run python scripts/sel_report.py
"""

from __future__ import annotations

import collections
import csv
import glob
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEL = ROOT / "data/selective"
SEED = 20261002
NPTS = ["N@a0.5d256", "N@a0.75d256", "N@a1.0d256"]
CAND = ["Nall@a1.0d256", "Nred@a1.0d256", "Nprod@a1.0d256", "Nsel@a1.0d256"]


def load(pattern: str) -> list[dict]:
    rows = []
    for f in sorted(glob.glob(str(ROOT / pattern))):
        rows += list(csv.DictReader(open(f)))
    return rows


def interp(pts: list[tuple[float, float]], s: float) -> float:
    pts = sorted(pts)
    for (s0, a0), (s1, a1) in zip(pts, pts[1:]):
        if s0 <= s <= s1:
            return a0 if s1 == s0 else a0 + (a1 - a0) * (s - s0) / (s1 - s0)
    return float("nan")


def ci(v) -> list[float]:
    v = np.asarray(v, float)
    v = v[~np.isnan(v)]
    if not len(v):
        return [float("nan")] * 2
    return [round(float(np.percentile(v, 2.5)), 2), round(float(np.percentile(v, 97.5)), 2)]


def cube(rows: list[dict], arms: list[str], seeds: int):
    """[question, arm, seed] arrays of correct / tokens / doubt / capped."""
    qs = sorted({r["question_id"] for r in rows})
    qi, ai = {q: i for i, q in enumerate(qs)}, {a: i for i, a in enumerate(arms)}
    shape = (len(qs), len(arms), seeds)
    out = {k: np.full(shape, np.nan) for k in ("correct", "tokens", "doubt", "capped")}
    for r in rows:
        if r["policy"] not in ai:
            continue
        ix = (qi[r["question_id"]], ai[r["policy"]], int(r["seed"]))
        out["correct"][ix] = float(r["correct"])
        out["tokens"][ix] = float(r["total_tokens"])
        out["doubt"][ix] = float(r["doubt_blocks"])
        out["capped"][ix] = float(r["capped"])
    missing = int(np.isnan(out["correct"]).sum())
    return qs, out, missing


def main() -> None:
    cohort = {json.loads(x)["question_id"]: json.loads(x) for x in open(SEL / "mathtest_sel400.jsonl")}
    rows = load("data/selective/eval/sel400_s*/rollouts.csv")
    arms = ["base", *NPTS] + [a for a in CAND if any(r["policy"] == a for r in rows)]
    qs, C, missing = cube(rows, arms, 4)
    lv = np.array([cohort[q]["level"] for q in qs])
    A = {a: i for i, a in enumerate(arms)}
    rep: dict = {"questions": len(qs), "arms": arms, "missing_cells": missing}

    def stats(ii: np.ndarray, with_base_pt: bool = False) -> dict:
        acc = np.nanmean(C["correct"][ii], axis=(0, 2))
        tok = np.nanmean(C["tokens"][ii], axis=(0, 2))
        sav = 100 * (1 - tok / tok[A["base"]])
        da = 100 * (acc - acc[A["base"]])
        pts = [(sav[A[a]], da[A[a]]) for a in NPTS] + ([(0.0, 0.0)] if with_base_pt else [])
        exc = {a: da[A[a]] - interp(pts, sav[A[a]]) for a in arms if a in CAND}
        return {"acc": acc, "tok": tok, "sav": sav, "da": da, "exc": exc}

    all_q = np.arange(len(qs))
    pt = stats(all_q)
    rng = np.random.default_rng(SEED)
    B = [stats(rng.integers(0, len(qs), len(qs))) for _ in range(2000)]
    rng = np.random.default_rng(SEED)
    B0 = [stats(rng.integers(0, len(qs), len(qs)), True) for _ in range(2000)]
    rep["arms_L45"] = {a: {"acc": round(100 * float(pt["acc"][A[a]]), 2),
                           "tokens": round(float(pt["tok"][A[a]]), 1),
                           "saving_pts": round(float(pt["sav"][A[a]]), 2),
                           "saving_ci": ci([b["sav"][A[a]] for b in B]),
                           "dacc_pp": round(float(pt["da"][A[a]]), 2),
                           "dacc_ci": ci([b["da"][A[a]] for b in B])} for a in arms}
    for tag, bs, wb in (("excess", B, False), ("excess_with_base_point", B0, True)):
        p = stats(all_q, wb)
        rep[tag] = {a: {"point": round(float(p["exc"][a]), 2), "ci": ci([b["exc"][a] for b in bs]),
                        "outside_curve_share": round(float(np.mean([np.isnan(b["exc"][a]) for b in bs])), 3)}
                    for a in p["exc"]}
    if "Nred@a1.0d256" in A and "Nall@a1.0d256" in A:
        d = lambda s: s["exc"]["Nred@a1.0d256"] - s["exc"]["Nall@a1.0d256"]  # noqa: E731
        sel_ci, red_ci = ci([d(b) for b in B]), rep["excess"]["Nred@a1.0d256"]["ci"]
        sav_ok = pt["sav"][A["Nred@a1.0d256"]] >= pt["sav"][A["N@a1.0d256"]] - 10
        rep["primary"] = {"selectivity_point": round(float(d(pt)), 2), "selectivity_ci": sel_ci,
                          "win": {"selectivity_ci>0": bool(sel_ci[0] > 0),
                                  "excess_red_ci>0": bool(red_ci[0] > 0),
                                  "saving_red>=saving_N-10": bool(sav_ok)}}
        rep["primary"]["WIN"] = all(rep["primary"]["win"].values())
    # secondary: per level vs base, doubt paragraphs, capped
    sec = {}
    for L in (4, 5):
        ii = np.flatnonzero(lv == L)
        rng = np.random.default_rng(SEED)
        bs = [rng.choice(ii, len(ii)) for _ in range(2000)]
        sec[f"L{L}"] = {}
        for a in arms:
            f = lambda j: 100 * (np.nanmean(C["correct"][j][:, A[a]]) - np.nanmean(C["correct"][j][:, A["base"]]))  # noqa: E731
            g = lambda j: 100 * (1 - np.nanmean(C["tokens"][j][:, A[a]]) / np.nanmean(C["tokens"][j][:, A["base"]]))  # noqa: E731
            sec[f"L{L}"][a] = {"acc": round(100 * float(np.nanmean(C["correct"][ii][:, A[a]])), 2),
                               "dacc": round(float(f(ii)), 2), "dacc_ci": ci([f(j) for j in bs]),
                               "saving": round(float(g(ii)), 2), "saving_ci": ci([g(j) for j in bs])}
    rep["per_level"] = sec
    rep["doubt_capped"] = {a: {"doubt_paragraphs": round(float(np.nanmean(C["doubt"][:, A[a]])), 2),
                               "capped_rate": round(float(np.nanmean(C["capped"][:, A[a]])), 4)}
                           for a in arms}
    aime = load("data/selective/eval/aime_s*/rollouts.csv")
    if aime:
        aarms = [a for a in arms if any(r["policy"] == a for r in aime)]
        aq, AC, am = cube(aime, aarms, 8)
        rep["aime_amc"] = {"missing_cells": am, **{a: {
            "acc": round(100 * float(np.nanmean(AC["correct"][:, i])), 2),
            "tokens": round(float(np.nanmean(AC["tokens"][:, i])), 1),
            "capped": round(float(np.nanmean(AC["capped"][:, i])), 4)} for i, a in enumerate(aarms)}}
    mech = SEL / "episodes_mech.jsonl"
    if mech.exists():
        eps = [json.loads(x) for x in open(mech)]
        nroll = collections.Counter()
        for x in open(SEL / "mech_select.jsonl"):
            nroll[json.loads(x)["policy"]] += 1
        cnt = collections.defaultdict(collections.Counter)
        for e in eps:
            cnt[e["policy"]][e["label"]] += 1
        rep["mechanism"] = {p: {"rollouts": nroll[p], **{f"{lab}_per_rollout": round(cnt[p][lab] / nroll[p], 3)
                                                        for lab in ("productive", "redundant", "harmful", "failed", "other")},
                                "doubt_episodes_per_rollout": round(sum(cnt[p].values()) / nroll[p], 2)}
                            for p in nroll}
    (SEL / "report.json").write_text(json.dumps(rep, indent=1, default=float) + "\n")
    print(json.dumps(rep, indent=1, default=float))


if __name__ == "__main__":
    main()
