#!/usr/bin/env python3
r"""Amendment F readout (EXPERIMENT-reviewer-baselines.md §F2/§F3): 1.7B N@a0.5d256 on hard128.

Reuses baselines_report's D69 regrade cache and question bootstrap (2,000 reps, seed
20261005; one resample shared by α 0.5 and α 1.0). Base and N@a1.0d256 come from c17_s*;
the α 0.5 arm from f17_s*; the reproducibility check compares f17_base_s01 with c17 base
on seeds 0-1. Language switch = share of rollouts whose thinking (text before </think>, or
all text if none) has >= 20 chars in U+4E00-U+9FFF.

Writes data/baselines/amendf.{json,md}.

    uv run python scripts/amendf_report.py
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from baselines_report import REPS, SEED, Boot, _grade_file, ci, diff, strip  # noqa: E402

B = ROOT / "data/baselines"
A10, A05 = "N@a1.0d256", "N@a0.5d256"
CJK = re.compile(r"[一-鿿]")


def journals(pat: str) -> list[Path]:
    return [Path(d) / "rollouts.jsonl" for d in sorted(glob.glob(str(B / pat)))
            if (Path(d) / "rollouts.csv").exists()]


def load(pat: str, pol: str) -> pd.DataFrame:
    parts = [pd.read_csv(_grade_file(str(j))) for j in journals(pat)]
    df = pd.concat(parts)
    df = df[df.policy == pol].copy()
    if df.duplicated(["question_id", "seed"]).any():
        raise SystemExit(f"duplicate (question, seed) rows: {pat} {pol}")
    return df


def lang(pat: str, pol: str) -> dict:
    n = hit = 0
    for j in journals(pat):
        with open(j) as f:
            for line in f:
                r = json.loads(line)
                if r["policy"] != pol:
                    continue
                think = r["text"].split("</think>", 1)[0]
                n += 1
                hit += len(CJK.findall(think)) >= 20
    return {"n": n, "switched": hit, "rate": hit / n * 100 if n else float("nan")}


def audit(pat: str, pol: str) -> dict:
    """The runner raises on any site/injection mismatch, so a journaled row passed it; this
    re-checks the journaled counts (injections == eligible sites)."""
    n = bad = inj = 0
    for j in journals(pat):
        with open(j) as f:
            for line in f:
                r = json.loads(line)
                if r["policy"] != pol:
                    continue
                n += 1
                inj += int(r["injections"])
                bad += int(r["injections"]) != int(r["eligible_sites"])
    return {"n": n, "bad": bad, "injections": inj}


def main() -> None:
    base = load("c17_s*", "base")
    arms = {"base": base, A10: load("c17_s*", A10), A05: load("f17_s[0-9][0-9]", A05)}
    q = sorted(set(base.question_id))
    bt = Boot(arms, q)
    r10, r05 = bt.stats(A10), bt.stats(A05)
    d = diff(r05, r10)
    sens = sorted(x for x in q if x.startswith(("aime2024:", "hmmt_feb2025:")))
    bs = Boot(arms, sens)
    s10, s05 = bs.stats(A10), bs.stats(A05)

    # base reproducibility (seeds 0-1)
    nb = load("f17_base_s01", "base")
    ob = base[base.seed.isin([0, 1])]
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(q), (REPS, len(q)))
    oq = ob.groupby("question_id")[["correct", "total_tokens"]].mean().reindex(q)
    nq = nb.groupby("question_id")[["correct", "total_tokens"]].mean().reindex(q)
    oacc_ci = ci(oq.correct.to_numpy()[idx].mean(1) * 100)
    oacc, nacc = oq.correct.mean() * 100, nq.correct.mean() * 100
    otok, ntok = ob.total_tokens.mean(), nb.total_tokens.mean()
    repro = {"c17_acc": oacc, "c17_acc_ci": oacc_ci, "f17_acc": nacc,
             "c17_tokens": otok, "f17_tokens": ntok, "tok_change_pct": (ntok / otok - 1) * 100,
             "n_c17": len(ob), "n_f17": len(nb),
             "acc_ok": oacc_ci[0] <= nacc <= oacc_ci[1],
             "tok_ok": abs(ntok / otok - 1) <= 0.05}
    repro["passes"] = repro["acc_ok"] and repro["tok_ok"]

    L = {"base": lang("c17_s*", "base"), A10: lang("c17_s*", A10),
         A05: lang("f17_s[0-9][0-9]", A05)}

    if r05["dacc_ci"][0] >= -2:
        reading = "At α 0.5 the 1.7B is non-inferior on competition problems."
    elif d["dacc_ci"][0] > 0:
        reading = (f"The gentler push loses less on competition problems: {r05['dacc']:+.1f} vs "
                   f"{r10['dacc']:+.1f}pp, at {r05['saving']:.1f}% vs {r10['saving']:.0f}% saving.")
    else:
        reading = ("On competition problems α 0.5 costs about as much as α 1.0; the language "
                   "drift is a side effect, not the source of the cost.")
    lead = L[A05]["rate"] <= 2 and r05["dacc"] >= r10["dacc"]

    rep = {"arms": {A10: strip(r10), A05: strip(r05)}, "a05_minus_a10": d,
           "sens_aime24_hmmt": {A10: strip(s10), A05: strip(s05)}, "repro": repro,
           "language_switch": L, "reading": reading, "a05_leads": lead,
           "audit": audit("f17_s[0-9][0-9]", A05)}
    (B / "amendf.json").write_text(json.dumps(rep, indent=1, default=float))

    def row(name: str, r: dict) -> str:
        return (f"| {name} | {r['acc']:.1f} | {r['dacc']:+.1f} [{r['dacc_ci'][0]:+.1f}, "
                f"{r['dacc_ci'][1]:+.1f}] | {r['saving']:.1f} [{r['saving_ci'][0]:.1f}, "
                f"{r['saving_ci'][1]:.1f}] | {r['capped']:.2f} | {r['doubt_removed']:.1f} | "
                f"{L[name]['rate']:.1f} | {r['n_pairs']} |")

    md = ["# Amendment F — 1.7B N@a0.5d256 on hard128", "",
          f"base (c17, 8 seeds): acc {r10['base_acc']:.1f}%, capped {r10['base_capped']:.2f}%, "
          f"language switch {L['base']['rate']:.1f}% ({L['base']['switched']}/{L['base']['n']})", "",
          "| arm | acc | Δacc [95% CI] | saving % [CI] | capped % | doubt removed % | lang switch % | pairs |",
          "|---|---:|---|---|---:|---:|---:|---:|", row(A10, r10), row(A05, r05), "",
          f"**α 0.5 − α 1.0:** Δacc {d['dacc']:+.1f} [{d['dacc_ci'][0]:+.1f}, {d['dacc_ci'][1]:+.1f}], "
          f"saving {d['saving']:+.1f} [{d['saving_ci'][0]:+.1f}, {d['saving_ci'][1]:+.1f}]", "",
          f"**AIME24+HMMT ({len(sens)} q):** α 1.0 {s10['dacc']:+.1f} [{s10['dacc_ci'][0]:+.1f}, "
          f"{s10['dacc_ci'][1]:+.1f}], α 0.5 {s05['dacc']:+.1f} [{s05['dacc_ci'][0]:+.1f}, "
          f"{s05['dacc_ci'][1]:+.1f}]", "",
          f"**Base reproducibility (seeds 0–1):** c17 {oacc:.1f}% [{oacc_ci[0]:.1f}, {oacc_ci[1]:.1f}], "
          f"f17 {nacc:.1f}%; tokens {otok:.0f} vs {ntok:.0f} ({repro['tok_change_pct']:+.1f}%) → "
          f"{'PASS' if repro['passes'] else 'FAIL'}", "",
          f"**Reading (F3):** {reading}", "",
          f"**Lead row:** α 0.5 {'leads' if lead else 'does not lead'} "
          f"(switch {L[A05]['rate']:.1f}% ≤ 2: {L[A05]['rate'] <= 2}; "
          f"Δacc {r05['dacc']:+.2f} ≥ {r10['dacc']:+.2f}: {r05['dacc'] >= r10['dacc']})", ""]
    (B / "amendf.md").write_text("\n".join(md))
    print("\n".join(md))
    print(json.dumps(rep["audit"]))


if __name__ == "__main__":
    main()
