#!/usr/bin/env python3
r"""Report for EXPERIMENT-reviewer-baselines.md (§B5 statistics, §B8 readings).

Every journal, new or reused, is regraded from its text with the current D69 grader. Arms
are paired with their base on (question, seed); per-question seed means; paired bootstrap
over questions (2,000 reps, seed 20261005), with one resample shared by every quantity of
a set (and across models on hard128). Saving = 1 - sum_q tok(arm) / sum_q tok(base).

Writes data/baselines/report.{json,md}. Missing runs are skipped and listed.

    uv run python scripts/baselines_report.py
"""

from __future__ import annotations

import glob
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

B = "data/baselines"
REPS, SEED = 2000, 20261005
CACHE = ROOT / B / "cache"

# (model, set) -> arm -> [(run-dir glob, policy in that dir, dataset filter or None)]
N17, N8, N30 = "N@a1.0d256", "NU26@a0.75d256", "NU34@a0.75d256"


def du(tag: str) -> str | None:
    p = ROOT / B / f"{tag}_selected.txt"
    s = p.read_text().strip() if p.exists() else ""
    return s if s and s != "NONE" else None


DU8, DU30 = du("b8"), du("b30")
SPECS: dict[tuple[str, str], dict[str, list]] = {
    ("1.7B", "confirm400"): {
        **{a: [(f"{B}/a17_confirm_s*", a, None)] for a in
           ("base", N17, "N@a0.5d256", "ban_all", "ban_para_d256", "SEAL", "SEALu@a1.0d256")},
        "DEER": [(f"{B}/d17_confirm_deer", "DEER", None)]},
    ("1.7B", "gate400"): {
        **{a: [(f"{B}/a17_gate_s*", a, None)] for a in
           ("base", N17, "N@a0.5d256", "ban_all", "ban_para_d256", "SEAL", "SEALu@a1.0d256")},
        "DEER": [(f"{B}/d17_gate_deer", "DEER", None)]},
    ("8B", "confirm400"): {
        "base": [("data/xfer8b/r8b_confirm_h[01]", "base", None)],
        N8: [("data/xfer8b/r8b_confirm_L26", N8, None)],
        "D26@a0.5d256": [("data/xfer8b/d26_confirm", "D26@a0.5d256", None)],
        **({DU8: [(f"{B}/b8_confirm", DU8, None)]} if DU8 else {}),
        **{a: [(f"{B}/a8_confirm_s*", a, None)] for a in ("ban_all", "ban_para_d256", "SEAL")},
        "DEER": [(f"{B}/d8_confirm_deer", "DEER", None)]},
    ("8B", "gate400"): {
        "base": [("data/distill_8b/online/s[0-3]", "base", "mathgate")],
        N8: [("data/distill_8b/online/s[0-3]", N8, "mathgate")],
        "SEAL": [(f"{B}/d8_gate_seal", "SEAL", None)],
        "DEER": [(f"{B}/d8_gate_deer", "DEER", None)]},
    ("30B", "confirm400"): {
        "base": [("data/xfer30b/r30_confirm_base", "base", None)],
        N30: [("data/xfer30b/r30_confirm_arm", N30, None)],
        "D34@a0.5d256": [("data/xfer30b/d34_confirm", "D34@a0.5d256", None)],
        **({DU30: [(f"{B}/b30_confirm", DU30, None)]} if DU30 else {})},
    ("1.7B", "hard128"): {
        **{a: [(f"{B}/c17_s*", a, None)] for a in
           ("base", N17, "D@a0.5d512", "ban_para_d256", "SEAL")},
        "DEER": [(f"{B}/c17_deer_s*", "DEER", None)]},
    ("8B", "hard128"): {
        **{a: [(f"{B}/c8_s*", a, None)] for a in
           ("base", N8, "D26@a0.5d256", *([DU8] if DU8 else []), "ban_para_d256", "SEAL")},
        "DEER": [(f"{B}/c8_deer_s*", "DEER", None)]},
    ("30B", "hard128"): {
        a: [(f"{B}/c30_s*", a, None)] for a in
        ("base", N30, "D34@a0.5d256", *([DU30] if DU30 else []))},
    ("235B", "hard128"): {
        a: [(f"{B}/c235_s*", a, None)] for a in ("base", "NU67@a1.5d256")},
}
N_ARM = {"1.7B": N17, "8B": N8, "30B": N30, "235B": "NU67@a1.5d256"}


def _grade_file(path: str) -> str:
    """Regrade one journal; cache (question, policy, seed, correct, ...) as csv keyed by
    the journal's size and mtime."""
    from reasoning_attention.grading import grade

    p = Path(path)
    st = p.stat()
    import hashlib

    key = (f"{hashlib.sha256(str(p.resolve()).encode()).hexdigest()[:16]}_{st.st_size}_"
           f"{int(st.st_mtime)}")
    out = CACHE / f"{key}.csv"
    if out.exists():
        return str(out)
    rows = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            g = grade(r["text"], r["gold"])
            rows.append({
                "question_id": r["question_id"], "policy": r["policy"], "seed": int(r["seed"]),
                "dataset": r.get("dataset", ""), "correct": int(g.is_correct),
                "correct_journal": int(r["correct"]), "total_tokens": int(r["total_tokens"]),
                "capped": int(r["capped"]), "doubt_blocks": int(r["doubt_blocks"]),
                "boundaries": int(r.get("boundaries", 0)),
                "trial_tokens": int(r.get("trial_tokens", 0) or 0),
                "deer_exit": r.get("deer_exit", ""), "deer_checks": int(r.get("deer_checks", 0) or 0),
                "seal_ratio": r.get("seal_ratio", ""), "nonseal_sites": r.get("nonseal_sites", ""),
                "ban_sites": r.get("ban_sites", ""),
            })
    tmp = out.with_suffix(".tmp")
    pd.DataFrame(rows).to_csv(tmp, index=False)
    tmp.rename(out)
    return str(out)


def load_all() -> tuple[dict, list[str]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    journals = set()
    for arms in SPECS.values():
        for specs in arms.values():
            for pat, _, _ in specs:
                for d in glob.glob(str(ROOT / pat)):
                    if (Path(d) / "rollouts.csv").exists() and (Path(d) / "rollouts.jsonl").exists():
                        journals.add(str(Path(d) / "rollouts.jsonl"))
    with ProcessPoolExecutor(16) as ex:
        cached = dict(zip(sorted(journals), ex.map(_grade_file, sorted(journals))))
    frames: dict = {}
    missing = []
    for ms, arms in SPECS.items():
        for arm, specs in arms.items():
            parts = []
            for pat, pol, ds in specs:
                for d in sorted(glob.glob(str(ROOT / pat))):
                    j = str(Path(d) / "rollouts.jsonl")
                    if j not in cached:
                        continue
                    df = pd.read_csv(cached[j])
                    df["total_with_trial"] = df.total_tokens + df.trial_tokens
                    df = df[df.policy == pol]
                    if ds:
                        df = df[df.dataset == ds]
                    parts.append(df)
            if not parts or not sum(len(p) for p in parts):
                missing.append(f"{ms[0]} {ms[1]} {arm}")
                continue
            df = pd.concat(parts)
            if df.duplicated(["question_id", "seed"]).any():
                raise SystemExit(f"duplicate (question, seed) rows in {ms} {arm}")
            frames.setdefault(ms, {})[arm] = df
    return frames, missing


class Boot:
    """Per-question seed means over the (question, seed) pairs shared with base, and one
    shared resample of questions."""

    def __init__(self, arms: dict[str, pd.DataFrame], questions: list[str], rng_seed: int = SEED):
        self.q = questions
        self.idx = np.random.default_rng(rng_seed).integers(0, len(questions), (REPS, len(questions)))
        self.arms = arms

    def paired(self, a: str, b: str, tok_col: str = "total_tokens") -> dict:
        A, Bf = self.arms[a], self.arms[b]
        m = A.merge(Bf, on=["question_id", "seed"], suffixes=("", "_b"))
        g = m.groupby("question_id")
        cols = ["correct", "correct_b", tok_col, "total_tokens_b", "doubt_blocks",
                "doubt_blocks_b", "capped", "capped_b"]
        qm = g[cols].mean().reindex(self.q)
        if qm.isna().any().any():
            raise SystemExit(f"{a} vs {b}: questions without paired rows")
        return {"n_pairs": len(m), "qm": qm, "tok_col": tok_col}

    def stats(self, a: str, base: str = "base", tok_col: str = "total_tokens") -> dict:
        p = self.paired(a, base, tok_col)
        qm = p["qm"]
        acc, accb = qm.correct.to_numpy(), qm.correct_b.to_numpy()
        ta, tb = qm[tok_col].to_numpy(), qm.total_tokens_b.to_numpy()
        da, db = qm.doubt_blocks.to_numpy(), qm.doubt_blocks_b.to_numpy()
        d = (acc - accb) * 100
        dboot = d[self.idx].mean(1)
        sav = (1 - ta.sum() / tb.sum()) * 100
        sboot = (1 - ta[self.idx].sum(1) / tb[self.idx].sum(1)) * 100
        return {
            "n_q": len(self.q), "n_pairs": p["n_pairs"], "acc": acc.mean() * 100,
            "base_acc": accb.mean() * 100, "dacc": d.mean(), "dacc_ci": ci(dboot),
            "saving": sav, "saving_ci": ci(sboot), "tokens": ta.mean(), "base_tokens": tb.mean(),
            "capped": qm.capped.mean() * 100, "base_capped": qm.capped_b.mean() * 100,
            "doubt": da.mean(), "base_doubt": db.mean(),
            "doubt_removed": (1 - da.sum() / db.sum()) * 100 if db.sum() else float("nan"),
            "_d": dboot, "_s": sboot, "_dpt": d, "_sav": sav,
        }


def ci(x: np.ndarray) -> list[float]:
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def diff(x: dict, y: dict) -> dict:
    """x - y for Δacc and saving, on the same resample."""
    return {"dacc": x["dacc"] - y["dacc"], "dacc_ci": ci(x["_d"] - y["_d"]),
            "saving": x["saving"] - y["saving"], "saving_ci": ci(x["_s"] - y["_s"])}


def pipeline(base: pd.DataFrame, arm: pd.DataFrame | None, tok_col: str = "total_tokens") -> dict:
    """Claim 3b: arm seeds 0-1, then base seeds 2-3 for questions with no correct arm
    rollout; arm=None is the staged base-only pipeline (base seeds 0-1, then 2-3)."""
    spend = covered = 0
    n = 0
    for qid, gb in base.groupby("question_id"):
        n += 1
        gb = gb.sort_values("seed")
        ga = gb.head(2) if arm is None else arm[arm.question_id == qid].sort_values("seed").head(2)
        col = "total_tokens" if arm is None else tok_col
        spend += ga[col].sum()
        ok = bool(ga.correct.any())
        if not ok:
            rest = gb[gb.seed.isin([2, 3])]
            spend += rest.total_tokens.sum()
            ok = bool(rest.correct.any())
        covered += ok
    return {"coverage": covered / n * 100, "tok_per_accepted": spend / max(covered, 1)}


def tok_col_for(arm: str) -> str:
    return "total_with_trial" if arm == "DEER" else "total_tokens"


def ban_reach() -> dict | None:
    """Share of base doubt paragraphs (1.7B confirm400 base, this session) whose first token
    is in V_ban: the matched ban's maximum reach (§B2)."""
    from transformers import AutoTokenizer

    from sel_common import Paragraphs

    js = sorted(glob.glob(str(ROOT / B / "a17_confirm_s*/rollouts.jsonl")))
    if not js:
        return None
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    P = Paragraphs(tok)
    ban = {int(x["id"]) for x in json.loads((ROOT / B / "ban_tokens.json").read_text())["ban"]}
    tot = hit = tot256 = hit256 = 0
    for j in js:
        for line in open(j):
            r = json.loads(line)
            if r["policy"] != "base":
                continue
            ids = r["token_ids"]
            sp = P.split(ids)
            for b, dbt in zip(sp["boundaries"], sp["doubt"]):
                if not dbt or b + 1 >= len(ids):
                    continue
                inb = ids[b + 1] in ban
                tot += 1
                hit += inb
                if b + 1 >= 256:
                    tot256 += 1
                    hit256 += inb
    return {"doubt_paragraphs": tot, "open_with_banned": hit, "share": hit / tot * 100,
            "g256": {"doubt_paragraphs": tot256, "share": hit256 / max(tot256, 1) * 100}}


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


def main() -> None:
    frames, missing = load_all()
    rep: dict = {"missing": missing, "sets": {}, "readings": {}}
    md = ["# Reviewer baselines — report", "",
          "All rollouts regraded with the D69 grader. Δacc in pp vs the set's base, paired on "
          "(question, seed); saving = 1 − tok(arm)/tok(base); 95% CIs from 2,000 question "
          "bootstraps (seed 20261005). DEER tokens include trial answers where noted.", ""]
    if missing:
        md += ["**Missing runs:** " + "; ".join(missing), ""]
    S: dict = {}
    boots: dict = {}
    for (model, sset), arms in frames.items():
        if "base" not in arms:
            continue
        q = sorted(set(arms["base"].question_id))
        bt = Boot(arms, q)
        boots[(model, sset)] = bt
        res = {}
        for arm in arms:
            if arm == "base":
                continue
            r = bt.stats(arm)
            if arm == "DEER":
                rt = bt.stats(arm, tok_col="total_with_trial")
                r["saving_with_trial"], r["saving_with_trial_ci"] = rt["saving"], rt["saving_ci"]
                r["_wt"] = rt
                df = arms[arm]
                r["trial_tokens_mean"] = df.trial_tokens.mean()
                r["exits"] = df.deer_exit.value_counts().to_dict()
                r["checks_mean"] = df.deer_checks.mean()
            if arm.startswith("SEAL") and arm == "SEAL":
                df = arms[arm]
                r["seal_ratio_mean"] = pd.to_numeric(df.seal_ratio, errors="coerce").mean()
                r["nonseal_sites_total"] = int(pd.to_numeric(df.nonseal_sites, errors="coerce").sum())
            res[arm] = r
        S[(model, sset)] = res
        # pipeline cost (confirm400 / gate400)
        pipe = {}
        if sset in ("confirm400", "gate400"):
            pipe["base"] = pipeline(arms["base"], None)
            for arm in arms:
                if arm != "base":
                    pipe[arm] = pipeline(arms["base"], arms[arm], tok_col_for(arm))
        rep["sets"][f"{model} {sset}"] = {"arms": {a: strip(r) for a, r in res.items()},
                                         "pipeline": pipe,
                                         "base": {"acc": arms["base"].correct.mean() * 100,
                                                  "tokens": arms["base"].total_tokens.mean(),
                                                  "n_rows": len(arms["base"])}}
        md += [f"## {model} — {sset}", "",
               f"base: acc {arms['base'].correct.mean() * 100:.1f}%, mean tokens "
               f"{arms['base'].total_tokens.mean():,.0f}, capped "
               f"{arms['base'].capped.mean() * 100:.2f}%, doubt/rollout "
               f"{arms['base'].doubt_blocks.mean():.1f} ({len(arms['base'])} rollouts)", "",
               "| arm | n pairs | acc | Δacc [CI] | saving [CI] | capped | doubt removed |"
               + (" pipeline cov | tok/accepted |" if pipe else ""),
               "|---|---:|---:|---|---|---:|---:|" + ("---:|---:|" if pipe else "")]
        for arm, r in res.items():
            line = (f"| {arm} | {r['n_pairs']} | {r['acc']:.1f} | {r['dacc']:+.2f} "
                    f"[{r['dacc_ci'][0]:+.2f}, {r['dacc_ci'][1]:+.2f}] | {r['saving']:.1f}% "
                    f"[{r['saving_ci'][0]:.1f}, {r['saving_ci'][1]:.1f}]"
                    + (f" (w/ trial {r['saving_with_trial']:.1f}%)" if "saving_with_trial" in r else "")
                    + f" | {r['capped']:.2f}% | {r['doubt_removed']:.1f}% |")
            if pipe:
                line += f" {pipe[arm]['coverage']:.1f}% | {pipe[arm]['tok_per_accepted']:,.0f} |"
            md.append(line)
        if pipe:
            md.append(f"| base-only pipeline | | | | | | | {pipe['base']['coverage']:.1f}% | "
                      f"{pipe['base']['tok_per_accepted']:,.0f} |")
        for arm, r in res.items():
            if arm == "DEER":
                md.append(f"\nDEER: exits {r['exits']}, checks/rollout {r['checks_mean']:.2f}, "
                          f"trial tokens/rollout {r['trial_tokens_mean']:.1f}")
            if arm == "SEAL":
                md.append(f"\nSEAL: mean ‖S‖/‖h‖ at sites {r['seal_ratio_mean']:.3f}; sites "
                          f"outside SEAL's split set: {r['nonseal_sites_total']}")
        md.append("")

    # ---------------- readings ----------------
    def beats(n: dict, x: dict) -> dict:
        d = diff(n, x)
        return {**d, "beats": d["saving_ci"][0] > 0 and d["dacc_ci"][0] >= -2}

    # Part A
    A = {}
    for ms in (("1.7B", "confirm400"), ("1.7B", "gate400"), ("8B", "confirm400")):
        if ms not in S or "ban_para_d256" not in S[ms]:
            continue
        r = S[ms]
        ns = [a for a in (N17, "N@a0.5d256") if a in r] if ms[0] == "1.7B" else [N8]
        out = {f"{n} vs B2": beats(r[n], r["ban_para_d256"]) for n in ns}
        n1 = r[ns[0]]
        dm = diff(r["ban_para_d256"], n1)
        out["ban_matches"] = (-3 <= dm["saving_ci"][0] and dm["saving_ci"][1] <= 3
                              and dm["dacc_ci"][0] >= -2)
        out["ban_minus_N"] = dm
        out["N_beats_B2"] = any(v["beats"] for k, v in out.items() if k.endswith("vs B2"))
        out["B2_doubt_removed"] = r["ban_para_d256"]["doubt_removed"]
        if "ban_all" in r:
            out["B1_vs_N"] = beats(r[ns[0]], r["ban_all"])
        A[f"{ms[0]} {ms[1]}"] = out
    reach = ban_reach()
    rep["readings"]["A"] = {"sets": A, "ban_reach": reach}
    if A:
        allbeat = all(v["N_beats_B2"] for v in A.values())
        anymatch = any(v["ban_matches"] for v in A.values())
        rep["readings"]["A"]["outcome"] = ("N beats B2 on every set" if allbeat else
                                           "the ban matches N" if anymatch else "mixed")
    # Part B
    Bres = {}
    for model, dsel, nat in (("8B", DU8, "D26@a0.5d256"), ("30B", DU30, "D34@a0.5d256")):
        ms = (model, "confirm400")
        if ms in S and dsel and dsel in S[ms]:
            r = S[ms]
            d = diff(r[N_ARM[model]], r[dsel])
            Bres[model] = {"DU": dsel, "N_minus_DU": d,
                           "N_saves_more": d["saving_ci"][0] > 0 and d["dacc_ci"][0] >= -2,
                           "DU_within3": (-3 <= d["saving_ci"][0] and d["saving_ci"][1] <= 3
                                          and -d["dacc_ci"][1] >= -2),
                           "native_D": strip(r[nat]) if nat in r else None}
        elif ms in frames:
            Bres[model] = {"DU": dsel or "NONE (failed dev or not run)"}
    rep["readings"]["B"] = Bres
    # Part C
    C = {}
    hb = {m: boots.get((m, "hard128")) for m in ("1.7B", "8B", "30B", "235B")}
    common = None
    for m, bt in hb.items():
        if bt and N_ARM[m] in bt.arms:
            s = set(bt.q)
            common = s if common is None else common & s
    if common:
        q = sorted(common)
        per = {}
        for m, bt in hb.items():
            if bt and N_ARM[m] in bt.arms:
                b2 = Boot(bt.arms, q)  # same seed -> same resample of the same question list
                per[m] = b2.stats(N_ARM[m])
                C[m] = strip(per[m])
                # by source dataset, and the AIME2024+HMMT sensitivity
                bys = {}
                for ds in ("aime2025", "aime2024", "amc23", "hmmt_feb2025"):
                    qs = sorted({x for x in q if x.startswith(ds + ":")})
                    if qs:
                        bys[ds] = strip(Boot(bt.arms, qs).stats(N_ARM[m]))
                sens = sorted(x for x in q if x.startswith(("aime2024:", "hmmt_feb2025:")))
                C[m]["by_dataset"] = bys
                C[m]["sens_aime24_hmmt"] = strip(Boot(bt.arms, sens).stats(N_ARM[m]))
        cross = {}
        for m in ("8B", "30B", "235B"):
            if m in per and "1.7B" in per:
                cross[f"{m} - 1.7B"] = {"ddacc": per[m]["dacc"] - per["1.7B"]["dacc"],
                                        "ci": ci(per[m]["_d"] - per["1.7B"]["_d"])}
        C["cross"] = cross
        if "8B - 1.7B" in cross and "30B - 1.7B" in cross:
            C["scale_claim_holds"] = all(cross[k]["ci"][0] > 0 for k in ("8B - 1.7B", "30B - 1.7B"))
        C["noninferior_2pp"] = {m: per[m]["dacc_ci"][0] >= -2 for m in per}
    rep["readings"]["C"] = C
    # Part D
    D = {}
    for ms, r in S.items():
        if ms[1] not in ("confirm400", "gate400", "hard128"):
            continue
        ns = [a for a in (N17, "N@a0.5d256") if a in r] if ms[0] == "1.7B" else [N_ARM[ms[0]]]
        ns = [n for n in ns if n in r]
        for X in ("SEAL", "DEER"):
            if X not in r or not ns:
                continue
            x = r[X]["_wt"] if X == "DEER" else r[X]  # DEER pays for its trial answers
            nb = {n: beats(r[n], x) for n in ns}
            xb = {n: beats(x, r[n]) for n in ns}
            ent = {"N_beats_X": any(v["beats"] for v in nb.values()),
                   "X_beats_N": any(v["beats"] for v in xb.values()),
                   "N_minus_X": {n: {k: v for k, v in nb[n].items()} for n in ns}}
            pipe = rep["sets"][f"{ms[0]} {ms[1]}"]["pipeline"]
            if pipe:
                cov0 = pipe["base"]["coverage"]
                elig = {a: pipe[a] for a in [*ns, X] if abs(pipe[a]["coverage"] - cov0) <= 1}
                ent["tiebreak_eligible"] = elig
                ent["tiebreak_winner"] = (min(elig, key=lambda a: elig[a]["tok_per_accepted"])
                                          if elig else None)
            D[f"{ms[0]} {ms[1]} {X}"] = ent
    rep["readings"]["D"] = D

    md += ["## Readings (§B8)", "", "### Part A — N vs the matched ban (B2)", ""]
    if reach:
        md.append(f"Ban reach (1.7B confirm400 base): {reach['share']:.1f}% of "
                  f"{reach['doubt_paragraphs']:,} doubt paragraphs open with a V_ban token "
                  f"({reach['g256']['share']:.1f}% at g ≥ 256).")
    for k, v in A.items():
        md.append(f"- **{k}**: " + "; ".join(
            f"{n}: Δsaving {x['saving']:+.1f} [{x['saving_ci'][0]:+.1f}, {x['saving_ci'][1]:+.1f}], "
            f"Δacc {x['dacc']:+.2f} [{x['dacc_ci'][0]:+.2f}, {x['dacc_ci'][1]:+.2f}] → "
            f"{'beats' if x['beats'] else 'does not beat'}"
            for n, x in v.items() if n.endswith("vs B2"))
            + f"; ban matches N: {v['ban_matches']}; B2 removes {v['B2_doubt_removed']:.1f}% of doubt")
    if "outcome" in rep["readings"]["A"]:
        md.append(f"- Outcome: **{rep['readings']['A']['outcome']}**")
    md += ["", "### Part B — N vs the mapped 1.7B diff-of-means (DU)", ""]
    for m, v in Bres.items():
        if "N_minus_DU" in v:
            d = v["N_minus_DU"]
            md.append(f"- **{m}** ({v['DU']}): saving(N) − saving(DU) {d['saving']:+.1f} "
                      f"[{d['saving_ci'][0]:+.1f}, {d['saving_ci'][1]:+.1f}]; Δacc(N − DU) "
                      f"{d['dacc']:+.2f} [{d['dacc_ci'][0]:+.2f}, {d['dacc_ci'][1]:+.2f}]; "
                      f"N saves more: {v['N_saves_more']}; DU within 3: {v['DU_within3']}")
        else:
            md.append(f"- **{m}**: {v['DU']}")
    md += ["", "### Part C — hard128, N vs base per model", ""]
    for m, v in C.items():
        if m in ("cross", "scale_claim_holds", "noninferior_2pp"):
            continue
        md.append(f"- **{m}**: Δacc {v['dacc']:+.2f} [{v['dacc_ci'][0]:+.2f}, {v['dacc_ci'][1]:+.2f}], "
                  f"saving {v['saving']:.1f}% (base acc {v['base_acc']:.1f}%); AIME24+HMMT only: "
                  f"Δacc {v['sens_aime24_hmmt']['dacc']:+.2f} "
                  f"[{v['sens_aime24_hmmt']['dacc_ci'][0]:+.2f}, {v['sens_aime24_hmmt']['dacc_ci'][1]:+.2f}]")
    for k, v in C.get("cross", {}).items():
        md.append(f"- Δacc({k}) {v['ddacc']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]")
    if "scale_claim_holds" in C:
        md.append(f"- Scale claim holds: **{C['scale_claim_holds']}**")
    md += ["", "### Part D — N vs SEAL and DEER", ""]
    for k, v in D.items():
        parts = [f"{n}: Δsaving {x['saving']:+.1f} [{x['saving_ci'][0]:+.1f}, {x['saving_ci'][1]:+.1f}], "
                 f"Δacc {x['dacc']:+.2f} [{x['dacc_ci'][0]:+.2f}, {x['dacc_ci'][1]:+.2f}]"
                 for n, x in v["N_minus_X"].items()]
        md.append(f"- **{k}**: " + "; ".join(parts) + f" → N beats: {v['N_beats_X']}, X beats N: "
                  f"{v['X_beats_N']}" + (f"; tiebreak winner {v['tiebreak_winner']}"
                                         if "tiebreak_winner" in v else ""))
    (ROOT / B / "report.json").write_text(json.dumps(rep, indent=2, default=float) + "\n")
    (ROOT / B / "report.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
