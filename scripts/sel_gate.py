#!/usr/bin/env python3
r"""EXPERIMENT-selective-doubt.md §S4 gate: does the boundary state separate productive from
redundant doubt?

  rows      productive (1) vs redundant (0) episodes, source traces
  state     raw layer-20 h at the *before* boundary; StandardScaler fit on training folds
  covs      level, token index, paragraph index, doubt paragraphs so far, stability
  model     L2 logistic regression, C by inner 5-fold CV grouped by question (AUC)
  outer     5-fold StratifiedGroupKFold by question, seed 20261002 -> out-of-fold scores
  metric    OOF AUC, question-clustered bootstrap CI (2,000 reps); paired diff state - covs
  extras    cos(N, w_hat) with w_hat = unit probe direction in raw h space (fit on all rows);
            AUC of <h, N> alone
  pass      >= 150 productive from >= 75 questions; AUC >= 0.65; diff >= 0.05 with CI > 0

    uv run python scripts/sel_gate.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261002
COVS = ["cov_level", "cov_token", "cov_paragraph", "cov_doubt_so_far", "cov_stable"]
CS = np.logspace(-4, 1, 6)


def fit(X, y, g, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000))
    inner = StratifiedGroupKFold(5, shuffle=True, random_state=seed)
    gs = GridSearchCV(pipe, {"logisticregression__C": CS}, scoring="roc_auc",
                      cv=list(inner.split(X, y, g)), n_jobs=6)
    return gs.fit(X, y)


def oof(X, y, g):
    from sklearn.model_selection import StratifiedGroupKFold

    s = np.zeros(len(y))
    chosen = []
    for f, (tr, te) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=SEED).split(X, y, g)):
        m = fit(X[tr], y[tr], g[tr], SEED + f + 1)
        s[te] = m.decision_function(X[te])
        chosen.append(float(m.best_params_["logisticregression__C"]))
    return s, chosen


def main() -> None:
    from sklearn.metrics import roc_auc_score

    sel = ROOT / "data/selective"
    eps = [json.loads(x) for x in open(sel / "episodes_source.jsonl")]
    st = [torch.load(p, weights_only=False) for p in sorted((sel / "states").glob("source_s*.pt"))]
    H = torch.cat([s["h"] for s in st]).float()
    idx = {k: i for i, k in enumerate(tuple(x) for s in st for x in s["index"])}
    rows = [e for e in eps if e["label"] in ("productive", "redundant")]
    Xh = H[[idx[(e["key"], e["k"])] for e in rows]].numpy()
    Xc = np.array([[e[c] for c in COVS] for e in rows], dtype=float)
    y = np.array([e["label"] == "productive" for e in rows], dtype=int)
    qids = sorted({e["question_id"] for e in rows})
    g = np.array([qids.index(e["question_id"]) for e in rows])
    n_prod, q_prod = int(y.sum()), len({e["question_id"] for e, t in zip(rows, y) if t})
    print(f"{len(rows)} rows: {n_prod} productive from {q_prod} questions, {len(y) - n_prod} redundant",
          flush=True)

    s_h, c_h = oof(Xh, y, g)
    s_c, c_c = oof(Xc, y, g)
    N = torch.load(ROOT / "data/pool/dir_A_1trace.pt", weights_only=False)["unit"].float().numpy()
    s_n = Xh @ N
    full = fit(Xh, y, g, SEED)
    lr = full.best_estimator_
    w = lr[-1].coef_[0] / lr[0].scale_
    w_hat = w / np.linalg.norm(w)
    np.save(sel / "w_hat.npy", w_hat)

    rng = np.random.default_rng(SEED)
    members = [np.flatnonzero(g == q) for q in range(len(qids))]
    boot = {"state": [], "covs": [], "diff": [], "projN": []}
    for _ in range(2000):
        ii = np.concatenate([members[q] for q in rng.integers(0, len(qids), len(qids))])
        if y[ii].min() == y[ii].max():
            continue
        a, b = roc_auc_score(y[ii], s_h[ii]), roc_auc_score(y[ii], s_c[ii])
        boot["state"].append(a)
        boot["covs"].append(b)
        boot["diff"].append(a - b)
        boot["projN"].append(roc_auc_score(y[ii], s_n[ii]))
    ci = {k: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
          for k, v in boot.items()}
    auc = {"state": roc_auc_score(y, s_h), "covs": roc_auc_score(y, s_c),
           "projN": roc_auc_score(y, s_n)}
    auc["diff"] = auc["state"] - auc["covs"]
    checks = {"n_productive>=150": n_prod >= 150, "productive_questions>=75": q_prod >= 75,
              "state_auc>=0.65": auc["state"] >= 0.65,
              "diff>=0.05_and_ci>0": auc["diff"] >= 0.05 and ci["diff"][0] > 0}
    rep = {"rows": len(rows), "productive": n_prod, "productive_questions": q_prod,
           "redundant": int(len(y) - n_prod), "questions": len(qids),
           "auc": {k: round(float(v), 4) for k, v in auc.items()}, "ci95": ci,
           "C_outer_state": c_h, "C_outer_covs": c_c,
           "C_full": float(full.best_params_["logisticregression__C"]),
           "cos_N_what": round(float(N @ w_hat), 4),
           "checks": checks, "PASS": all(checks.values())}
    (sel / "gate.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    sys.exit(main())
