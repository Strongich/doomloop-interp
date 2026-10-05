#!/usr/bin/env python3
r"""EXPERIMENT-probe-gated.md §G3: the boundary probe, its thresholds and the go/stop gate.

  data     data/gated/states_source.pt (gate_states.py): every boundary of the 1,000 source
           traces; y = 1 before a productive doubt episode
  split    by question, 80% fit / 20% held-out, seed 20261003
  model    StandardScaler + L2 LogisticRegression(class_weight="balanced"); C by inner 5-fold
           StratifiedGroupKFold by question on the fit split (AUC)
  export   raw coordinates: w = coef / scale, b = intercept - sum(coef * mean / scale);
           the hook scores s = <h, w> + b on the pre-edit block output
  held-out AUC (question-clustered bootstrap, 2,000 reps); tau_q = the (1 - q) quantile of
           held-out scores, q in {5, 10, 20}%; recall of positives with s > tau_q
  gate     recall at q = 10% >= 25%

Output: data/gated/probe.pt {w, b, tau: {"5","10","20"}, ...}, data/gated/probe_report.json

    uv run python scripts/gate_probe.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/gated"
SEED = 20261003
CS = np.logspace(-5, 0, 6)
QS = (5, 10, 20)


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    d = torch.load(OUT / "states_source.pt", weights_only=False)
    X = d["h"].float().numpy()
    rows = d["rows"]
    y = np.array([r["y"] for r in rows])
    qid = np.array([r["question_id"] for r in rows])
    g = np.array([r["g"] for r in rows])
    questions = sorted(set(qid))
    random.Random(SEED).shuffle(questions)
    fit_q = set(questions[: int(round(0.8 * len(questions)))])
    tr = np.array([q in fit_q for q in qid])
    te = ~tr
    print(f"{len(y)} boundaries, {y.sum()} positive; fit {tr.sum()} ({y[tr].sum()} pos), "
          f"held-out {te.sum()} ({y[te].sum()} pos)", flush=True)

    pipe = make_pipeline(StandardScaler(),
                         LogisticRegression(class_weight="balanced", max_iter=3000))
    inner = StratifiedGroupKFold(5, shuffle=True, random_state=SEED)
    gs = GridSearchCV(pipe, {"logisticregression__C": CS}, scoring="roc_auc",
                      cv=list(inner.split(X[tr], y[tr], qid[tr])), n_jobs=6, refit=True)
    gs.fit(X[tr], y[tr])
    sc, lr = gs.best_estimator_.named_steps.values()
    coef = lr.coef_[0]
    w = coef / sc.scale_
    b = float(lr.intercept_[0] - np.sum(coef * sc.mean_ / sc.scale_))
    s = X @ w + b
    ref = gs.best_estimator_.decision_function(X[te])
    assert np.allclose(s[te], ref, atol=1e-3 * np.abs(ref).max()), "raw export mismatch"

    st, yt, qt, gt = s[te], y[te], qid[te], g[te]
    auc = float(roc_auc_score(yt, st))
    uq = sorted(set(qt))
    by = {q: np.flatnonzero(qt == q) for q in uq}
    rng = np.random.default_rng(SEED)
    boots = []
    for _ in range(2000):
        ii = np.concatenate([by[uq[j]] for j in rng.integers(0, len(uq), len(uq))])
        if 0 < yt[ii].sum() < len(ii):
            boots.append(roc_auc_score(yt[ii], st[ii]))
    tau, recall = {}, {}
    for q in QS:
        t = float(np.quantile(st, 1 - q / 100))
        tau[str(q)] = t
        fire = st > t
        rec = float(fire[yt == 1].mean())
        rb = []
        rng = np.random.default_rng(SEED)
        for _ in range(2000):
            ii = np.concatenate([by[uq[j]] for j in rng.integers(0, len(uq), len(uq))])
            if yt[ii].sum():
                rb.append(fire[ii][yt[ii] == 1].mean())
        late = gt >= 256
        recall[str(q)] = {
            "recall": round(rec, 4),
            "recall_ci": [round(float(np.percentile(rb, 2.5)), 4),
                          round(float(np.percentile(rb, 97.5)), 4)],
            "vs_random": round(rec / (q / 100), 2),
            "fire_rate_all": round(float(fire.mean()), 4),
            "fire_rate_g256": round(float(fire[late].mean()), 4),
            "recall_g256": round(float(fire[late & (yt == 1)].mean()), 4),
        }
    go = recall["10"]["recall"] >= 0.25
    rep = {
        "boundaries": int(len(y)), "positives": int(y.sum()),
        "fit": {"rows": int(tr.sum()), "pos": int(y[tr].sum()), "questions": len(fit_q)},
        "heldout": {"rows": int(te.sum()), "pos": int(yt.sum()), "questions": len(uq)},
        "C": float(gs.best_params_["logisticregression__C"]),
        "inner_cv_auc": {str(c): round(float(m), 4)
                         for c, m in zip(CS, gs.cv_results_["mean_test_score"])},
        "heldout_auc": round(auc, 4),
        "heldout_auc_ci": [round(float(np.percentile(boots, 2.5)), 4),
                           round(float(np.percentile(boots, 97.5)), 4)],
        "tau": tau, "recall": recall,
        "gate": {"rule": "held-out recall at q=10% >= 0.25", "GO": go},
    }
    torch.save({"w": torch.tensor(w, dtype=torch.float32), "b": b, "tau": tau,
                "C": rep["C"], "heldout_auc": auc, "layer": d["layer"],
                "note": "s = <h, w> + b on the raw pre-edit layer-20 block output; fire if s > tau[q]"},
               OUT / "probe.pt")
    (OUT / "probe_report.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
