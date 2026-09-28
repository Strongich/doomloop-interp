#!/usr/bin/env python3
"""Frozen 1.7B NLA as a read/write interface on Qwen3-8B states (8B layer 22).

  verbalize --which s   AV(h_s)            explanations of the 1.7B state (source path)
  verbalize --which b   AV(1000 q(E h_b))  explanations of the PAIRED 8B state (transfer)
  analyze               AR reconstructions, U_AR calibration, bottleneck metrics,
                        grounding, and pooled edit directions (N-interface)

Row sets come from data/xfer8b/pairs.pt, chosen deterministically (seed 0):
  calib   train split, 500 per origin (web / r17 / r8b)  -- fits U_AR only
  eval    test split, 200 per origin                     -- all reported metrics
  der8b   train split, 8B-trace boundaries whose next block is doubt, 400
          -- the target endpoints for N-interface
  der17   the same from 1.7B traces, 400 -- source endpoints (1.7B), for the ablation
The same rows get both explanations, so every comparison is paired.

E and U are the raw affine ridge maps in dirs_clean/L22_maps.pt (outlier-filtered).
The AV input is always rescaled to the checkpoint's 1000 (nla_meta.yaml).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "data/xfer8b/interface"
LAYER = 22
PAIRS = ROOT / "data/xfer8b/pairs.pt"
MAPS = ROOT / "data/xfer8b/dirs_clean"
DIRS_OUT = ROOT / "data/xfer8b/dirs_iface"
TARGET_ORIGIN = "r8b"


def rows(P: dict) -> dict[str, np.ndarray]:
    meta = P["meta"]
    split = np.array([m["split"] for m in meta]); origin = np.array([m["origin"] for m in meta])
    kind = np.array([m["kind"] for m in meta]); doubt = np.array([m["doubt"] for m in meta])
    keep = (P["h_s"].float().norm(dim=-1) < 5 * P["h_s"].float().norm(dim=-1).median()).numpy()
    for l in P["layers"]:
        nb = P["h_b"][l].float().norm(dim=-1); keep &= (nb < 5 * nb.median()).numpy()
    rng = np.random.default_rng(0)
    out, used = {}, set()

    def take(mask: np.ndarray, n: int) -> np.ndarray:
        idx = np.array([i for i in np.flatnonzero(mask & keep) if i not in used])
        pick = np.sort(rng.choice(idx, min(n, len(idx)), replace=False))
        used.update(pick.tolist())
        return pick

    out["der8b"] = take((split == "train") & (origin == TARGET_ORIGIN) & (kind == "boundary") & (doubt == 1), 400)
    out["der17"] = take((split == "train") & (origin == "r17") & (kind == "boundary") & (doubt == 1), 400)
    out["calib"] = np.concatenate([take((split == "train") & (origin == o), 500) for o in ("web", "r17", TARGET_ORIGIN)])
    out["eval"] = np.concatenate([take((split == "test") & (origin == o), 200) for o in ("web", "r17", TARGET_ORIGIN)])
    return out


def E_of(P: dict, idx: np.ndarray) -> torch.Tensor:
    m = torch.load(MAPS / f"L{LAYER}_maps.pt")
    hb = P["h_b"][LAYER][idx].float()
    return hb @ m["W_E"].T + m["b_E"]


def cmd_verbalize(args: argparse.Namespace) -> None:
    from xfer_nla_tools import FrozenNLA

    P = torch.load(PAIRS, weights_only=False)
    R = rows(P)
    f = FrozenNLA()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, idx in R.items():
        if args.which == "s" and name == "der8b":
            continue  # source explanations of 8B-trace doubt sites are not needed
        if args.which == "b" and name == "der17":
            continue
        path = OUT / f"z_{args.which}_{name}.json"
        if path.exists():
            continue
        vec = P["h_s"][idx].float() if args.which == "s" else E_of(P, idx)
        z = f.verbalize(vec)
        path.write_text(json.dumps({"rows": idx.tolist(), "z": z}))
        print(f"{args.which} {name}: {len(z)} explanations, empty {sum(not t for t in z)}", flush=True)


def fve(pred: torch.Tensor, gold: torch.Tensor, mu: torch.Tensor) -> float:
    return float(1 - ((pred - gold) ** 2).sum() / ((gold - mu) ** 2).sum())


def q(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(a, b, dim=-1).mean())


DOUBT_CUES = ("wait", "double-check", "double check", "verify", "verif", "reconsider", "re-check",
              "recheck", "mistake", "hmm", "but ", "however", "actually", "hold on", "check")


def predicts_doubt(z: str) -> bool:
    paras = [p for p in z.split("\n\n") if p.strip()] or [z]
    last = paras[-1].lower()
    return any(c in last for c in DOUBT_CUES)


def cmd_analyze(args: argparse.Namespace) -> None:
    from steer_demo import CONTINUE_PARAGRAPH, DOUBT_PARAGRAPH, edit_explanation
    from xfer_fit_maps import auc, ridge
    from xfer_nla_tools import FrozenNLA

    P = torch.load(PAIRS, weights_only=False)
    R = rows(P)
    Z = {}
    for w in ("s", "b"):
        for name in R:
            p = OUT / f"z_{w}_{name}.json"
            if p.exists():
                d = json.loads(p.read_text())
                assert d["rows"] == R[name].tolist(), f"row mismatch {p}"
                Z[(w, name)] = d["z"]
    f = FrozenNLA()
    maps = torch.load(MAPS / f"L{LAYER}_maps.pt")
    WU, bU = maps["W_U"].cuda(), maps["b_U"].cuda()
    Hs = P["h_s"].float().cuda(); Hb = P["h_b"][LAYER].float().cuda()
    s_norm = float(Hs.norm(dim=-1).median())
    meta = P["meta"]

    def AR(texts: list[str]) -> torch.Tensor:
        return f.reconstruct(texts).cuda()

    rep: dict = {}
    # ---------------- bottleneck ----------------
    ev, ca = R["eval"], R["calib"]
    tr_mask = np.array([m["split"] == "train" for m in meta])
    mu_s = q(Hs[torch.tensor(tr_mask)]).mean(0); mu_b = q(Hb[torch.tensor(tr_mask)]).mean(0)
    xs, xb = q(Hs[ev]), q(Hb[ev])
    Eb = (Hb[ev] @ maps["W_E"].cuda().T + maps["b_E"].cuda())
    r_s, r_b = AR(Z[("s", "eval")]), AR(Z[("b", "eval")])
    r_b_cal = AR(Z[("b", "calib")]); r_s_cal = AR(Z[("s", "calib")])
    perm = torch.randperm(len(ev), generator=torch.Generator().manual_seed(1)).cuda()

    def U_state(a: torch.Tensor) -> torch.Tensor:  # raw U applied at source-typical norm
        return q(q(a) * s_norm @ WU.T + bU)

    # U_AR: unit-space ridge q(AR(z_b)) -> q(h_b), fitted on calib only
    def ridge_cv(X: torch.Tensor, Y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        """lambda by a 80/20 holdout inside calib (fixed split), then refit on all calib."""
        n = len(X); cut = int(0.8 * n)
        order = torch.randperm(n, generator=torch.Generator().manual_seed(2)).cuda()
        tr_, va_ = order[:cut], order[cut:]
        best = max((fve(q(X[va_] @ W.T + b0), Y[va_], mu_b), lam)
                   for lam in (1e-2, 1e-1, 1.0, 10.0, 100.0)
                   for W, b0 in [ridge(X[tr_], Y[tr_], lam)])
        W, b0 = ridge(X, Y, best[1])
        return W, b0, best[1]

    WA, bA, lamA = ridge_cv(q(r_b_cal), q(Hb[ca]))
    rep["U_AR_lambda"] = lamA
    def U_AR(a: torch.Tensor) -> torch.Tensor:
        return q(q(a) @ WA.T + bA)
    # the same calibration, but on SOURCE explanations (tests U_AR on the source path)
    WAs, bAs, _ = ridge_cv(q(r_s_cal), q(Hb[ca]))

    b = {}
    b["1.7B space: cos(AR(AV(h_s)), h_s)  [source NLA]"] = cos(r_s, xs)
    b["1.7B space: cos(E(h_b), h_s)       [continuous]"] = cos(Eb, xs)
    b["1.7B space: cos(AR(AV(E h_b)), h_s) [8B expl -> paired 1.7B]"] = cos(r_b, xs)
    b["1.7B space: cos(AR(AV(E h_b)), E h_b)"] = cos(r_b, Eb)
    b["1.7B space: FVE source"] = fve(q(r_s), xs, mu_s)
    b["1.7B space: FVE 8B-explanation vs h_s"] = fve(q(r_b), xs, mu_s)
    b["8B space: direct U(h_s)"] = (cos(U_state(Hs[ev]), xb), fve(U_state(Hs[ev]), xb, mu_b))
    b["8B space: continuous U(E h_b)"] = (cos(U_state(Eb), xb), fve(U_state(Eb), xb, mu_b))
    b["8B space: source text U(AR(AV h_s))"] = (cos(U_state(r_s), xb), fve(U_state(r_s), xb, mu_b))
    b["8B space: transferred U(AR(AV E h_b))"] = (cos(U_state(r_b), xb), fve(U_state(r_b), xb, mu_b))
    b["8B space: calibrated U_AR(AR(AV E h_b))"] = (cos(U_AR(r_b), xb), fve(U_AR(r_b), xb, mu_b))
    b["8B space: calibrated-src U_ARs(AR(AV h_s))"] = (
        cos(q(q(r_s) @ WAs.T + bAs), xb), fve(q(q(r_s) @ WAs.T + bAs), xb, mu_b))
    b["8B space: mean-state control"] = (cos(mu_b.expand_as(xb), xb), 0.0)
    b["8B space: shuffled-explanation control U_AR"] = (cos(U_AR(r_b[perm]), xb), fve(U_AR(r_b[perm]), xb, mu_b))
    org = np.array([meta[i]["origin"] for i in ev])
    for o in ("web", "r17", TARGET_ORIGIN):
        m = torch.tensor(org == o).cuda()
        b[f"8B space by origin {o}: calibrated U_AR FVE"] = fve(U_AR(r_b[m]), xb[m], mu_b)
        b[f"8B space by origin {o}: shuffled FVE"] = fve(U_AR(r_b[perm][m]), xb[m], mu_b)
    rep["bottleneck"] = b
    for k, v in b.items():
        print(f"{k:62s} {v}")

    # ---------------- grounding: does the explanation predict the next block? --------
    evb = [j for j, i in enumerate(ev) if meta[i]["kind"] == "boundary" and meta[i]["doubt"] >= 0]
    y = np.array([meta[ev[j]]["doubt"] for j in evb])
    g = {"n_boundary_eval": len(evb), "doubt_rate": float(y.mean())}
    for w in ("s", "b"):
        pr = np.array([predicts_doubt(Z[(w, "eval")][j]) for j in evb])
        g[f"z_{w}: P(pred doubt | doubt next)"] = float(pr[y == 1].mean())
        g[f"z_{w}: P(pred doubt | plain next)"] = float(pr[y == 0].mean())
    ps = np.array([predicts_doubt(Z[("s", "eval")][j]) for j in evb])
    pb = np.array([predicts_doubt(Z[("b", "eval")][j]) for j in evb])
    g["agreement z_s vs z_b on doubt cue"] = float((ps == pb).mean())
    # a continuous readout: AR(z) projected on N separates doubt/plain?
    N = torch.load(ROOT / "data/pool/dir_A_1trace.pt")["unit"].float().cuda()
    evb_t = torch.tensor(evb).cuda()
    for w, r in (("s", r_s), ("b", r_b)):
        g[f"AUC plain-vs-doubt of <AR(z_{w}), N>"] = auc((q(r[evb_t]) @ N).cpu().numpy(), (y == 0).astype(int))
    g["AUC of <h_s, N> (reference)"] = auc((xs[evb_t] @ N).cpu().numpy(), (y == 0).astype(int))
    rep["grounding"] = g
    for k, v in g.items():
        print(f"{k:62s} {v}")
    ex = []
    for j in evb[:6]:
        ex.append({"doubt_next": int(meta[ev[j]]["doubt"]), "z_s": Z[("s", "eval")][j], "z_b": Z[("b", "eval")][j]})
    (OUT / "examples.json").write_text(json.dumps(ex, indent=1))

    # ---------------- N-interface: pooled English-edit directions -------------------
    dirs = {}
    for name, w in (("der8b", "b"), ("der17", "s")):
        z0 = Z[(w, name)]
        ok = [i for i, z in enumerate(z0) if len([p for p in z.split("\n\n") if p.strip()]) >= 2]
        z_c = [edit_explanation(z0[i], CONTINUE_PARAGRAPH) for i in ok]
        z_d = [edit_explanation(z0[i], DOUBT_PARAGRAPH) for i in ok]
        a0, ac, ad = AR([z0[i] for i in ok]), AR(z_c), AR(z_d)
        raw = (ac - a0)                                   # 1.7B-space deltas (N's convention)
        rep[f"{name}_n_valid"] = len(ok)
        mean_raw = raw.mean(0)
        dirs[f"{name}_raw_1p7"] = q(mean_raw)
        # 8B-space via the raw affine U: linear part only (bias cancels)
        dirs[f"{name}_WU"] = q(WU @ mean_raw)
        # full F with per-endpoint normalization and the calibrated U_AR (the notes' F)
        dF = U_AR(ac) - U_AR(a0)
        dirs[f"{name}_F"] = q(q(dF).mean(0))
        dFd = U_AR(ad) - U_AR(a0)
        dirs[f"{name}_F_doubt"] = q(q(dFd).mean(0))
        # one-trace variant, mimicking N (single trace with the most doubt sites)
        if name == "der8b":
            docs = np.array([meta[R[name][i]]["doc_id"] for i in ok])
            top = max(set(docs.tolist()), key=lambda d: (docs == d).sum())
            m = torch.tensor(docs == top).cuda()
            rep["der8b_one_trace"] = {"doc": top, "n": int(m.sum())}
            dirs["der8b_1trace_F"] = q(q(dF[m]).mean(0))
            dirs["der8b_1trace_WU"] = q(WU @ raw[m].mean(0))
    NU = torch.load(MAPS / f"L{LAYER}_NU.pt")["unit"].float().cuda()
    D8 = torch.load(MAPS / f"L{LAYER}_D8.pt")["unit"].float().cuda()
    ref = {"NU22 (N-direct)": NU, "D8 (8B diff-of-means)": D8}
    c = {}
    c["cos(der17_raw_1p7, N)  [re-derived 1.7B N vs the file]"] = float(dirs["der17_raw_1p7"] @ N)
    c["cos(der8b_raw_1p7, N)  [8B-derived 1.7B-space delta vs N]"] = float(dirs["der8b_raw_1p7"] @ N)
    for k, v in dirs.items():
        if v.shape[0] == Hb.shape[1]:
            for rn, rv in ref.items():
                c[f"cos({k}, {rn})"] = float(v @ rv)
    c["cos(der8b_F, der8b_F_doubt)"] = float(dirs["der8b_F"] @ dirs["der8b_F_doubt"])
    # held-out readout AUC on 8B test boundaries, as in fit_report
    tb = [i for i, m in enumerate(meta) if m["split"] == "test" and m["kind"] == "boundary" and m["doubt"] >= 0]
    yb = np.array([meta[i]["doubt"] == 0 for i in tb]).astype(int)
    for k, v in list(dirs.items()) + list(ref.items()):
        if v.shape[0] == Hb.shape[1]:
            c[f"AUC {k}"] = auc((Hb[tb] @ v).cpu().numpy(), yb)
    rep["directions"] = c
    for k, v in c.items():
        print(f"{k:62s} {v:.3f}")
    DIRS_OUT.mkdir(parents=True, exist_ok=True)
    for k, v in dirs.items():
        if v.shape[0] == Hb.shape[1]:
            torch.save({"unit": v.cpu(), "layer": LAYER, "method": k, "source": "xfer_interface.py"},
                       DIRS_OUT / f"L{LAYER}_{k}.pt")
    (OUT / "report.json").write_text(json.dumps(rep, indent=1, default=float))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verbalize"); v.add_argument("--which", choices=("s", "b"), required=True)
    sub.add_parser("analyze")
    for sp in sub.choices.values():
        sp.add_argument("--layer", type=int, default=22)
        sp.add_argument("--root", default="data/xfer8b", help="pairs.pt, dirs_clean/ live here")
        sp.add_argument("--target-origin", default="r8b", help="origin label of target-model traces")
    args = ap.parse_args()
    global LAYER, PAIRS, MAPS, DIRS_OUT, OUT, TARGET_ORIGIN
    LAYER = args.layer
    TARGET_ORIGIN = args.target_origin
    base = ROOT / args.root
    PAIRS, MAPS, DIRS_OUT, OUT = base / "pairs.pt", base / "dirs_clean", base / "dirs_iface", base / "interface"
    {"verbalize": cmd_verbalize, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
