#!/usr/bin/env python3
"""Fit 1.7B(l=20) <-> 8B(l) affine ridge maps and transport the N direction.

For every 8B layer in the pairs file:
  U : h_s -> h_b   raw affine ridge (N-direct: u_b = normalize(W_U @ N); bias cancels)
  E : h_b -> h_s   raw affine ridge (the input translator; also gives the
                   "readout" transport u_b = normalize(W_E^T @ N): the 8B direction
                   that most increases E(h_b)'s component along N)
Unit-space variants (x = h/||h||) are fitted as well, as the transfer notes require.

Candidate 8B directions written to data/xfer8b/dirs/L{l}_{name}.pt:
  NU     normalize(W_U N)             -- N-direct, raw
  NUu    normalize(W_U^unit N)        -- N-direct, unit space
  NEt    normalize(W_E^T N)           -- readout transport
  DU     normalize(W_U D17)           -- the 1.7B diff-of-means direction, mapped
  D8     8B-native diff of means, plain - doubt boundary (train split)
  RU     normalize(W_U R101)          -- mapped random control (same map as NU)
  R8     isotropic random in 8B space

Diagnostics on held-out TEST boundaries: AUC of <h, u> separating plain (next block
has no doubt opener) from doubt boundaries, next to N's AUC on the 1.7B states of
the SAME positions. Reported FVE uses a fixed train-mean denominator.

    uv run python scripts/xfer_fit_maps.py --pairs data/xfer8b/pairs.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


def ridge(X: torch.Tensor, Y: torch.Tensor, lam_rel: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Y ~ X W^T + b, closed form on centered data; lam relative to mean eigenvalue."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = X - mx, Y - my
    G = Xc.T @ Xc
    lam = lam_rel * G.diagonal().mean()
    W = torch.linalg.solve(G + lam * torch.eye(G.shape[0], device=G.device, dtype=G.dtype),
                           Xc.T @ Yc).T  # (d_out, d_in)
    return W, my - W @ mx


def fve(pred: torch.Tensor, Y: torch.Tensor, mu: torch.Tensor) -> float:
    return float(1 - ((pred - Y) ** 2).sum() / ((Y - mu) ** 2).sum())


def cosmean(pred: torch.Tensor, Y: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(pred, Y, dim=-1).mean())


def auc(score: np.ndarray, y: np.ndarray) -> float:
    """P(score_pos > score_neg), ties half."""
    order = np.argsort(score)
    ranks = np.empty(len(score))
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks for ties
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j + 2) / 2
        i = j + 1
    npos, nneg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def unit(v: torch.Tensor) -> torch.Tensor:
    return v / v.norm()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=Path, default=Path("data/xfer8b/pairs.pt"))
    ap.add_argument("--out", type=Path, default=Path("data/xfer8b/dirs"))
    ap.add_argument("--lams", type=float, nargs="+", default=[1e-4, 1e-3, 1e-2, 1e-1, 1.0])
    args = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    P = torch.load(args.pairs, weights_only=False)
    meta = P["meta"]
    split = np.array([m["split"] for m in meta])
    kind = np.array([m["kind"] for m in meta])
    doubt = np.array([m["doubt"] for m in meta])
    origin = np.array([m["origin"] for m in meta])
    # Qwen3-8B has rare massive-activation tokens mid-sequence (norm ~14k vs median
    # ~350 at l=26, all web digits); ten such rows dominate any squared loss. Drop
    # rows above 5x the median norm in either model before fitting and scoring.
    keep = (P["h_s"].float().norm(dim=-1) < 5 * P["h_s"].float().norm(dim=-1).median()).numpy()
    for l in P["layers"]:
        nb = P["h_b"][l].float().norm(dim=-1)
        keep &= (nb < 5 * nb.median()).numpy()
    print(f"dropping {(~keep).sum()} outlier-norm rows")
    tr, va, te = (torch.tensor((split == s) & keep) for s in ("train", "val", "test"))
    print({s: int((split == s).sum()) for s in ("train", "val", "test")},
          {o: int((origin == o).sum()) for o in np.unique(origin)},
          "boundaries", int((kind == "boundary").sum()), "doubt", int((doubt == 1).sum()))

    Hs = P["h_s"].float().to(dev)
    N = torch.load("data/pool/dir_A_1trace.pt")["unit"].float().to(dev)
    D17 = torch.load("data/dom/dir_D_diffmeans.pt")["unit"].float().to(dev)
    R17 = torch.load("data/dirs_random/dir_R101.pt")["unit"].float().to(dev)

    bnd_te = (kind == "boundary") & (split == "test") & (doubt >= 0) & keep
    y_te = (doubt[bnd_te] == 0).astype(int)  # plain = positive, like N (points plain-ward)
    ref = {"N@1.7B": auc((Hs[torch.tensor(bnd_te)] @ N).cpu().numpy(), y_te),
           "D17@1.7B": auc((Hs[torch.tensor(bnd_te)] @ D17).cpu().numpy(), y_te)}
    # Recompute a 1.7B diff of means on OUR boundaries as a pipeline sanity check.
    bnd_tr = torch.tensor((kind == "boundary") & (split == "train") & keep)
    pl, db = torch.tensor(doubt == 0), torch.tensor(doubt == 1)
    D17mine = unit(Hs[bnd_tr & pl].mean(0) - Hs[bnd_tr & db].mean(0))
    ref["cos(D17 mine, D17 file)"] = float(D17mine @ D17)
    ref["cos(D17 mine, N)"] = float(D17mine @ N)
    ref["D17mine@1.7B"] = auc((Hs[torch.tensor(bnd_te)] @ D17mine).cpu().numpy(), y_te)
    print("1.7B reference:", json.dumps(ref, indent=1))

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"reference_1p7b": ref, "n_test_boundaries": int(bnd_te.sum()),
                    "test_doubt_rate": float(1 - y_te.mean()), "layers": {}}
    gen = torch.Generator().manual_seed(8)
    for layer in P["layers"]:
        Hb = P["h_b"][layer].float().to(dev)
        r: dict = {"norm_b_median": float(Hb.norm(dim=-1).median()),
                   "norm_s_median": float(Hs.norm(dim=-1).median())}
        # --- raw U: s -> b, lambda chosen on val
        best = None
        for lam in args.lams:
            W, b = ridge(Hs[tr], Hb[tr], lam)
            f = fve(Hs[va] @ W.T + b, Hb[va], Hb[tr].mean(0))
            best = max(best or (-9, 0, None, None), (f, lam, W, b), key=lambda t: t[0])
        fU, lamU, WU, bU = best
        r["U_raw"] = {"lam": lamU, "fve_val": fU,
                      "fve_test": fve(Hs[te] @ WU.T + bU, Hb[te], Hb[tr].mean(0)),
                      "cos_test": cosmean(Hs[te] @ WU.T + bU, Hb[te])}
        # --- raw E: b -> s
        best = None
        for lam in args.lams:
            W, b = ridge(Hb[tr], Hs[tr], lam)
            f = fve(Hb[va] @ W.T + b, Hs[va], Hs[tr].mean(0))
            best = max(best or (-9, 0, None, None), (f, lam, W, b), key=lambda t: t[0])
        fE, lamE, WE, bE = best
        r["E_raw"] = {"lam": lamE, "fve_val": fE,
                      "fve_test": fve(Hb[te] @ WE.T + bE, Hs[te], Hs[tr].mean(0)),
                      "cos_test": cosmean(Hb[te] @ WE.T + bE, Hs[te])}
        # --- unit-space U
        xs, xb = Hs / Hs.norm(dim=-1, keepdim=True), Hb / Hb.norm(dim=-1, keepdim=True)
        best = None
        for lam in args.lams:
            W, b = ridge(xs[tr], xb[tr], lam)
            f = fve(xs[va] @ W.T + b, xb[va], xb[tr].mean(0))
            best = max(best or (-9, 0, None, None), (f, lam, W, b), key=lambda t: t[0])
        fUu, lamUu, WUu, bUu = best
        r["U_unit"] = {"lam": lamUu, "fve_val": fUu,
                       "fve_test": fve(xs[te] @ WUu.T + bUu, xb[te], xb[tr].mean(0))}

        D8 = unit(Hb[bnd_tr & pl].mean(0) - Hb[bnd_tr & db].mean(0))
        R8 = unit(torch.randn(Hb.shape[1], generator=gen).to(dev))
        dirs = {"NU": unit(WU @ N), "NUu": unit(WUu @ N), "NEt": unit(WE.T @ N),
                "DU": unit(WU @ D17), "D8": D8, "RU": unit(WU @ R17), "R8": R8}
        # How much of N survives the round trip b-space -> E -> s-space
        r["cos(E(NU)-E(0), N)"] = float(unit(WE @ dirs["NU"]) @ N)
        Hbt = Hb[torch.tensor(bnd_te)]
        r["auc_test"] = {k: auc((Hbt @ v).cpu().numpy(), y_te) for k, v in dirs.items()}
        r["cos"] = {f"{a},{b2}": float(dirs[a] @ dirs[b2])
                    for a in dirs for b2 in dirs if a < b2}
        for k, v in dirs.items():
            torch.save({"unit": v.cpu(), "layer": layer, "method": k,
                        "source": "xfer_fit_maps.py", "pairs": str(args.pairs)},
                       args.out / f"L{layer}_{k}.pt")
        torch.save({"W_U": WU.cpu(), "b_U": bU.cpu(), "W_E": WE.cpu(), "b_E": bE.cpu(),
                    "lam_U": lamU, "lam_E": lamE}, args.out / f"L{layer}_maps.pt")
        report["layers"][layer] = r
        print(f"L{layer}: U fve {fU:.3f} (test {r['U_raw']['fve_test']:.3f}, cos "
              f"{r['U_raw']['cos_test']:.3f})  E fve {fE:.3f}  Uunit {fUu:.3f} | AUC "
              + " ".join(f"{k}={v:.3f}" for k, v in r["auc_test"].items())
              + f" | cos(NU,D8)={r['cos']['D8,NU']:.3f} cos(NU,NEt)={r['cos']['NEt,NU']:.3f}",
              flush=True)
        del Hb
    (args.out / "fit_report.json").write_text(json.dumps(report, indent=1))
    print(f"wrote {args.out / 'fit_report.json'}")


if __name__ == "__main__":
    main()
