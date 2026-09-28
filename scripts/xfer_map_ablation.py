#!/usr/bin/env python3
"""How robust is the transported direction to the map's training data?

Refit the raw U map (1.7B l20 -> 8B l) on subsets -- by origin and by size -- and
report cos(normalize(W_sub N), normalize(W_full N)) and the held-out doubt/plain AUC
of the subset direction. CPU is fine: ridge on <=37k x 2048.
"""
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from xfer_fit_maps import auc, ridge  # noqa: E402

torch.set_num_threads(64)
P = torch.load("data/xfer8b/pairs.pt", weights_only=False)
meta = P["meta"]
split = np.array([m["split"] for m in meta]); origin = np.array([m["origin"] for m in meta])
kind = np.array([m["kind"] for m in meta]); doubt = np.array([m["doubt"] for m in meta])
Hs = P["h_s"].float()
keep = (Hs.norm(dim=-1) < 5 * Hs.norm(dim=-1).median()).numpy()
for l in P["layers"]:
    nb = P["h_b"][l].float().norm(dim=-1); keep &= (nb < 5 * nb.median()).numpy()
N = torch.load("data/pool/dir_A_1trace.pt")["unit"].float()
te = (kind == "boundary") & (split == "test") & (doubt >= 0) & keep
y = (doubt[te] == 0).astype(int)
rng = np.random.default_rng(0)
for layer in (22, 26):
    Hb = P["h_b"][layer].float()
    def fitdir(mask, lam=1e-2):
        W, _ = ridge(Hs[torch.tensor(mask)], Hb[torch.tensor(mask)], lam)
        u = W @ N; return u / u.norm()
    full = fitdir((split == "train") & keep)
    print(f"L{layer} full: AUC {auc((Hb[torch.tensor(te)] @ full).numpy(), y):.3f}")
    subsets = {o: (split == "train") & keep & (origin == o) for o in ("web", "r17", "r8b")}
    subsets["reasoning (r17+r8b)"] = (split == "train") & keep & (origin != "web")
    tr_idx = np.flatnonzero((split == "train") & keep)
    for n in (1000, 3000, 10000):
        m = np.zeros(len(split), bool); m[rng.choice(tr_idx, n, replace=False)] = True
        subsets[f"random {n}"] = m
    for name, m in subsets.items():
        u = fitdir(m)
        print(f"  {name:22s} n={m.sum():6d} cos(full) {float(u @ full):.3f}  AUC {auc((Hb[torch.tensor(te)] @ u).numpy(), y):.3f}")
