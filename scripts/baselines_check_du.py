#!/usr/bin/env python3
"""EXPERIMENT-reviewer-baselines.md §B3 pre-check: DU and NU of each target come from the
same map fit. Recomputes normalize(W_U @ D_1.7B) and normalize(W_U @ N_1.7B) from L*_maps.pt
and asserts cos > 0.9999 with the stored DU / NU; reports cos(DU, NU) and cos(DU, native D).
Writes data/baselines/du_check.json."""
import json
import torch

N17 = torch.load("data/pool/dir_A_1trace.pt", map_location="cpu", weights_only=False)["unit"].double()
D17 = torch.load("data/dom/dir_D_diffmeans.pt", map_location="cpu", weights_only=False)["unit"].double()
cos = lambda a, b: float(torch.nn.functional.cosine_similarity(a.double(), b.double(), dim=0))
unit = lambda v: v / v.norm()
out = {}
for name, d in (("8b", "data/xfer8b/dirs_clean/L26"), ("30b", "data/xfer30b/dirs_L34/L34")):
    m = torch.load(f"{d}_maps.pt", map_location="cpu", weights_only=False)
    L = lambda k: torch.load(f"{d}_{k}.pt", map_location="cpu", weights_only=False)["unit"].double()
    WU = m["W_U"].double()
    du_re, nu_re = unit(WU @ D17), unit(WU @ N17)
    r = {"cos_DU_recomputed": cos(L("DU"), du_re), "cos_NU_recomputed": cos(L("NU"), nu_re),
         "cos_DU_NU": cos(L("DU"), L("NU")), "cos_DU_nativeD": cos(L("DU"), L("D8")),
         "cos_NU_nativeD": cos(L("NU"), L("D8")), "cos_N17_D17": cos(N17, D17)}
    assert r["cos_DU_recomputed"] > 0.9999 and r["cos_NU_recomputed"] > 0.9999, (name, r)
    out[name] = r
print(json.dumps(out, indent=2))
open("data/baselines/du_check.json", "w").write(json.dumps(out, indent=2) + "\n")
