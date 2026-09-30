#!/usr/bin/env python3
"""Ceiling check (EXPERIMENT-transfer-claim4.md): how well can ANY map from the 1.7B's
layer-20 state predict the target state? That bounds the adapted NLA's target-space FVE.

Train rows: the mix-map train split (old pairs + fresh web4k/chat, xfer_refit_mix.py).
Eval rows:  T1 positions (data/xfer_claim4/t1), never fitted on; reported per domain.
Metric:     the T1 map ceiling M = fve_nrm(U(h_s), h_b) and cycle C = fve_nrm(U(E h_b), h_b),
            with U applied under the interface convention q(q(x) * s_norm @ W^T + b).

  curve  affine ridge (lambda 0.1) on 10/25/50/100% of train rows -> M, C
  mlp    MLP h_s -> h_b trained on the fve_nrm numerator ||q(f(x)) - q(y)||^2, early-stopped
         on val, next to an affine map trained identically -> held-out FVE

    CUDA_VISIBLE_DEVICES=4 uv run python scripts/xfer_ceiling.py --target 8b
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

from xfer_fidelity import fve_nrm, q  # noqa: E402
from xfer_fit_maps import ridge  # noqa: E402
from xfer_refit_mix import OUT as MIX, T  # noqa: E402


def load_rows(target: str, dev: str) -> dict:
    cfg = T[target]
    L = cfg["layer"]
    items = torch.load(MIX / "items.pt", weights_only=False)["items"]
    P = torch.load(ROOT / cfg["pairs"], weights_only=False)
    Hs = torch.cat([P["h_s"].float(), torch.load(MIX / "h_s.pt").float()])
    Hb = torch.cat([P["h_b"][L].float(), torch.load(MIX / f"h_{target}.pt").float()])
    split = np.concatenate([[m["split"] for m in P["meta"]],
                            [it["split"] for it in items for _ in it["pos"]]])
    keep = np.ones(len(split), bool)
    for H in (Hs, Hb):
        n = H.norm(dim=-1)
        keep &= (n < 5 * n.median()).numpy()
    tr = np.flatnonzero((split == "train") & keep)
    va = np.flatnonzero((split == "val") & keep)
    t1 = torch.load(ROOT / "data/xfer_claim4/t1/items.pt", weights_only=False)["items"]
    origin = np.array([it["origin"] for it in t1 for _ in it["pos"]])
    es = torch.load(ROOT / "data/xfer_claim4/t1/h_s.pt")["h"].float()
    eb = torch.load(ROOT / f"data/xfer_claim4/t1/h_{target}.pt")["h"].float()
    nb = eb.norm(dim=-1)
    ok = (nb < 5 * nb.median()).numpy()
    groups = {"web+chat": np.isin(origin, ["web", "chat"]) & ok, "web": (origin == "web") & ok,
              "chat": (origin == "chat") & ok, "reasoning": np.char.startswith(origin, "reasoning") & ok}
    return {"Hs": Hs.to(dev), "Hb": Hb.to(dev), "tr": tr, "va": va, "es": es.to(dev), "eb": eb.to(dev),
            "groups": {g: torch.tensor(m, device=dev) for g, m in groups.items()},
            "s_norm": float(torch.load(ROOT / "data/xfer_claim4/t1/h_s.pt")["h"].float().norm(dim=-1).median())}


def curve(R: dict, lam: float, fracs: list[float], seed: int) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for f in fracs:
        idx = torch.tensor(rng.permutation(R["tr"])[: int(f * len(R["tr"]))], device=R["Hs"].device)
        WU, bU = ridge(R["Hs"][idx], R["Hb"][idx], lam)
        WE, bE = ridge(R["Hb"][idx], R["Hs"][idx], lam)

        def U(a: torch.Tensor) -> torch.Tensor:
            return q(q(a) * R["s_norm"] @ WU.T + bU)

        Eh = R["eb"] @ WE.T + bE
        row = {"n_train": int(len(idx))}
        for g, m in R["groups"].items():
            row[g] = {"M": fve_nrm(U(R["es"][m]), R["eb"][m]), "C": fve_nrm(U(Eh[m]), R["eb"][m])}
        out[f"{int(100 * f)}%"] = row
        print(f"{int(100 * f):3d}% n={len(idx):6d}  " + "  ".join(
            f"{g} M {v['M']:.3f} C {v['C']:.3f}" for g, v in row.items() if g != "n_train"), flush=True)
    return out


def train_map(R: dict, hidden: int, seed: int, epochs: int = 60) -> tuple[torch.nn.Module, dict]:
    """hidden 0 = affine. Inputs are the unit state (times sqrt(d)); loss on unit outputs."""
    torch.manual_seed(seed)
    d_s, d_b = R["Hs"].shape[1], R["Hb"].shape[1]
    lin = torch.nn.Linear(d_s, d_b)
    if hidden:
        mlp = torch.nn.Sequential(torch.nn.Linear(d_s, hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(hidden, hidden), torch.nn.GELU(), torch.nn.Dropout(0.1),
                                  torch.nn.Linear(hidden, d_b))

        class Res(torch.nn.Module):  # affine + MLP residual: can only add to the affine map
            def __init__(self) -> None:
                super().__init__()
                self.lin, self.mlp = lin, mlp

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return self.lin(x) + self.mlp(x)

        net: torch.nn.Module = Res()
    else:
        net = lin
    net = net.to(R["Hs"].device)
    scale = d_s ** 0.5
    X = q(R["Hs"]) * scale
    Y = q(R["Hb"])
    tr = torch.tensor(R["tr"], device=X.device)
    va = torch.tensor(R["va"], device=X.device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3 if hidden else 3e-3, weight_decay=1e-2)
    steps = epochs * (len(tr) // 512 + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=opt.param_groups[0]["lr"], total_steps=steps)
    best, best_state, hist = -9.0, None, []
    for ep in range(epochs):
        net.train()
        for b in tr[torch.randperm(len(tr), device=X.device)].split(512):
            loss = ((q(net(X[b])) - Y[b]) ** 2).sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        net.eval()
        with torch.no_grad():
            f = fve_nrm(net(X[va]), Y[va])
        hist.append(f)
        if f > best:
            best, best_state = f, {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    net.eval()
    return net, {"val_fve_best": best, "best_epoch": int(np.argmax(hist)), "val_curve": hist}


def cmd(args: argparse.Namespace) -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    R = load_rows(args.target, dev)
    print(f"{args.target}: train {len(R['tr'])}, val {len(R['va'])}, T1 eval "
          + ", ".join(f"{g} {int(m.sum())}" for g, m in R["groups"].items()), flush=True)
    rep: dict = {"target": args.target, "curve (ridge, lambda 0.1)": curve(R, 0.1, [0.1, 0.25, 0.5, 1.0], 0)}
    scale = R["Hs"].shape[1] ** 0.5
    for name, hidden in (("affine (same loss/optimizer)", 0), (f"affine+MLP hidden {args.hidden}", args.hidden)):
        net, info = train_map(R, hidden, 0, args.epochs)
        with torch.no_grad():
            row = {g: fve_nrm(net(q(R["es"][m]) * scale), R["eb"][m]) for g, m in R["groups"].items()}
        rep[name] = {"T1 eval FVE (map ceiling)": row, "val_fve_best": info["val_fve_best"],
                     "best_epoch": info["best_epoch"]}
        print(f"{name:32s} val {info['val_fve_best']:.3f} (ep {info['best_epoch']})  "
              + "  ".join(f"{g} {v:.3f}" for g, v in row.items()), flush=True)
    out = ROOT / f"data/xfer_claim4/ceiling_{args.target}.json"
    out.write_text(json.dumps(rep, indent=1))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", choices=list(T), required=True)
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=60)
    cmd(ap.parse_args())


if __name__ == "__main__":
    main()
