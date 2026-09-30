#!/usr/bin/env python3
"""Claim-4 follow-up: refit the reading maps on a web + chat + reasoning mix.

T1 found retention T/S of 0.39 (8B) / 0.53 (30B) on web+chat but 0.81 / 0.96 on the
reasoning slice. The Finding-15 maps were fitted on reasoning traces plus the FIRST 768
tokens of 3,000 web docs, with no chat at all. This refits E and U with the same ridge,
adding FRESH web and chat documents that match T1's evaluation recipe, and saves the maps
under data/xfer_claim4/maps_mix/ for T1/T2 re-runs (`--variant mix`).

Hygiene:
  * Documents never used anywhere: Ultra-FineWeb from index 300,000 and WildChat from
    200,000. T1 evaluates on 200,000+ and 100,000+ respectively (1,000 each), so the ranges
    are disjoint. The old fit used web docs 0-2,999.
  * Positions follow T1's datagen recipe: ctx <= 4096, pos >= 50, special tokens
    excluded, 6 per doc.
  * The old pair rows (web768 / r17 / target traces) are kept, so the mix ADDS domains.
    Split 80/10/10 by document hash.
  * lambda is chosen on the pooled val split. Nothing is fitted on T1/T2 rows.
  * Steering is unchanged: the adopted steering vectors stay on the old maps. This
    reports cos(NU_mix, NU_old) so the effect of a refit on steering is visible.

    uv run python scripts/xfer_refit_mix.py build
    CUDA_VISIBLE_DEVICES=4 uv run python scripts/xfer_refit_mix.py extract --target 8b
    uv run python scripts/xfer_refit_mix.py fit --target 8b
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "data/xfer_claim4/maps_mix"
T = {
    "8b": {"model": "Qwen/Qwen3-8B", "layer": 26, "pairs": "data/xfer8b/pairs.pt",
           "old_maps": "data/xfer8b/dirs_clean/L26_maps.pt", "old_nu": "data/xfer8b/dirs_clean/L26_NU.pt"},
    "30b": {"model": "Qwen/Qwen3-30B-A3B", "layer": 34, "pairs": "data/xfer30b/pairs_L34.pt",
            "old_maps": "data/xfer30b/dirs_L34/L34_maps.pt", "old_nu": "data/xfer30b/dirs_L34/L34_NU.pt"},
}
MAX_CTX, MIN_POS, PER_DOC = 4096, 50, 6


def split_of(doc_id: str) -> str:
    h = int(hashlib.sha256(doc_id.encode()).hexdigest()[:8], 16) % 10
    return "train" if h < 8 else ("val" if h == 8 else "test")


def cmd_build(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from reasoning_attention.config import CORPUS_CONFIG, CORPUS_ID, CORPUS_SPLIT, CORPUS_TEXT_COLUMN
    from reasoning_attention.datagen.extract import render_chat

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    special = set(tok.all_special_ids)
    rng = random.Random(20260929)
    seen = {hashlib.sha256(json.loads(line)["text"].encode()).hexdigest()
            for line in open(ROOT / "data/xfer8b/web3000.jsonl")}
    items: list[dict] = []

    def add(doc_id: str, origin: str, text: str) -> bool:
        ids = tok(text, add_special_tokens=True)["input_ids"][:MAX_CTX]
        cand = [i for i, t in enumerate(ids) if i >= MIN_POS and t not in special]
        if len(cand) < PER_DOC:
            return False
        items.append({"doc_id": doc_id, "origin": origin, "ids": ids,
                      "pos": sorted(rng.sample(cand, PER_DOC)), "split": split_of(doc_id)})
        return True

    n = 0
    web = load_dataset(CORPUS_ID, CORPUS_CONFIG, split=CORPUS_SPLIT, streaming=True).skip(args.web_start)
    for k, r in enumerate(web):
        text = r[CORPUS_TEXT_COLUMN]
        if hashlib.sha256(text.encode()).hexdigest() in seen:
            continue
        n += add(f"mixweb:{args.web_start + k}", "web4k", text)
        if n >= args.n_docs:
            break
    n = 0
    chat = load_dataset("allenai/WildChat-1M", split="train", streaming=True).skip(args.chat_start)
    for k, r in enumerate(chat):
        text = render_chat(r["conversation"], tok)
        if text:
            n += add(f"mixchat:{args.chat_start + k}", "chat", text)
        if n >= args.n_docs:
            break
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"items": items, "web_start": args.web_start, "chat_start": args.chat_start}, OUT / "items.pt")
    from collections import Counter
    print(f"wrote {len(items)} docs {Counter(i['origin'] for i in items)}, "
          f"{sum(len(i['pos']) for i in items)} positions")


def cmd_extract(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM

    cfg = T[args.target]
    items = torch.load(OUT / "items.pt", weights_only=False)["items"]
    for tag, model_id, layer in (("s", "Qwen/Qwen3-1.7B", 20), (args.target, cfg["model"], cfg["layer"])):
        path = OUT / f"h_{tag}.pt"
        if path.exists():
            continue
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        grab: dict[str, torch.Tensor] = {}
        model.model.layers[layer].register_forward_hook(
            lambda _m, _i, out: grab.__setitem__("h", out[0] if isinstance(out, tuple) else out))
        with torch.inference_mode():
            x = torch.tensor([items[0]["ids"][:256]], device="cuda")
            ref = model(input_ids=x, output_hidden_states=True).hidden_states[layer + 1]
            assert torch.equal(grab["h"], ref), "hook != hidden_states[l+1]"
        model.model.layers = model.model.layers[: layer + 1]
        H = []
        with torch.inference_mode():
            for k, it in enumerate(items):
                model.model(input_ids=torch.tensor([it["ids"]], device="cuda"), use_cache=False)
                H.append(grab["h"][0, torch.tensor(it["pos"], device="cuda", dtype=torch.long)].cpu())
                if k % 500 == 0:
                    print(f"{tag} {k}/{len(items)}", flush=True)
        torch.save(torch.cat(H), path)
        print(f"wrote {path}", flush=True)
        del model
        torch.cuda.empty_cache()


def cmd_fit(args: argparse.Namespace) -> None:
    from xfer_fit_maps import auc, ridge

    cfg = T[args.target]
    L = cfg["layer"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    items = torch.load(OUT / "items.pt", weights_only=False)["items"]
    hs_new = torch.load(OUT / "h_s.pt").float()
    hb_new = torch.load(OUT / f"h_{args.target}.pt").float()
    org_new = np.array([it["origin"] for it in items for _ in it["pos"]])
    spl_new = np.array([it["split"] for it in items for _ in it["pos"]])
    P = torch.load(ROOT / cfg["pairs"], weights_only=False)
    meta = P["meta"]
    hs_old, hb_old = P["h_s"].float(), P["h_b"][L].float()
    org_old = np.array([m["origin"] for m in meta])
    spl_old = np.array([m["split"] for m in meta])
    Hs = torch.cat([hs_old, hs_new]).to(dev)
    Hb = torch.cat([hb_old, hb_new]).to(dev)
    origin = np.concatenate([org_old, org_new])
    split = np.concatenate([spl_old, spl_new])
    keep = np.ones(len(origin), bool)
    for H in (Hs, Hb):
        nrm = H.norm(dim=-1)
        keep &= (nrm < 5 * nrm.median()).cpu().numpy()
    print(f"rows: old {len(org_old)}, new {len(org_new)}; dropped {(~keep).sum()} outliers; "
          f"origins {dict(zip(*np.unique(origin, return_counts=True)))}")
    tr = torch.tensor((split == "train") & keep, device=dev)
    va = torch.tensor((split == "val") & keep, device=dev)

    def fve(pred: torch.Tensor, Y: torch.Tensor, mu: torch.Tensor) -> float:
        return float(1 - ((pred - Y) ** 2).sum() / ((Y - mu) ** 2).sum())

    old = torch.load(ROOT / cfg["old_maps"])
    out, rep = {}, {"target": args.target, "layer": L}
    for name, X, Y in (("U", Hs, Hb), ("E", Hb, Hs)):
        best = None
        for lam in args.lams:
            W, b = ridge(X[tr], Y[tr], lam)
            f = fve(X[va] @ W.T + b, Y[va], Y[tr].mean(0))
            best = max(best or (-9.0, 0.0, None, None), (f, lam, W, b), key=lambda t: t[0])
        f, lam, W, b = best
        out[f"W_{name}"], out[f"b_{name}"], out[f"lam_{name}"] = W.cpu(), b.cpu(), lam
        Wo, bo = old[f"W_{name}"].float().to(dev), old[f"b_{name}"].float().to(dev)
        per = {}
        for o in np.unique(origin):
            m = torch.tensor((split == "val") & keep & (origin == o), device=dev)
            if int(m.sum()) < 20:
                continue
            mu = Y[tr].mean(0)
            per[o] = {"old": fve(X[m] @ Wo.T + bo, Y[m], mu), "mix": fve(X[m] @ W.T + b, Y[m], mu)}
        rep[name] = {"lam": lam, "val_fve_pooled": f, "val_fve_by_origin": per}
        print(f"{name}: lam {lam}  pooled val FVE {f:.3f}  by origin "
              + ", ".join(f"{o} {v['old']:.3f}->{v['mix']:.3f}" for o, v in per.items()), flush=True)
    N = torch.load(ROOT / "data/pool/dir_A_1trace.pt")["unit"].float().to(dev)
    nu_mix = out["W_U"].to(dev) @ N
    nu_mix = nu_mix / nu_mix.norm()
    nu_old = torch.load(ROOT / cfg["old_nu"])["unit"].float().to(dev)
    rep["cos(NU_mix, NU_old)"] = float(nu_mix @ nu_old)
    kind = np.array([m["kind"] for m in meta])
    doubt = np.array([m["doubt"] for m in meta])
    te = (kind == "boundary") & (spl_old == "test") & (doubt >= 0) & keep[: len(meta)]
    y = (doubt[te] == 0).astype(int)
    hb_te = hb_old[torch.tensor(te)].to(dev)
    rep["AUC test boundaries: NU_old / NU_mix"] = (auc((hb_te @ nu_old).cpu().numpy(), y),
                                                   auc((hb_te @ nu_mix).cpu().numpy(), y))
    print({k: rep[k] for k in ("cos(NU_mix, NU_old)", "AUC test boundaries: NU_old / NU_mix")})
    torch.save(out, OUT / f"{args.target}_maps.pt")
    (OUT / f"{args.target}_fit_report.json").write_text(json.dumps(rep, indent=1))
    print(f"wrote {OUT / f'{args.target}_maps.pt'}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--web-start", type=int, default=300_000)
    b.add_argument("--chat-start", type=int, default=200_000)
    b.add_argument("--n-docs", type=int, default=2000)
    for c in ("extract", "fit"):
        s = sub.add_parser(c)
        s.add_argument("--target", choices=list(T), required=True)
        s.add_argument("--lams", type=float, nargs="+", default=[1e-3, 1e-2, 1e-1, 1.0])
    args = ap.parse_args()
    {"build": cmd_build, "extract": cmd_extract, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    main()
