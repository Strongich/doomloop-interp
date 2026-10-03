#!/usr/bin/env python3
r"""EXPERIMENT-selective-doubt.md §S5: N_red, N_all, N_prod, N_sel with N's recipe.

  delta(h) = AR(edit(AV(h), CONTINUE_PARAGRAPH)) - AR(AV(h))   (raw, as derive_pool.py; AV
             greedy, 300 new tokens, explanation stripped; batched FrozenNLA, parity-checked
             against the single-vector path)
  direction = normalize(mean over questions of the per-question mean delta)

  N_red   redundant boundaries: candidates shuffled (seed 20261002), accepted in order while
          the question has < 3, until 300 usable deltas (an explanation that cannot be edited
          is skipped and the next candidate taken)
  N_all   the same rule over every labeled doubt boundary (all labels except `excluded`)
  N_prod  all productive boundaries, up to 300 (random 300 if more), no per-question cap
  N_sel   normalize(N_all - (N_all . w_hat) w_hat), w_hat from sel_gate.py

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/sel_directions.py
"""

from __future__ import annotations

import collections
import json
import random
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
SEED = 20261002
SEL = ROOT / "data/selective"


def capped_order(eps: list[dict], tag: str, cap: int | None) -> list[dict]:
    xs = sorted(eps, key=lambda e: (e["key"], e["k"]))
    random.Random(f"{SEED}:{tag}").shuffle(xs)
    if cap is None:
        return xs
    per: collections.Counter = collections.Counter()
    out = []
    for e in xs:
        if per[e["question_id"]] < cap:
            per[e["question_id"]] += 1
            out.append(e)
    return out


def main() -> None:
    from steer_demo import CONTINUE_PARAGRAPH, edit_explanation
    from xfer_nla_tools import FrozenNLA

    eps = [json.loads(x) for x in open(SEL / "episodes_source.jsonl")]
    st = [torch.load(p, weights_only=False) for p in sorted((SEL / "states").glob("source_s*.pt"))]
    H = torch.cat([s["h"] for s in st]).float()
    idx = {k: i for i, k in enumerate(tuple(x) for s in st for x in s["index"])}
    nla = FrozenNLA()
    specs = {
        "N_red": capped_order([e for e in eps if e["label"] == "redundant"], "red", 3),
        "N_all": capped_order([e for e in eps if e["label"] != "excluded"], "all", 3),
        "N_prod": capped_order([e for e in eps if e["label"] == "productive"], "prod", None),
    }
    dirs, info = {}, {}
    for name, cand in specs.items():
        got: list[tuple[dict, torch.Tensor, str]] = []
        pos = 0
        while len(got) < 300 and pos < len(cand):
            batch = cand[pos : pos + 64]
            pos += len(batch)
            hs = H[[idx[(e["key"], e["k"])] for e in batch]]
            zs = [z.strip() for z in nla.verbalize(hs, max_new_tokens=300)]
            ok = []
            for e, z in zip(batch, zs):
                try:
                    ok.append((e, z, edit_explanation(z, CONTINUE_PARAGRAPH)))
                except SystemExit:
                    continue
            if not ok:
                continue
            a = nla.reconstruct([z for _, z, _ in ok])
            b = nla.reconstruct([ed for _, _, ed in ok])
            for (e, z, _), d in zip(ok, (b - a).float().cpu()):
                if len(got) < 300:
                    got.append((e, d, z))
            print(f"{name}: {len(got)} deltas from {pos} candidates", flush=True)
        byq: dict = collections.defaultdict(list)
        for e, d, _ in got:
            byq[e["question_id"]].append(d)
        mean = torch.stack([torch.stack(v).mean(0) for v in byq.values()]).mean(0)
        unit = mean / mean.norm()
        D = torch.stack([d for _, d, _ in got])
        info[name] = {"n": len(got), "questions": len(byq), "candidates_used": pos,
                      "candidates_available": len(cand),
                      "labels": dict(collections.Counter(e["label"] for e, _, _ in got)),
                      "mean_delta_norm": float(D.norm(dim=1).mean()),
                      "cos_to_mean": float(torch.nn.functional.cosine_similarity(D, mean[None]).mean())}
        dirs[name] = unit
        torch.save({"unit": unit, "stats": info[name], "recipe": "CONTINUE_PARAGRAPH raw deltas, "
                    "equal weight per question", "sources": [(e["key"], e["k"]) for e, _, _ in got],
                    "explanations": [z for _, _, z in got], "deltas": D}, SEL / f"dir_{name}.pt")
    w = torch.tensor(__import__("numpy").load(SEL / "w_hat.npy")).float()
    na = dirs["N_all"]
    sel = na - (na @ w) * w
    dirs["N_sel"] = sel / sel.norm()
    torch.save({"unit": dirs["N_sel"], "stats": {"from": "N_all minus w_hat component",
                                                 "cos_N_all_w_hat": float(na @ w)}}, SEL / "dir_N_sel.pt")
    dirs["N"] = torch.load(ROOT / "data/pool/dir_A_1trace.pt", weights_only=False)["unit"].float()
    dirs["w_hat"] = w
    names = ["N", "N_all", "N_red", "N_prod", "N_sel", "w_hat"]
    cos = {a: {b: round(float(dirs[a] @ dirs[b]), 4) for b in names} for a in names}
    rep = {"directions": info, "cosines": cos,
           "note_red_vs_all": "cos(N_red, N_all) > 0.98: selectivity can barely act"
           if cos["N_red"]["N_all"] > 0.98 else ""}
    (SEL / "directions.json").write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
