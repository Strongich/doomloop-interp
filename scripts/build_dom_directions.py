#!/usr/bin/env python3
"""Is the NLA necessary to find the suppression direction?

Every direction in this study was built the expensive way: verbalize the
activation with the AV, edit the English, reconstruct with the AR, subtract. This
builds the two directions a practitioner would reach for FIRST, from the same
probes, and reports how similar they are.

  D  difference of means -- mean(h | plain boundary) - mean(h | doubt boundary).
     The standard contrastive-activation-addition direction.
  P  linear probe -- logistic regression separating the two classes at the same
     positions; the weight vector, pointing plain-ward.

Both point the same way as the NLA direction by construction: away from doubt.

The probe set is already paired: all 297 traces in block_explanations_3way.csv
carry BOTH a `doubt_wait` boundary and a `plain` one, so D is a within-trace
contrast and the depth confound is small by construction (median token index 335
against 313; the doubt probe is the later one in 172 of 297, near chance).

The cosines printed at the end are half the experiment. Finding 5d set the scale:
paraphrases of the same meaning sit at +0.754, different constructions of the
same vector at 0.93-0.999. So cos(D, N) > 0.9 means the NLA found nothing a mean
difference would not have.

    uv run python scripts/build_dom_directions.py
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_attention.config import MODEL_ID, NLAConfig  # noqa: E402
from reasoning_attention.nla.arch import inner_transformer  # noqa: E402
from reasoning_attention.tokenview import _chat_header  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--probes", type=Path, default=Path("data/block_explanations_3way.csv"))
    p.add_argument("--traces", type=Path, default=Path("data/traces_all.jsonl"))
    p.add_argument("--out-dir", type=Path, default=Path("data/dom"))
    p.add_argument("--max-prefix", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NLAConfig()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    probes = collections.defaultdict(dict)
    for r in csv.DictReader(args.probes.open()):
        if r["kind"] in ("doubt_wait", "plain"):
            probes[(r["question_id"], r["rollout_index"])][r["kind"]] = r
    keys = [k for k, v in probes.items() if len(v) == 2]
    print(f"{len(keys)} traces with both a doubt and a plain probe")

    need = set(keys)
    traces: dict[tuple[str, str], dict[str, Any]] = {}
    for line in args.traces.open():
        r = json.loads(line)
        k = (r["question_id"], str(r["rollout_index"]))
        if k in need:
            traces[k] = r
    print(f"{len(traces)} trace texts resolved")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    trunk = inner_transformer(model)
    cap: dict[str, Any] = {"h": None, "site": -1}

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.shape[1] > cap["site"] >= 0:
            cap["h"] = hidden[0, cap["site"]].detach().float().cpu()
        return output

    handle = trunk.layers[cfg.extraction_layer].register_forward_hook(hook)

    H: dict[str, list[torch.Tensor]] = {"doubt_wait": [], "plain": []}
    depth: dict[str, list[int]] = {"doubt_wait": [], "plain": []}
    kept = 0
    try:
        for i, k in enumerate(keys, 1):
            t = traces.get(k)
            if t is None:
                continue
            header = _chat_header(tok, t["question"])
            h_ids = tok(header, return_tensors="pt")["input_ids"][0]
            r_ids = tok(t["response"], add_special_tokens=False, return_tensors="pt")["input_ids"][0]
            pair = {}
            for kind in ("doubt_wait", "plain"):
                j = int(probes[k][kind]["token"])
                if j >= len(r_ids) or len(h_ids) + j + 1 > args.max_prefix:
                    break
                ids = torch.cat([h_ids, r_ids[: j + 1]]).to(model.device)
                cap["site"] = len(ids) - 1
                with torch.no_grad():
                    trunk(input_ids=ids[None])
                if cap["h"] is None:
                    break
                pair[kind] = (cap["h"].clone(), j)
            if len(pair) == 2:           # keep only complete pairs: D stays within-trace
                for kind, (h, j) in pair.items():
                    H[kind].append(h)
                    depth[kind].append(j)
                kept += 1
            if i % 50 == 0:
                print(f"  [{i}/{len(keys)}] kept {kept}", flush=True)
    finally:
        handle.remove()

    dw = torch.stack(H["doubt_wait"])
    pl = torch.stack(H["plain"])
    print(f"\n{kept} complete pairs")
    print(f"depth (token index): doubt median {sorted(depth['doubt_wait'])[kept//2]}, "
          f"plain median {sorted(depth['plain'])[kept//2]}")
    print(f"||h||: doubt {dw.norm(dim=1).mean():.1f}, plain {pl.norm(dim=1).mean():.1f}")

    # D — difference of means, pointing AWAY from doubt (plain minus doubt)
    d_raw = pl.mean(0) - dw.mean(0)
    D = d_raw / d_raw.norm()

    # P — logistic regression weights, same orientation
    X = torch.cat([dw, pl]).double()
    y = torch.cat([torch.zeros(kept), torch.ones(kept)]).double()
    Xn = (X - X.mean(0)) / X.std(0).clamp_min(1e-6)
    w = torch.zeros(Xn.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        z = Xn @ w + b
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y) + 1e-3 * w.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)  # type: ignore[arg-type]
    with torch.no_grad():
        acc = (((Xn @ w + b) > 0).double() == y).double().mean()
    P = (w.detach() / X.std(0).clamp_min(1e-6)).float()
    P = P / P.norm()
    print(f"linear probe train accuracy {100*acc:.1f}%  (chance 50%)")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, u, stats in (
        ("dir_D_diffmeans.pt", D, {"n": float(kept), "method": "diff-of-means"}),
        ("dir_P_probe.pt", P, {"n": float(kept), "method": "logistic-probe",
                               "train_acc": float(acc)}),
    ):
        torch.save({"unit": u, "stats": stats, "edit": "continue"}, args.out_dir / name)
        print(f"  wrote {name}")

    others = {
        "N_global297": Path("data/delta_suppress_mean.pt"),
        "N_A_1trace": Path("data/pool/dir_A_1trace.pt"),
    }
    units = {"D_diffmeans": D, "P_probe": P}
    for n, path in others.items():
        if path.exists():
            units[n] = torch.load(path, weights_only=False)["unit"].float()
    names = list(units)
    print("\npairwise cosine (NLA-derived vs cheap baselines):")
    print("              " + "".join(f"{n[:11]:>13}" for n in names))
    for a in names:
        row = "".join(
            f"{torch.nn.functional.cosine_similarity(units[a][None], units[b][None]).item():>13.3f}"
            for b in names
        )
        print(f"{a[:13]:<14}" + row)
    print("\nScale from Finding 5d: paraphrases of the same meaning sit at +0.754;")
    print("different constructions of the same vector at 0.93-0.999.")
    print("cos(D, N) > 0.9 -> the NLA found nothing a mean difference would not have.")


if __name__ == "__main__":
    main()
