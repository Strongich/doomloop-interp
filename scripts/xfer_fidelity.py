#!/usr/bin/env python3
"""Claim-4 T1: fidelity retention of the 1.7B NLA adapted to 8B / 30B-A3B (EXPERIMENT-transfer-claim4.md).

Fresh positions, never seen by any NLA stage or map fit:
  web        Ultra-FineWeb (streamed) docs from index 200,000, text-hash deduped
             against data/xfer8b/web3000.jsonl
  chat       WildChat-1M conversations from index 100,000, rendered with the chat
             template (add_generation_prompt=False), as build_rl_data.sh --corpus-kind chat
  reasoning  test-split trace positions from the map-fit pairs files (r17 / 8B / 30B
             traces), half boundaries, half random; reported separately
Positions follow the datagen recipe: add_special_tokens=True, left context <= 4096,
position >= 50, special tokens excluded, 5 per document.

Phases (the same token sequences through every model; HF for all three models, so no
capture rows are mixed in):
  build                          -> t1/items.pt
  extract --model M --layer L    -> t1/h_<tag>.pt      (gate: hook == hidden_states[l+1])
  verbalize --which s|8b|30b     -> t1/z_<which>.json + t1/r_<which>.pt   (3 samples, T=1)
  report                         -> t1/report.json

Metric: the reference fve_nrm. Both vectors are normalized, and the mean is taken over
the normalized golds of the space being scored, on the T1 eval rows themselves.
Maps are frozen: 8B dirs_clean/L26_maps.pt, 30B dirs_L34/L34_maps.pt. Nothing is fitted
on T1 rows.
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

OUT = ROOT / "data/xfer_claim4/t1"
TARGETS = {
    "8b": {"model": "Qwen/Qwen3-8B", "layer": 26,
           "maps": "data/xfer8b/dirs_clean/L26_maps.pt", "pairs": "data/xfer8b/pairs.pt"},
    "30b": {"model": "Qwen/Qwen3-30B-A3B", "layer": 34,
            "maps": "data/xfer30b/dirs_L34/L34_maps.pt", "pairs": "data/xfer30b/pairs_L34.pt"},
}
MAX_CTX, MIN_POS, PER_DOC = 4096, 50, 5
SUFFIX = ""  # "--variant mix": the web+chat+reasoning refit maps (xfer_refit_mix.py)


def set_variant(v: str) -> None:
    global SUFFIX
    if v == "mix":
        SUFFIX = "_mix"
        for tag in TARGETS:
            TARGETS[tag]["maps"] = f"data/xfer_claim4/maps_mix/{tag}_maps.pt"


def q(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


def fve_nrm(pred: torch.Tensor, gold: torch.Tensor) -> float:
    g, p = q(gold.float()), q(pred.float())
    mu = g.mean(0)
    return float(1 - ((p - g) ** 2).sum(-1).mean() / ((g - mu) ** 2).sum(-1).mean())


def cosm(pred: torch.Tensor, gold: torch.Tensor) -> float:
    return float(torch.nn.functional.cosine_similarity(pred.float(), gold.float(), dim=-1).mean())


# ---------------------------------------------------------------------------------- build
def positions(ids: list[int], special: set[int], rng: random.Random) -> list[int]:
    cand = [i for i, t in enumerate(ids) if i >= MIN_POS and t not in special]
    return sorted(rng.sample(cand, min(PER_DOC, len(cand))))


def cmd_build(args: argparse.Namespace) -> None:
    from datasets import load_dataset
    from transformers import AutoTokenizer

    from reasoning_attention.config import CORPUS_CONFIG, CORPUS_ID, CORPUS_SPLIT, CORPUS_TEXT_COLUMN
    from reasoning_attention.datagen.extract import render_chat

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    special = set(tok.all_special_ids)
    rng = random.Random(20260928)
    seen = {hashlib.sha256(json.loads(line)["text"].encode()).hexdigest()
            for line in open(ROOT / "data/xfer8b/web3000.jsonl")}
    items: list[dict] = []

    def add(doc_id: str, origin: str, text: str, special_tokens: bool = True) -> bool:
        ids = tok(text, add_special_tokens=special_tokens)["input_ids"][:MAX_CTX]
        pos = positions(ids, special, rng)
        if len(pos) < PER_DOC:
            return False
        items.append({"doc_id": doc_id, "origin": origin, "ids": ids, "pos": pos})
        return True

    n_web = n_dup = 0
    web = load_dataset(CORPUS_ID, CORPUS_CONFIG, split=CORPUS_SPLIT, streaming=True).skip(args.web_start)
    for k, r in enumerate(web):
        text = r[CORPUS_TEXT_COLUMN]
        if hashlib.sha256(text.encode()).hexdigest() in seen:
            n_dup += 1
            continue
        n_web += add(f"t1web:{args.web_start + k}", "web", text)
        if n_web >= args.n_docs:
            break
    print(f"web: {n_web} docs ({n_dup} dropped as web3000 duplicates)", flush=True)

    n_chat = 0
    chat = load_dataset("allenai/WildChat-1M", split="train", streaming=True).skip(args.chat_start)
    for k, r in enumerate(chat):
        text = render_chat(r["conversation"], tok)
        if text:
            n_chat += add(f"t1chat:{args.chat_start + k}", "chat", text)
        if n_chat >= args.n_docs:
            break
    print(f"chat: {n_chat} conversations", flush=True)

    # reasoning: test-split trace rows of the map-fit pairs (never fitted on), rebuilt
    from xfer_extract_pairs_vllm import load_qtext, trace_ids

    qtext = load_qtext()
    traces = {}
    for origin, path in (("r17", "data/xfer8b/r17_fit_base/rollouts.jsonl"),
                         ("r8b", "data/xfer8b/r8b_fit_base/rollouts.jsonl"),
                         ("r30", "data/xfer30b/r30_fit_base/rollouts.jsonl")):
        for line in open(ROOT / path):
            r = json.loads(line)
            if r.get("policy", "base") == "base" and r.get("seed", 0) == 0:
                traces[(origin, r["question_id"])] = r
    from xfer_extract_pairs import split_of

    is_b = [tok.convert_ids_to_tokens(i).count("Ċ") >= 2 for i in range(len(tok))]
    close = tok.convert_tokens_to_ids("</think>")
    per_origin = args.n_reasoning // 3
    for origin in ("r17", "r8b", "r30"):
        keys = sorted(k for k in traces if k[0] == origin and split_of(k[1]) == "test")
        rng.shuffle(keys)
        got = 0
        for key in keys:
            ids, P = trace_ids(tok, traces[key], qtext, MAX_CTX)
            gen = ids[P:]
            end = P + (gen.index(close) if close in gen else len(gen))
            b = [p for p in range(max(P, MIN_POS), end) if is_b[ids[p]]]
            o = [p for p in range(max(P, MIN_POS), end) if not is_b[ids[p]]]
            if len(b) < 2 or len(o) < 2:
                continue
            pos = sorted(rng.sample(b, 2) + rng.sample(o, 2))
            items.append({"doc_id": f"t1{origin}:{key[1]}", "origin": f"reasoning_{origin}",
                          "ids": ids, "pos": pos, "kind": ["b" if is_b[ids[p]] else "r" for p in pos]})
            got += 4
            if got >= per_origin:
                break
        print(f"reasoning {origin}: {got} positions", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"items": items, "web_start": args.web_start, "chat_start": args.chat_start}, OUT / "items.pt")
    print(f"wrote {OUT / 'items.pt'}: {len(items)} sequences, {sum(len(i['pos']) for i in items)} positions")


# -------------------------------------------------------------------------------- extract
def cmd_extract(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM

    items = torch.load(OUT / "items.pt", weights_only=False)["items"]
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    grab: dict[str, torch.Tensor] = {}
    model.model.layers[args.layer].register_forward_hook(
        lambda _m, _i, out: grab.__setitem__("h", out[0] if isinstance(out, tuple) else out))
    with torch.inference_mode():
        x = torch.tensor([items[0]["ids"][:256]], device="cuda")
        ref = model(input_ids=x, output_hidden_states=True).hidden_states[args.layer + 1]
        assert torch.equal(grab["h"], ref), "hook != hidden_states[l+1]"
    print("gate: hook == hidden_states[l+1]", flush=True)
    model.model.layers = model.model.layers[: args.layer + 1]
    H = []
    with torch.inference_mode():
        for k, it in enumerate(items):
            model.model(input_ids=torch.tensor([it["ids"]], device="cuda"), use_cache=False)
            H.append(grab["h"][0, torch.tensor(it["pos"], device="cuda", dtype=torch.long)].cpu())
            if k % 500 == 0:
                print(f"{k}/{len(items)}", flush=True)
    torch.save({"h": torch.cat(H), "model": args.model, "layer": args.layer}, OUT / f"h_{args.tag}.pt")
    print(f"wrote h_{args.tag}.pt", flush=True)


# ------------------------------------------------------------------------------ verbalize
def source_vectors(which: str) -> torch.Tensor:
    if which == "s":
        return torch.load(OUT / "h_s.pt")["h"].float()
    t = TARGETS[which]
    m = torch.load(ROOT / t["maps"])
    hb = torch.load(OUT / f"h_{which}.pt")["h"].float()
    return hb @ m["W_E"].float().T + m["b_E"].float()


def cmd_verbalize(args: argparse.Namespace) -> None:
    from xfer_nla_tools import FrozenNLA

    f = FrozenNLA()
    vecs = source_vectors(args.which)
    Z, R = [], []
    for s in range(args.samples):
        z = f.verbalize(vecs, batch=args.batch, temperature=1.0, seed=1000 + s)
        Z.append(z)
        R.append(f.reconstruct(z, batch=args.batch).cpu())
        print(f"sample {s}: {sum(1 for t in z if not t)} empty of {len(z)}", flush=True)
    sfx = SUFFIX if args.which != "s" else ""  # the source path does not use the maps
    (OUT / f"z_{args.which}{sfx}.json").write_text(json.dumps(Z))
    torch.save(torch.stack(R, 1), OUT / f"r_{args.which}{sfx}.pt")  # [N, samples, 2048]
    print(f"wrote z_{args.which}{sfx}.json, r_{args.which}{sfx}.pt", flush=True)


# --------------------------------------------------------------------------------- report
def cmd_report(args: argparse.Namespace) -> None:
    items = torch.load(OUT / "items.pt", weights_only=False)["items"]
    origin = np.array([it["origin"] for it in items for _ in it["pos"]])
    hs = torch.load(OUT / "h_s.pt")["h"].float()
    rs = torch.load(OUT / "r_s.pt").float()
    s_norm = float(hs.norm(dim=-1).median())
    groups = {"web+chat": np.isin(origin, ["web", "chat"]), "web": origin == "web",
              "chat": origin == "chat", "reasoning": np.char.startswith(origin, "reasoning")}
    rep: dict = {"s_norm": s_norm, "n": {g: int(m.sum()) for g, m in groups.items()}}

    def per_sample(fn, R: torch.Tensor, gold: torch.Tensor) -> float:
        return float(np.mean([fn(R[:, j], gold) for j in range(R.shape[1])]))

    # S: source anchor, 1.7B space
    rep["S"] = {g: {"fve": per_sample(fve_nrm, rs[m], hs[m]), "cos": per_sample(cosm, rs[m], hs[m])}
                for g, m in groups.items()}
    for tag, t in TARGETS.items():
        if not (OUT / f"h_{tag}.pt").exists() or not (OUT / f"r_{tag}{SUFFIX}.pt").exists():
            continue
        maps = torch.load(ROOT / t["maps"])
        WU, bU = maps["W_U"].float(), maps["b_U"].float()
        WE, bE = maps["W_E"].float(), maps["b_E"].float()
        hb = torch.load(OUT / f"h_{tag}.pt")["h"].float()
        rb = torch.load(OUT / f"r_{tag}{SUFFIX}.pt").float()
        nb = hb.norm(dim=-1)
        keep = (nb < 5 * nb.median()).numpy()

        def U(a: torch.Tensor) -> torch.Tensor:  # the interface convention (xfer_interface.U_state)
            return q(q(a) * s_norm @ WU.T + bU)

        Eh = hb @ WE.T + bE
        gen = torch.Generator().manual_seed(0)
        out = {"dropped_outliers": int((~keep).sum())}
        for g, m0 in groups.items():
            m = torch.tensor(m0 & keep)
            if not m.any():
                continue
            gold_b, gold_s = hb[m], hs[m]
            rbm, rsm = rb[m], rs[m]
            perm = torch.randperm(int(m.sum()), generator=gen)
            T = per_sample(lambda a, y: fve_nrm(U(a), y), rbm, gold_b)
            row = {
                "T  U(AR(AV(E h_b))) vs h_b": T,
                "T_sh AR(AV(E h_b)) vs h_s": per_sample(fve_nrm, rbm, gold_s),
                "S->U U(AR(AV(h_s))) vs h_b": per_sample(lambda a, y: fve_nrm(U(a), y), rsm, gold_b),
                "M  U(h_s) vs h_b": fve_nrm(U(gold_s), gold_b),
                "C  U(E h_b) vs h_b": fve_nrm(U(Eh[m]), gold_b),
                "shuf U(AR(AV(E h_b)))[perm] vs h_b": per_sample(lambda a, y: fve_nrm(U(a[perm]), y), rbm, gold_b),
                "mean-state": fve_nrm(q(gold_b).mean(0).expand_as(gold_b), gold_b),
                "cos T": per_sample(lambda a, y: cosm(U(a), y), rbm, gold_b),
                "S (same rows)": per_sample(fve_nrm, rsm, gold_s),
            }
            row["R_abs = T / S"] = T / row["S (same rows)"]
            out[g] = row
        rep[tag + SUFFIX] = out
    rep["maps"] = {t: TARGETS[t]["maps"] for t in TARGETS}
    (OUT / f"report{SUFFIX}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--web-start", type=int, default=200_000)
    b.add_argument("--chat-start", type=int, default=100_000)
    b.add_argument("--n-docs", type=int, default=1000)
    b.add_argument("--n-reasoning", type=int, default=1000)
    e = sub.add_parser("extract")
    e.add_argument("--model", required=True)
    e.add_argument("--layer", type=int, required=True)
    e.add_argument("--tag", required=True)
    v = sub.add_parser("verbalize")
    v.add_argument("--which", choices=["s", "8b", "30b"], required=True)
    v.add_argument("--samples", type=int, default=3)
    v.add_argument("--batch", type=int, default=128)
    sub.add_parser("report")
    for sp in sub.choices.values():
        sp.add_argument("--variant", choices=["orig", "mix"], default="orig")
    args = ap.parse_args()
    set_variant(args.variant)
    {"build": cmd_build, "extract": cmd_extract, "verbalize": cmd_verbalize, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
