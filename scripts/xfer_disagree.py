#!/usr/bin/env python3
"""Claim-4 T2: does the adapted NLA read the TARGET's own state, beyond the shared text?

Sites: the held-out paragraph breaks of the single-site screens, where each model's own
P(next token is a doubt opener) is already measured on the SAME prefix. The
disagreement is dp = p_target - p_1.7B. A reader that only sees the text predicts the
same for both models, so it cannot track dp. Only a reader of model-specific state can.

Phases (outputs under data/xfer_claim4/t2/<target>/):
  sites --target 8b|30b    re-derive the screen's sites (same rng / traces / order),
                           check them against the screen files row by row, and attach
                           the TRUE trace origin. The 8B screen files label every site
                           "r8b": its origin rule matched "xfer8b" in the r17 path.
  extract --target T       HF block outputs: 1.7B L20 and the target at its rule layer
  verbalize --target T     z_s = AV(h_s), z_b = AV(E h_b): 3 samples at T=1, plus AR(z)
  judge --target T         blind judge (gpt-5.6-luna, low effort) on sample 0 of z_s and
                           z_b, plus the text-only baseline: the prefix's last 2 paragraphs
  report --target T        the primary statistic, the secondary AUCs, the baselines and
                           the positive control

Primary: Spearman(score(z_b) - score(z_s), dp), with a question-clustered bootstrap CI,
overall and within each trace origin (a site's text is on-policy for exactly one model,
so dp partly encodes whose text it is; the stratified correlation removes that).
Scores: (i) the lexical doubt cue in the last paragraph (Finding 15), averaged over
samples; (ii) <q(AR(z)), N>, averaged over samples; (iii) the judge's 0-100.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

T = {
    "8b": {"model": "Qwen/Qwen3-8B", "layer": 26, "maps": "data/xfer8b/dirs_clean/L26_maps.pt",
           "screen17": "data/xfer8b/local/17b.jsonl", "screenT": "data/xfer8b/local/8b.jsonl",
           "traces": ["data/xfer8b/r8b_fit_base/rollouts.jsonl", "data/xfer8b/r17_fit_base/rollouts.jsonl"],
           "origin": {"r8b_fit": "r8b", "r17_fit": "r17"}},
    "30b": {"model": "Qwen/Qwen3-30B-A3B", "layer": 34, "maps": "data/xfer30b/dirs_L34/L34_maps.pt",
            "screen17": "data/xfer30b/local/17b.jsonl", "screenT": "data/xfer30b/local/30b_L34.jsonl",
            "traces": ["data/xfer30b/r30_fit_base/rollouts.jsonl", "data/xfer8b/r17_fit_base/rollouts.jsonl"],
            "origin": {"r30_fit": "r30", "r17_fit": "r17"}},
}
N_PATH = "data/pool/dir_A_1trace.pt"


VARIANT = "orig"


def out_dir(tag: str) -> Path:
    """t2/<tag>; the "mix" variant writes t2/<tag>_mix and links the map-independent
    inputs (sites, h_s, h_b, z_s/r_s, the z_s and text judgments) from t2/<tag>."""
    base = ROOT / "data/xfer_claim4/t2" / tag
    base.mkdir(parents=True, exist_ok=True)
    if VARIANT == "orig":
        return base
    d = ROOT / "data/xfer_claim4/t2" / f"{tag}_mix"
    d.mkdir(parents=True, exist_ok=True)
    for f in ("sites.pt", "h_s.pt", "h_b.pt", "z_s.json", "r_s.pt"):
        if (base / f).exists() and not (d / f).exists():
            (d / f).symlink_to(base / f)
    if (base / "judge.jsonl").exists() and not (d / "judge.jsonl").exists():
        with open(d / "judge.jsonl", "w") as out:  # keep only map-independent judgments
            for line in open(base / "judge.jsonl"):
                if not json.loads(line)["id"].startswith("b:"):
                    out.write(line)
    return d


def q(x: torch.Tensor) -> torch.Tensor:
    return x / x.norm(dim=-1, keepdim=True)


# ---------------------------------------------------------------------------------- sites
def cmd_sites(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer
    from xfer_local_injection import FORMAT, first_sentence_doubt, split_of

    cfg = T[args.target]
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    is_b = [tok.convert_ids_to_tokens(i).count("Ċ") >= 2 for i in range(len(tok))]
    close_id = tok.convert_tokens_to_ids("</think>")
    qtext = {}
    for c in list((ROOT / "data/xfer8b").glob("*_math*.jsonl")) + list((ROOT / "data/policy").glob("*.jsonl")):
        for line in c.open():
            r = json.loads(line)
            qtext[r["question_id"]] = r["question"]
    # Verbatim selection logic of xfer_local_injection.py (defaults: val/test, 3 per
    # label per trace, max prefix 6144, rng seed 0, trace order as run).
    rng = random.Random(0)
    sites = []
    for path in cfg["traces"]:
        origin = next(v for k, v in cfg["origin"].items() if k in path)
        for line in open(ROOT / path):
            r = json.loads(line)
            if r["policy"] != "base" or r["seed"] != 0 or split_of(r["question_id"]) not in ("val", "test"):
                continue
            msgs = [{"role": "system", "content": FORMAT},
                    {"role": "user", "content": qtext[r["question_id"]].strip()}]
            prompt = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                 enable_thinking=True), add_special_tokens=False)["input_ids"]
            gen = r["token_ids"]
            close = gen.index(close_id) if close_id in gen else len(gen)
            P = len(prompt)
            cands = {0: [], 1: []}
            for i, t in enumerate(gen[: close - 1]):
                if is_b[t] and P + i < 6144 and i >= 16:
                    cands[int(first_sentence_doubt(tok.decode(gen[i + 1 : i + 48])))].append(P + i)
            for lab in (0, 1):
                for p in rng.sample(cands[lab], min(3, len(cands[lab]))):
                    sites.append({"origin": origin, "qid": r["question_id"], "pos": p, "label": lab,
                                  "ids": (prompt + gen)[: p + 1]})
    s17 = [json.loads(line) for line in open(ROOT / cfg["screen17"])]
    sT = [json.loads(line) for line in open(ROOT / cfg["screenT"])]
    assert len(s17) == len(sT) == len(sites), (len(s17), len(sT), len(sites))
    for s, a, b in zip(sites, s17, sT, strict=True):
        assert (s["qid"], s["pos"], s["label"]) == (a["qid"], a["pos"], a["label"]) == (b["qid"], b["pos"], b["label"])
        s["p_s"], s["p_b"] = a["p_base"], b["p_base"]
    torch.save(sites, out_dir(args.target) / "sites.pt")
    from collections import Counter
    print(f"{len(sites)} sites verified against both screens; origins {Counter(s['origin'] for s in sites)}")


# -------------------------------------------------------------------------------- extract
def cmd_extract(args: argparse.Namespace) -> None:
    from transformers import AutoModelForCausalLM

    cfg = T[args.target]
    d = out_dir(args.target)
    sites = torch.load(d / "sites.pt", weights_only=False)
    # One forward per trace prefix covering all its sites (causal: identical states).
    groups: dict[tuple[str, str], list[int]] = {}
    for k, s in enumerate(sites):
        groups.setdefault((s["origin"], s["qid"]), []).append(k)
    for which, model_id, layer in (("s", "Qwen/Qwen3-1.7B", 20), ("b", cfg["model"], cfg["layer"])):
        if (d / f"h_{which}.pt").exists():
            continue
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, device_map="cuda").eval()
        grab: dict[str, torch.Tensor] = {}
        model.model.layers[layer].register_forward_hook(
            lambda _m, _i, out: grab.__setitem__("h", out[0] if isinstance(out, tuple) else out))
        with torch.inference_mode():
            x = torch.tensor([sites[0]["ids"][:256]], device="cuda")
            ref = model(input_ids=x, output_hidden_states=True).hidden_states[layer + 1]
            assert torch.equal(grab["h"], ref), "hook != hidden_states[l+1]"
        model.model.layers = model.model.layers[: layer + 1]
        H = torch.zeros(len(sites), model.config.hidden_size, dtype=torch.bfloat16)
        with torch.inference_mode():
            for ks in groups.values():
                longest = max(ks, key=lambda k: len(sites[k]["ids"]))
                ids = sites[longest]["ids"]
                for k in ks:  # every site's prefix is a prefix of the longest one
                    assert ids[: len(sites[k]["ids"])] == sites[k]["ids"]
                model.model(input_ids=torch.tensor([ids], device="cuda"), use_cache=False)
                for k in ks:
                    H[k] = grab["h"][0, sites[k]["pos"]].cpu()
        torch.save(H, d / f"h_{which}.pt")
        print(f"wrote h_{which}.pt ({model_id} L{layer})", flush=True)
        del model
        torch.cuda.empty_cache()


# ------------------------------------------------------------------------------ verbalize
def cmd_verbalize(args: argparse.Namespace) -> None:
    from xfer_nla_tools import FrozenNLA

    cfg = T[args.target]
    d = out_dir(args.target)
    m = torch.load(ROOT / cfg["maps"])
    vec = {"s": torch.load(d / "h_s.pt").float(),
           "b": torch.load(d / "h_b.pt").float() @ m["W_E"].float().T + m["b_E"].float()}
    f = FrozenNLA()
    for w, v in vec.items():
        if (d / f"z_{w}.json").exists():  # z_s is map-independent (linked in the mix variant)
            continue
        Z, R = [], []
        for s in range(args.samples):
            z = f.verbalize(v, batch=128, temperature=1.0, seed=2000 + s)
            Z.append(z)
            R.append(f.reconstruct(z, batch=128).cpu())
        (d / f"z_{w}.json").write_text(json.dumps(Z))
        torch.save(torch.stack(R, 1), d / f"r_{w}.pt")
        print(f"verbalized z_{w}", flush=True)


# ---------------------------------------------------------------------------------- judge
JUDGE_EXPL = """Below is a natural-language description of what a language model is thinking at one point in the middle of solving a math problem. The model has just finished a paragraph of its reasoning.

<description>
{z}
</description>

Based only on this description, how likely is it that the model's NEXT paragraph starts by doubting, re-checking or second-guessing (e.g. "Wait", "Hmm", "But let me verify", "Actually"), rather than continuing the derivation?

Return a JSON object inside <json></json>: {{"p_doubt": <integer 0-100>}}"""

JUDGE_TEXT = """Below are the last two paragraphs a language model wrote while solving a math problem.

<text>
{t}
</text>

How likely is it that the model's NEXT paragraph starts by doubting, re-checking or second-guessing (e.g. "Wait", "Hmm", "But let me verify", "Actually"), rather than continuing the derivation?

Return a JSON object inside <json></json>: {{"p_doubt": <integer 0-100>}}"""


def parse_p(raw: str | None) -> float | None:
    if not raw:
        return None
    m = re.search(r'"p_doubt"\s*:\s*(\d+(?:\.\d+)?)', raw)
    return float(m.group(1)) if m else None


def cmd_judge(args: argparse.Namespace) -> None:
    from transformers import AutoTokenizer

    from reasoning_attention.config import ExplainerConfig
    from reasoning_attention.datagen.providers import OpenAIProvider

    d = out_dir(args.target)
    sites = torch.load(d / "sites.pt", weights_only=False)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    prompts: dict[str, str] = {}
    for w in ("s", "b"):
        z = json.loads((d / f"z_{w}.json").read_text())[0]
        for k, t in enumerate(z):
            prompts[f"{w}:{k}"] = JUDGE_EXPL.format(z=t)
    for k, s in enumerate(sites):
        paras = [p for p in tok.decode(s["ids"]).split("\n\n") if p.strip()]
        prompts[f"text:{k}"] = JUDGE_TEXT.format(t="\n\n".join(paras[-2:]))
    path = d / "judge.jsonl"
    done = {}
    if path.exists():
        for line in open(path):
            r = json.loads(line)
            if parse_p(r["raw"]) is not None:
                done[r["id"]] = r
    todo = [k for k in prompts if k not in done]
    print(f"{len(done)} done, {len(todo)} to judge", flush=True)
    prov = OpenAIProvider(ExplainerConfig(reasoning_effort="low", max_output_tokens=4000, concurrency=32))
    for i in range(0, len(todo), 64):
        chunk = todo[i : i + 64]
        outs = prov.complete([prompts[k] for k in chunk])
        with open(path, "a") as f:
            for k, raw in zip(chunk, outs, strict=True):
                f.write(json.dumps({"id": k, "raw": raw}) + "\n")
        print(f"  {i + len(chunk)}/{len(todo)}", flush=True)


# --------------------------------------------------------------------------------- report
DOUBT_CUES = ("wait", "double-check", "double check", "verify", "verif", "reconsider", "re-check",
              "recheck", "mistake", "hmm", "but ", "however", "actually", "hold on", "check")


def cue(z: str) -> float:  # Finding 15's predicts_doubt
    paras = [p for p in z.split("\n\n") if p.strip()] or [z]
    return float(any(c in paras[-1].lower() for c in DOUBT_CUES))


def _rank(x: np.ndarray) -> np.ndarray:
    """Average ranks (ties share the mean rank)."""
    order = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    sums = np.bincount(inv, weights=r)
    return (sums / cnt)[inv]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _rank(np.asarray(a, float)), _rank(np.asarray(b, float))
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def cluster_ci(x: np.ndarray, y: np.ndarray, groups: np.ndarray, B: int = 2000) -> tuple[float, float, float]:
    rng = np.random.default_rng(0)
    uq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uq}
    stats = []
    for _ in range(B):
        pick = np.concatenate([idx[g] for g in rng.choice(uq, len(uq))])
        stats.append(spearman(x[pick], y[pick]))
    stats = [v for v in stats if v == v]
    return spearman(x, y), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    from xfer_fit_maps import auc as _auc

    return _auc(np.asarray(score, float), np.asarray(y)) if 0 < y.sum() < len(y) else float("nan")


def cmd_report(args: argparse.Namespace) -> None:
    cfg = T[args.target]
    d = out_dir(args.target)
    sites = torch.load(d / "sites.pt", weights_only=False)
    dp = np.array([s["p_b"] - s["p_s"] for s in sites])
    pT = np.array([s["p_b"] for s in sites])
    p17 = np.array([s["p_s"] for s in sites])
    qid = np.array([s["qid"] for s in sites])
    origin = np.array([s["origin"] for s in sites])
    Z = {w: json.loads((d / f"z_{w}.json").read_text()) for w in ("s", "b")}
    R = {w: torch.load(d / f"r_{w}.pt").float() for w in ("s", "b")}
    N = torch.load(ROOT / N_PATH)["unit"].float()
    maps = torch.load(ROOT / cfg["maps"])
    hs = torch.load(d / "h_s.pt").float()
    hb = torch.load(d / "h_b.pt").float()
    Eh = hb @ maps["W_E"].float().T + maps["b_E"].float()
    # N points plain-ward, so NEGATE projections to make every score "higher = more doubt".
    scores = {
        "cue": {w: np.mean([[cue(z) for z in Zs] for Zs in Z[w]], 0) for w in Z},
        "AR.N": {w: -(q(R[w]) @ N).mean(1).numpy() for w in R},
    }
    jpath = d / "judge.jsonl"
    text_base = None
    if jpath.exists():
        J = {}
        for line in open(jpath):
            r = json.loads(line)
            p = parse_p(r["raw"])
            if p is not None:
                J[r["id"]] = p
        if all(f"{w}:{k}" in J for w in "sb" for k in range(len(sites))):
            scores["judge"] = {w: np.array([J[f"{w}:{k}"] for k in range(len(sites))]) for w in "sb"}
        if all(f"text:{k}" in J for k in range(len(sites))):
            text_base = np.array([J[f"text:{k}"] for k in range(len(sites))])
    rep: dict = {"n_sites": len(sites), "dp_mean": float(dp.mean()), "dp_sd": float(dp.std()),
                 "origins": {o: int((origin == o).sum()) for o in np.unique(origin)}}
    strata = {"all": np.ones(len(sites), bool), **{f"origin={o}": origin == o for o in np.unique(origin)}}
    prim = {}
    for name, sc in scores.items():
        diff = sc["b"] - sc["s"]
        prim[name] = {st: cluster_ci(diff[m], dp[m], qid[m]) for st, m in strata.items()}
    # positive control: the continuous path carries a target-specific signal
    cont = -((q(Eh) @ N) - (q(hs) @ N)).numpy()
    prim["control: <E h_b,N> - <h_s,N> (no text)"] = {st: cluster_ci(cont[m], dp[m], qid[m]) for st, m in strata.items()}
    rep["primary: spearman(score_b - score_s, dp) [point, lo, hi]"] = prim
    # secondary: on the disagreement quartiles, predict the TARGET's behaviour
    lo, hi = np.percentile(dp, [25, 75])
    dis = (dp <= lo) | (dp >= hi)
    yT = (pT[dis] > 0.5).astype(int)
    sec = {name: {"AUC score(z_b)": auc(sc["b"][dis], yT), "AUC score(z_s)": auc(sc["s"][dis], yT)}
           for name, sc in scores.items()}
    sec["text-only: 1.7B p_base"] = auc(p17[dis], yT)
    if text_base is not None:
        sec["text-only: judge on last 2 paragraphs"] = auc(text_base[dis], yT)
        rep["text-only judge vs dp (should be ~0)"] = cluster_ci(text_base, dp, qid)
    rep[f"secondary: AUC for p_target > 0.5 on the dp quartiles (n={int(dis.sum())})"] = sec
    (d / "report.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["sites", "extract", "verbalize", "judge", "report"])
    ap.add_argument("--target", choices=list(T), required=True)
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--variant", choices=["orig", "mix"], default="orig")
    args = ap.parse_args()
    global VARIANT
    VARIANT = args.variant
    if VARIANT == "mix":
        for tag in T:
            T[tag]["maps"] = f"data/xfer_claim4/maps_mix/{tag}_maps.pt"
    {"sites": cmd_sites, "extract": cmd_extract, "verbalize": cmd_verbalize,
     "judge": cmd_judge, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
