#!/usr/bin/env python3
r"""SEAL's steering vector (Chen et al. 2025, arXiv 2504.07986), EXPERIMENT-reviewer-baselines.md
§B4b, following the official code (github.com/VITA-Group/SEAL, `hidden_analysis.py` +
`vector_generation.py`):

  thoughts   the thinking text (between <think> and </think>) split at every token whose
             piece contains "\n\n" ("ĊĊ" in the vocabulary string); the thought after split
             token i runs to the next split token (or the end of thinking), decoded and
             stripped of " " and "\n". Text before the first split is not a thought.
  reflection starts with "Wait", or contains verify / make sure / hold on / think again /
             's correct / 's incorrect / Let me check / seems right (case-insensitive)
  transition (else) starts with "Alternatively", or contains think differently / another
             way / approach / method / solution / strategy / technique
  execution  every other thought
  state      the block-l output at the split token that opens the thought
  S          mean(H, execution) - mean(H, reflection U transition), pooled over thoughts,
             NOT normalized (the official code's vector is the negative, used at coef -1)

Source: 1,000 correct (D69 grader, regraded here) base MATH-train rollouts of the model's own
generation, drawn with random.Random(20261005) from the sorted candidate keys; states captured
through vLLM (CaptureWorkerExtension, prefill-only), as in gate_states.py.

Output {"vector": S, "unit": S/||S||, ...}: `--dir SEAL=<out>` reads "vector" (absolute
push), `--dir SEALu=<out>` reads "unit" (N's norm-relative schedule).

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/baselines_seal_vector.py --model Qwen/Qwen3-1.7B \
        --layer 20 --journals 'data/distill/gen/shard*/rollouts.jsonl' \
        --questions 'data/distill/gen/shard*.jsonl' --out data/baselines/seal/seal_1p7b_L20.pt \
        --compare N=data/pool/dir_A_1trace.pt D=data/dom/dir_D_diffmeans.pt
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import random
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

SEED = 20261005
N_TRACES = 1000
CHECK_WORDS = ["verify", "make sure", "hold on", "think again", "'s correct", "'s incorrect",
               "Let me check", "seems right"]
CHECK_PREFIX = ["Wait"]
# Protocol text; the official list spells the first one "think differenly" (a typo that
# never matches), recorded in §B10.
SWITCH_WORDS = ["think differently", "another way", "another approach", "another method",
                "another solution", "another strategy", "another technique"]
SWITCH_PREFIX = ["Alternatively"]


def classify(step: str) -> str:
    s = step.lower()
    if any(s.startswith(p.lower()) for p in CHECK_PREFIX) or any(w.lower() in s for w in CHECK_WORDS):
        return "reflection"
    if any(s.startswith(p.lower()) for p in SWITCH_PREFIX) or any(w.lower() in s for w in SWITCH_WORDS):
        return "transition"
    return "execution"


def thoughts(ids: list[int], tok, split: set[int], open_id: int, close_id: int):
    """[(index into ids of the split token, label)] over the thinking span, official rule."""
    if open_id not in ids:
        return []
    start = ids.index(open_id) + 1
    end = ids.index(close_id, start) if close_id in ids[start:] else len(ids)
    think = ids[start:end]
    idx = [i for i, t in enumerate(think) if t in split] + [len(think)]
    out = []
    for i in range(len(idx) - 1):
        step = tok.decode(think[idx[i] + 1 : idx[i + 1]]).strip(" ").strip("\n")
        out.append((start + idx[i], classify(step)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--journals", required=True)
    ap.add_argument("--questions", required=True, help="glob of cohort jsonl with question text")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--compare", nargs="*", default=[], metavar="NAME=PATH",
                    help="unit directions to report cos(S, .) against")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"{args.out} exists")

    from transformers import AutoTokenizer

    from reasoning_attention.grading import grade
    from sel_common import Paragraphs
    from xfer_vllm_capture import build_capture_llm, capture

    tok = AutoTokenizer.from_pretrained(args.model)
    P = Paragraphs(tok)
    split = {i for t, i in tok.get_vocab().items() if "ĊĊ" in t}
    open_id = int(tok.convert_tokens_to_ids("<think>"))
    close_id = int(tok.convert_tokens_to_ids("</think>"))
    qs = {}
    for f in sorted(glob.glob(str(ROOT / args.questions))):
        for x in open(f):
            r = json.loads(x)
            qs[r["question_id"]] = r

    cands = {}
    n_seen = n_wrong = 0
    for f in sorted(glob.glob(str(ROOT / args.journals))):
        for x in open(f):
            r = json.loads(x)
            if r["policy"] != "base" or r["dataset"] != "mathtrain":
                continue
            n_seen += 1
            if not grade(r["text"], r["gold"]).is_correct:
                n_wrong += 1
                continue
            cands[f"{r['question_id']}|{r['seed']}"] = r
    keys = sorted(cands)
    pick = random.Random(SEED).sample(keys, N_TRACES)
    print(f"{n_seen} base MATH-train rollouts, {n_wrong} wrong under D69, {len(keys)} "
          f"candidates; picked {len(pick)}", flush=True)

    seqs, poss, labels = [], [], []
    for k in pick:
        r = cands[k]
        prompt = P.prompt_ids(qs[r["question_id"]]["question"])
        if len(prompt) != r["prompt_tokens"]:
            raise SystemExit(f"{k}: rebuilt prompt has {len(prompt)} tokens, journal {r['prompt_tokens']}")
        th = thoughts(r["token_ids"], tok, split, open_id, close_id)
        if not th:
            continue
        seqs.append(prompt + r["token_ids"][: th[-1][0] + 1])
        poss.append([len(prompt) + i for i, _ in th])
        labels.append([lab for _, lab in th])

    d = None
    sums: dict[str, torch.Tensor] = {}
    counts = {"execution": 0, "reflection": 0, "transition": 0}
    norm_sum, norm_n = 0.0, 0
    max_len = max(len(s) for s in seqs) + 8
    llm = build_capture_llm(args.model, [args.layer], max_len,
                            gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=64)
    tmp = tempfile.mkdtemp(prefix="seal_cap_", dir="/workspace/baselines_logs")
    B = 64
    for i in range(0, len(seqs), B):
        H = capture(llm, seqs[i : i + B], poss[i : i + B], [args.layer], tmp)[args.layer].double()
        labs = [lab for ls in labels[i : i + B] for lab in ls]
        assert H.shape[0] == len(labs)
        d = H.shape[1]
        norm_sum += float(H.norm(dim=-1).sum())
        norm_n += H.shape[0]
        for c in counts:
            m = torch.tensor([lab == c for lab in labs])
            counts[c] += int(m.sum())
            sums[c] = sums.get(c, torch.zeros(d, dtype=torch.float64)) + H[m].sum(0)
        print(f"{min(i + B, len(seqs))}/{len(seqs)} traces", flush=True)

    mean_exec = sums["execution"] / counts["execution"]
    nonexec = counts["reflection"] + counts["transition"]
    mean_rt = (sums["reflection"] + sums["transition"]) / nonexec
    S = (mean_exec - mean_rt).float()
    mean_h = norm_sum / norm_n
    cos = {}
    for spec in args.compare:
        name, _, path = spec.partition("=")
        u = torch.load(path, map_location="cpu", weights_only=False)["unit"].float()
        cos[name] = float(torch.nn.functional.cosine_similarity(S, u, dim=0))
    out = {
        "vector": S, "unit": S / S.norm(), "kind": "SEAL", "model": args.model, "layer": args.layer,
        "counts": counts, "traces": len(seqs), "norm": float(S.norm()),
        "mean_h_norm_at_splits": mean_h, "norm_ratio": float(S.norm()) / mean_h,
        "cos": cos, "seed": SEED, "selected": pick,
        "selected_sha256": hashlib.sha256("\n".join(pick).encode()).hexdigest(),
        "rule": "S = mean(exec) - mean(reflection U transition), official SEAL thought split",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, args.out)
    summary = {k: v for k, v in out.items() if k not in ("vector", "unit", "selected")}
    args.out.with_suffix(".json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
