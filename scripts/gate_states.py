#!/usr/bin/env python3
r"""EXPERIMENT-probe-gated.md §G2: layer-20 block output at EVERY paragraph boundary inside
<think> of the 1,000 selective-study source traces, captured through vLLM
(CaptureWorkerExtension, prefill-only) -- the engine family that computes the gate.

  site      prompt_len + b for every boundary b of sel_common.Paragraphs.split (the probe
            records' `boundaries`), i.e. the consumed boundary token the hook edits
  label     1 if the doubt episode at that boundary is `productive` (episodes_source.jsonl),
            else 0 (redundant / other / harmful / failed / excluded doubt, or no doubt)

Output: data/gated/states_source.pt = {"h": [N, 2048] bf16, "rows": [dict], "layer": 20}

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/gate_states.py
"""

from __future__ import annotations

import glob
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sel_common import Paragraphs  # noqa: E402
from sel_probe import key_of, load_selected  # noqa: E402
from xfer_vllm_capture import build_capture_llm, capture  # noqa: E402

SEL = ROOT / "data/selective"
OUT = ROOT / "data/gated/states_source.pt"
LAYER = 20


def main() -> None:
    from transformers import AutoTokenizer

    from reasoning_attention.config import MODEL_ID

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    P = Paragraphs(tok)
    recs = load_selected("data/distill/gen/shard*/rollouts.jsonl", SEL / "source1000.jsonl", "0/1")
    qs = {json.loads(x)["question_id"]: json.loads(x) for x in open(ROOT / "data/distill/train.jsonl")}
    probes = {}
    for f in sorted(glob.glob(str(SEL / "probes/source_s*.jsonl"))):
        for x in open(f):
            r = json.loads(x)
            probes[r["key"]] = {"boundaries": r["boundaries"], "doubt": r["doubt"]}
    lab = {}
    for x in open(SEL / "episodes_source.jsonl"):
        e = json.loads(x)
        lab[(e["key"], e["k"])] = e["label"]
    seqs, poss, rows = [], [], []
    for r in recs:
        key = key_of(r)
        prompt = P.prompt_ids(qs[r["question_id"]]["question"])
        assert len(prompt) == r["prompt_tokens"], key
        sp = P.split(r["token_ids"])
        bs = sp["boundaries"]
        assert bs == probes[key]["boundaries"] and sp["doubt"] == probes[key]["doubt"], key
        if not bs:
            continue
        seqs.append(prompt + r["token_ids"][: bs[-1] + 1])
        poss.append([len(prompt) + b for b in bs])
        for k, b in enumerate(bs):
            ep = lab.get((key, k), "" if not sp["doubt"][k] else "missing")
            assert ep != "missing", (key, k)
            rows.append({"key": key, "question_id": r["question_id"], "level": qs[r["question_id"]].get("level"),
                         "k": k, "cut": b, "g": b + 1, "doubt": int(sp["doubt"][k]),
                         "episode": ep, "y": int(ep == "productive")})
    max_len = max(len(s) for s in seqs) + 8
    llm = build_capture_llm(MODEL_ID, [LAYER], max_len, gpu_memory_utilization=0.85, max_num_seqs=64)
    hs = []
    tmp = tempfile.mkdtemp(prefix="gate_cap_", dir="/workspace/distill_logs")
    for i in range(0, len(seqs), 64):
        hs.append(capture(llm, seqs[i : i + 64], poss[i : i + 64], [LAYER], tmp)[LAYER].to(torch.bfloat16))
        print(f"{min(i + 64, len(seqs))}/{len(seqs)} traces", flush=True)
    H = torch.cat(hs)
    assert H.shape[0] == len(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"h": H, "rows": rows, "layer": LAYER, "engine": "vllm CaptureWorkerExtension"}, OUT)
    print(f"wrote {len(rows)} boundary states ({sum(r['y'] for r in rows)} positive) -> {OUT}",
          flush=True)


if __name__ == "__main__":
    main()
