#!/usr/bin/env python3
"""Paired-activation capture through vLLM (for targets too large for HF, e.g. 235B-A22B).

`capture(llm, seqs, positions)` runs each token sequence as a prefill-only request
(max_tokens=1) and returns {layer: tensor[n_positions_total, d]} of FULL block outputs
(mlp_out + residual), ordered like the flattened positions. Works with pipeline
parallelism: only the rank that owns a layer captures it.

    uv run python scripts/xfer_vllm_capture.py --check      # parity vs HF on the 1.7B
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def build_capture_llm(model: str, layers: list[int], max_model_len: int, pp: int = 1,
                      gpu_memory_utilization: float = 0.85, max_num_seqs: int = 64) -> Any:
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM

    llm = LLM(model=model, dtype="bfloat16", tensor_parallel_size=1, pipeline_parallel_size=pp,
              enforce_eager=True, async_scheduling=False, enable_prefix_caching=False,
              enable_chunked_prefill=True, max_model_len=max_model_len, max_num_seqs=max_num_seqs,
              max_num_batched_tokens=8192, gpu_memory_utilization=gpu_memory_utilization,
              worker_extension_cls="reasoning_attention.serving.vllm_steering.CaptureWorkerExtension")
    owned = [r["owned"] for r in llm.collective_rpc("install_capture", args=(layers,))]
    if sorted(sum(owned, [])) != sorted(layers):
        raise RuntimeError(f"capture layers not owned exactly once: {owned}")
    return llm


def capture(llm: Any, seqs: list[list[int]], positions: list[list[int]],
            layers: list[int]) -> dict[int, torch.Tensor]:
    from vllm import SamplingParams

    params = [SamplingParams(max_tokens=1, temperature=0.0,
                             extra_args={"capture_id": str(i), "capture_positions": list(p)})
              for i, p in enumerate(positions)]
    llm.generate([{"prompt_token_ids": s} for s in seqs], params, use_tqdm=False)
    merged: dict[str, dict[int, dict[int, torch.Tensor]]] = {}
    for part in llm.collective_rpc("pop_captures"):
        for cid, by_layer in part.items():
            for layer, by_pos in by_layer.items():
                merged.setdefault(cid, {}).setdefault(layer, {}).update(by_pos)
    out = {}
    for layer in layers:
        rows = []
        for i, pos in enumerate(positions):
            got = merged.get(str(i), {}).get(layer, {})
            missing = [p for p in pos if p not in got]
            if missing:
                raise RuntimeError(f"seq {i} layer {layer}: missing positions {missing[:5]}")
            rows += [torch.as_tensor(got[p], dtype=torch.float32) for p in pos]
        out[layer] = torch.stack(rows)
    return out


def check() -> None:
    import json
    import random

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    rng = random.Random(0)
    seqs, pos = [], []
    for i, line in enumerate(open(ROOT / "data/xfer8b/web3000.jsonl")):
        if i >= 48:
            break
        ids = tok(json.loads(line)["text"], add_special_tokens=False)["input_ids"][:3000]
        seqs.append(ids)
        pos.append(sorted(rng.sample(range(16, len(ids)), 6)))
    llm = build_capture_llm("Qwen/Qwen3-1.7B", [20], 4096, gpu_memory_utilization=0.5)
    got = capture(llm, seqs, pos, [20])[20].float()
    del llm
    hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16,
                                              device_map="cuda").eval()
    ref = []
    with torch.inference_mode():
        for s, p in zip(seqs, pos):
            h = hf(input_ids=torch.tensor([s], device="cuda"), output_hidden_states=True).hidden_states[21]
            ref.append(h[0, p].float().cpu())
    ref = torch.cat(ref)
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    rel = (got - ref).norm(dim=-1) / ref.norm(dim=-1)
    print(f"{len(ref)} positions: cos min {cos.min():.5f} mean {cos.mean():.5f}; "
          f"rel err median {rel.median():.4f} max {rel.max():.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    if ap.parse_args().check:
        check()
