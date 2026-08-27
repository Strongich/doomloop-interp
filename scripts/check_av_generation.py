"""Does the SFT'd AV actually generate usable explanations?

Stage-1 was signed off on teacher-forced loss (AV v2: 1.3609, ppl 3.90) — which
says nothing about free-running generation. At the first working GRPO step the AV
turned out to emit degenerate text ("a the word", "or or", stray CJK/Arabic
tokens) and closed its <explanation> tag in only 1 of 16 samples, so ~94% of
rollouts took the flat failure reward and the critic was fit against noise.

This isolates the AV from the entire RL harness: our own injection path, our own
checkpoint, no Ray/sglang/miles. If output is clean here, the problem is in the RL
rollout path; if it is degenerate here too, Stage-1 is not finished.

Reports the completion rate (did it close the tag inside the RL response cap) and
a degeneracy proxy (immediate token repetition, non-latin fraction), greedy vs
sampled — RL samples at T=1, `NLA.verbalize` defaults to greedy, and the two can
differ sharply for a weak generator.

Usage:
    python scripts/check_av_generation.py \
        --checkpoint /workspace/data/checkpoints/av_sft \
        --parquet /workspace/data/rl/rl.parquet --n 8
"""

import argparse
import dataclasses
import re
import unicodedata

import numpy as np
import pyarrow.parquet as pq
import torch

from reasoning_attention.config import NLAConfig
from reasoning_attention.nla.model import NLA

EXPL_CLOSE = "</explanation>"


def _degeneracy(text: str) -> tuple[float, float]:
    """(immediate-word-repetition rate, non-latin char fraction)."""
    words = re.findall(r"\w+", text.lower())
    reps = sum(1 for a, b in zip(words, words[1:]) if a == b) / max(1, len(words) - 1)
    letters = [c for c in text if c.isalpha()]
    nonlatin = sum(
        1 for c in letters if not unicodedata.name(c, "").startswith("LATIN")
    ) / max(1, len(letters))
    return reps, nonlatin


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="AV checkpoint dir (HF)")
    ap.add_argument("--parquet", required=True, help="parquet with an activation_vector column")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=150,
        help="match --rollout-max-response-len so the completion rate is comparable",
    )
    args = ap.parse_args()

    cfg = dataclasses.replace(NLAConfig(), model_id=args.checkpoint)
    print(f"loading AV from {args.checkpoint} (injection_scale={cfg.injection_scale})")
    nla = NLA.from_pretrained(cfg)

    # Slice BEFORE to_pylist(). A row group here is ~100k rows x 2048 floats, and
    # materializing that as Python floats takes minutes and gigabytes — the same
    # trap their data_source hit (list -> np.asarray, ~2.6s GC stalls).
    pf = pq.ParquetFile(args.parquet)
    batch = next(pf.iter_batches(batch_size=args.n, columns=["activation_vector"]))
    flat = batch.column("activation_vector").flatten().to_numpy(zero_copy_only=False)
    vecs = flat.astype(np.float32).reshape(len(batch), -1)[: args.n]
    print(f"{len(vecs)} activations, norms {vecs.__abs__().max():.0f} max, "
          f"L2 {np.linalg.norm(vecs, axis=1).mean():.0f} mean\n")

    # Explicit sampling, NOT the checkpoint's generation_config.json. Qwen3 ships
    # temperature=0.6/top_k=20/top_p=0.95 there, so a bare do_sample=True silently
    # runs the model page's recommended settings — which is NOT what the RL rollout
    # does (miles defaults: T=1.0, top_p=1.0, top_k=-1).
    regimes = [
        ("greedy", {"do_sample": False}),
        ("rollout T=1.0/top_p=1.0/top_k=off", {"do_sample": True, "temperature": 1.0, "top_p": 1.0, "top_k": 0}),
        ("qwen3 rec T=0.7/top_p=0.8/top_k=20", {"do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20}),
    ]
    for label, gen_kwargs in regimes:
        closed = 0
        reps_all, nl_all = [], []
        print(f"================ {label} ================")
        for i, v in enumerate(vecs):
            text = nla.verbalize(
                torch.from_numpy(v),
                max_new_tokens=args.max_new_tokens,
                enable_thinking=False,
                **gen_kwargs,
            )
            ok = EXPL_CLOSE in text
            closed += ok
            reps, nl = _degeneracy(text)
            reps_all.append(reps)
            nl_all.append(nl)
            print(f"--- {i} closed={ok} rep={reps:.2f} nonlatin={nl:.2f} ---")
            print(text.strip()[:400])
        print(f"\n{label}: closed {closed}/{len(vecs)}, "
              f"mean rep {np.mean(reps_all):.3f}, mean nonlatin {np.mean(nl_all):.3f}\n")


if __name__ == "__main__":
    main()
