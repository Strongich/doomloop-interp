#!/usr/bin/env python3
"""GPU checks the CPU suite cannot make: does the delay bite in the real engine?

Three things are verified with GREEDY decoding, so any difference is the
intervention and not sampling noise:

1. **alpha=0 identity.** A configured direction at alpha 0 must reproduce the
   unsteered token IDs exactly. This catches a hook that perturbs the residual
   even when it believes it is disabled.
2. **Delay bites, and only after it.** Two runs of the SAME direction and alpha
   differing only in start_delay must agree token-for-token up to the first site
   that the earlier policy edits and the later one does not -- and then be free
   to diverge. An off-by-one in `g`, or a delay that silently does nothing, both
   show up here and nowhere else.
3. **Nonzero intervention is observable.** At least one delay pair must actually
   diverge; identical output everywhere would mean the direction is being
   applied at no strength.

    uv run python scripts/check_policy_delay.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import json

import torch


def main() -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from reasoning_attention.config import MODEL_ID, NLAConfig
    from reasoning_attention.data.math_datasets import build_messages
    from reasoning_attention.serving.vllm_steering import build_steering_llm
    from reasoning_policy_vllm import DIRECTIONS, expected_sites
    from suppress_answer import NEWLINE_CHAR

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    boundaries = [
        i for i in range(len(tok)) if tok.convert_ids_to_tokens(i).count(NEWLINE_CHAR) >= 2
    ]
    boundary_set = set(boundaries)
    close_id = int(tok.convert_tokens_to_ids("</think>"))
    unit = (
        torch.load(DIRECTIONS["N"], map_location="cpu", weights_only=False)["unit"]
        .float()
        .tolist()
    )

    rows = [
        json.loads(x) for x in Path("data/policy/dev200.jsonl").read_text().splitlines() if x
    ][:8]
    prompts = []
    for r in rows:
        text = tok.apply_chat_template(
            build_messages(r["question"]), tokenize=False,
            add_generation_prompt=True, enable_thinking=True,
        )
        prompts.append({"prompt_token_ids": list(tok(text, add_special_tokens=False)["input_ids"])})

    max_new = 1024
    llm = build_steering_llm(
        MODEL_ID, NLAConfig().extraction_layer, boundaries, close_id,
        max(len(p["prompt_token_ids"]) for p in prompts) + max_new, 8, 0.85, 2048,
    )
    greedy = [SamplingParams(temperature=0.0, max_tokens=max_new) for _ in prompts]

    def run(unit_arg, alpha, delay):
        llm.collective_rpc("configure_reasoning_steering", args=(unit_arg, alpha, True, delay))
        outs = llm.generate(prompts, greedy, use_tqdm=False)
        return [list(o.outputs[0].token_ids) for o in outs]

    failures = []
    base = run(None, 0.0, 0)

    # 1. alpha=0 with a real direction must be bit-identical to no direction.
    zero = run(unit, 0.0, 0)
    same = sum(a == b for a, b in zip(base, zero, strict=True))
    print(f"[1] alpha=0 identity: {same}/{len(base)} sequences identical")
    if same != len(base):
        failures.append("alpha=0 changed the output")

    # 2/3. Delay must agree up to the first site the earlier policy edits alone.
    runs = {d: run(unit, 1.0, d) for d in (0, 128, 512)}
    diverged = 0
    for i, ids_base in enumerate(base):
        plen = len(prompts[i]["prompt_token_ids"])
        for early, late in ((0, 128), (128, 512)):
            a, b = runs[early][i], runs[late][i]
            sites_e = expected_sites(a, plen, boundary_set, close_id, early)
            sites_l = expected_sites(a, plen, boundary_set, close_id, late)
            extra = [s for s in sites_e if s not in sites_l]
            if not extra:
                continue  # no site distinguishes these policies on this question
            # First token the earlier policy could influence is the one AFTER the
            # edited site, i.e. generated index (site - plen) + 1.
            cut = extra[0] - plen + 1
            if a[:cut] != b[:cut]:
                failures.append(f"q{i} d{early} vs d{late}: diverged before the first extra site")
            if a != b:
                diverged += 1
    print(f"[2] prefix agreement before the first distinguishing site: "
          f"{'OK' if not failures else 'FAILED'}")
    print(f"[3] delay pairs that actually diverged afterwards: {diverged}")
    if diverged == 0:
        failures.append("no delay pair diverged; the intervention is not biting")

    if failures:
        for f in failures:
            print("FAIL:", f)
        raise SystemExit(1)
    print("\nAll delay checks passed")


if __name__ == "__main__":
    main()
