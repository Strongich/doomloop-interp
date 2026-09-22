#!/usr/bin/env python3
"""Is the NLA's low reconstruction score on reasoning traces a DOMAIN gap?

`ar_fidelity.py` found ~0.25-0.31 FVE on block-boundary states in math reasoning
traces, against the 0.630 the RL run reported on its own eval. Before spending a
pod restart and a full RL epoch on a reasoning corpus, check whether the gap is
the domain or our measurement.

Same code path, two sources:

  web              Ultra-FineWeb, raw text — exactly the Stage-1/2 training
                   distribution
  reasoning_rand   our own Qwen3-1.7B traces through the chat template, positions
                   sampled uniformly — isolates the DOMAIN change
  reasoning_block  the same traces, positions at block boundaries (last token
                   before a `\n\n`) — the positions every finding is measured at,
                   so this isolates the POSITION change on top
  web_block        web prose at ITS paragraph breaks. Ultra-FineWeb contains no
                   `\n\n` at all (0 of 60 sampled documents) — its paragraphs are
                   single newlines — so the delimiter is per source, not global.
  reasoning_sent   reasoning at sentence ends that are NOT block boundaries. The
                   sharpest control: same domain, same "structural break" status,
                   but the model is continuing a paragraph rather than choosing
                   what to do next.
  reasoning_gold   reasoning MID-SENTENCE, inside the sentence that first states
                   the gold answer, at positions before the answer itself. Not a
                   break at all, so if the positional gap is about breaks this
                   should score like reasoning_rand — and it is a position the
                   study can actually use: the state while the model is composing
                   its answer, rather than at a structural seam.

Together these separate two explanations of the positional gap: training sampled
positions uniformly, so any structural break is off-distribution (then web_block
and reasoning_sent both drop), versus block boundaries being special because the
state there is forward-looking (then only reasoning_block drops).

Three sources, not two, because `ar_fidelity.py` scored 0.25-0.31 at block
boundaries while a first pass at random positions in the same traces scored
~0.50. If that holds, domain and position are separate gaps and only one of them
is fixed by a reasoning RL corpus.

If web comes back near 0.63 and reasoning near 0.3, the gap is domain and a
reasoning RL corpus should close it. If web also comes back near 0.3, the fault
is in the measurement (positions, normalization, this code) and retraining fixes
nothing.

Cosine is the headline number, not FVE: FVE divides by a per-source baseline, so
two sources with different spreads are not directly comparable, while
cos(AR(e), h) is baseline-free and is what the direction-only loss optimizes.

    uv run python scripts/fve_by_domain.py --docs 120
"""

from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steer_demo import load_ar, reconstruct  # noqa: E402

from reasoning_attention.config import CORPUS_ID, MODEL_ID, NLAConfig  # noqa: E402
from reasoning_attention.loops import gold_span  # noqa: E402
from reasoning_attention.nla.arch import inner_transformer  # noqa: E402
from reasoning_attention.nla.injection import normalize_activation  # noqa: E402
from reasoning_attention.tokenview import _chat_header  # noqa: E402
from reasoning_attention.training.data import predict_mean_baselines  # noqa: E402

# Matches DataGenConfig: 5 positions per document, 4096-token context.
POSITIONS_PER_DOC = 5
MAX_CONTEXT = 4096
# Skip the first tokens: an activation 3 tokens into a document has almost no
# context to describe, which would depress both sources equally but adds noise.
MIN_POSITION = 32
# Sentinel distinguishing the gold-sentence sampler from a delimiter pair.
GOLD_MODE = ("__gold__", None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--docs", type=int, default=120, help="documents per source")
    p.add_argument("--traces", type=Path, default=Path("data/correct_sample.jsonl"))
    p.add_argument("--base", default=MODEL_ID)
    p.add_argument("--av", type=Path, default=Path("checkpoints/av_rl_ep1"))
    p.add_argument("--ar", type=Path, default=Path("checkpoints/ar_rl_ep1"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sources", nargs="+", default=None, help="subset of cells to run")
    p.add_argument("--max-new-tokens", type=int, default=256)
    return p.parse_args()


def web_texts(n: int) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset(CORPUS_ID, name="default", split="en", streaming=True)
    out = []
    for row in ds:
        text = str(row["content"])
        if len(text) > 600:
            out.append(text)
        if len(out) >= n:
            break
    return [(t, "") for t in out]


def reasoning_texts(path: Path, tokenizer: Any, n: int) -> list[tuple[str, str]]:
    """(full templated text, gold answer) per trace."""
    rows = [json.loads(line) for line in path.open()][:n]
    return [(_chat_header(tokenizer, r["question"]) + r["response"], str(r["gold"])) for r in rows]


def gold_sentence_positions(text: str, gold: str, tokenizer: Any, enc: Any) -> list[int]:
    """Token indices inside the sentence that first states the gold answer.

    Restricted to positions BEFORE the answer's own tokens, so the state is the
    model composing the answer rather than having just written it. Mid-sentence
    by construction: the sentence's own final token is excluded.
    """
    span = gold_span(text, gold)
    if span is None:
        return []
    start = max(
        text.rfind(". ", 0, span[0]),
        text.rfind("\n", 0, span[0]),
        text.rfind("? ", 0, span[0]),
    )
    start = 0 if start < 0 else start + 2
    if span[0] - start < 8:
        return []
    offsets = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_CONTEXT,
    )["offset_mapping"]
    n_extra = int(enc["input_ids"].shape[1]) - len(offsets)
    return [
        i + n_extra
        for i, (a, _b) in enumerate(offsets)
        if start <= a < span[0] and i + n_extra > MIN_POSITION
    ]


def break_positions(
    text: str, tokenizer: Any, enc: Any, delim: str, exclude: str | None = None
) -> list[int]:
    """Token indices of the last token before each occurrence of `delim`.

    `exclude` drops matches also followed by that string, so two break
    conditions can be kept to disjoint position sets rather than overlapping.
    The delimiter is per source because the corpora differ: Ultra-FineWeb has no
    `\n\n` at all (0 of 60 sampled documents) — its paragraphs are single
    newlines — while the reasoning traces use blank lines.
    """
    offsets = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=True,
        max_length=MAX_CONTEXT,
    )["offset_mapping"]
    n_extra = int(enc["input_ids"].shape[1]) - len(offsets)
    out = []
    pos = text.find(delim)
    while pos != -1:
        if not (exclude and text.startswith(exclude, pos)):
            idx = next((i for i, (_a, b) in enumerate(offsets) if b > pos - 1), None)
            if idx is not None and idx > MIN_POSITION:
                out.append(idx + n_extra)
        pos = text.find(delim, pos + len(delim))
    return out


@torch.no_grad()
def collect(
    model: Any,
    tokenizer: Any,
    texts: list[str],
    layer: int,
    rng: random.Random,
    mode: tuple[str, str | None] | None = None,
):
    """h_l at `POSITIONS_PER_DOC` positions per text, one forward pass per text."""
    grab: dict[str, torch.Tensor] = {}
    trunk = inner_transformer(model)
    handle = trunk.layers[layer].register_forward_hook(
        lambda m, i, o: grab.__setitem__("h", (o[0] if isinstance(o, tuple) else o)[0].detach())
    )
    out: list[tuple[str, torch.Tensor]] = []
    for text, gold in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_CONTEXT).to(
            "cuda"
        )
        n_tok = int(enc["input_ids"].shape[1])
        if n_tok <= MIN_POSITION + 1:
            continue
        trunk(**enc)
        states = grab["h"]
        if mode is not None:
            if mode == GOLD_MODE:
                cands = [
                    p for p in gold_sentence_positions(text, gold, tokenizer, enc) if p < n_tok
                ]
            else:
                delim, exclude = mode
                cands = [
                    p for p in break_positions(text, tokenizer, enc, delim, exclude) if p < n_tok
                ]
            if not cands:
                continue
            picks = rng.sample(cands, k=min(POSITIONS_PER_DOC, len(cands)))
        else:
            picks = rng.sample(
                range(MIN_POSITION, n_tok), k=min(POSITIONS_PER_DOC, n_tok - MIN_POSITION)
            )
        for pos in picks:
            # The snippet the activation summarizes is everything up to and
            # including that token — what the AV is meant to describe.
            out.append(
                (tokenizer.decode(enc["input_ids"][0, : pos + 1]), states[pos].float().cpu())
            )
        del states
        torch.cuda.empty_cache()
    handle.remove()
    return out


def main() -> None:
    args = parse_args()
    cfg = NLAConfig()
    scale = cfg.resolve_injection_scale(2048)
    rng = random.Random(args.seed)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    print(f"streaming {args.docs} Ultra-FineWeb documents...")
    reasoning = reasoning_texts(args.traces, tokenizer, args.docs)
    web = web_texts(args.docs)
    sources: dict[str, tuple[list[str], tuple[str, str | None] | None]] = {
        "web_rand": (web, None),
        "web_block": (web, ("\n", None)),
        "reasoning_rand": (reasoning, None),
        # ". " that is not a paragraph end, so the two break conditions are
        # disjoint position sets.
        "reasoning_sent": (reasoning, (". ", None)),
        "reasoning_block": (reasoning, ("\n\n", None)),
        "reasoning_gold": (reasoning, GOLD_MODE),
    }
    if args.sources:
        keep = set(args.sources)
        sources = {k: v for k, v in sources.items() if k in keep}
    for k, (v, _mode) in sources.items():
        print(f"  {k}: {len(v)} texts, median {int(st.median([len(t) for t in v]))} chars")

    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    collected = {
        k: collect(model, tokenizer, v, cfg.extraction_layer, rng, mode=mode)
        for k, (v, mode) in sources.items()
    }
    for k, v in collected.items():
        print(f"  {k}: {len(v)} activations")
    del model
    torch.cuda.empty_cache()

    from reasoning_attention.nla.model import NLA

    nla = NLA.av_only(str(args.av))
    nla.av.eval()
    explained: dict[str, list[tuple[str, torch.Tensor]]] = {}
    for k, items in collected.items():
        rows = []
        for i, (_snippet, h) in enumerate(items, 1):
            e = nla.verbalize(h, max_new_tokens=args.max_new_tokens, return_explanation=True)
            rows.append((e.strip(), h))
            if i % 100 == 0:
                print(f"  verbalized {k} [{i}/{len(items)}]")
        explained[k] = rows
    del nla
    torch.cuda.empty_cache()

    ar_tok, ar_backbone, affine = load_ar(args.ar)
    print(f"\n{'source':<12}{'n':>6}{'ar_cos':>10}{'ar_fve':>10}{'‖h‖ mean':>11}{'baseline':>11}")
    results = {}
    for k, rows in explained.items():
        golds = np.stack([h.numpy() for _e, h in rows])
        base = predict_mean_baselines(golds, scale)
        cos, fve = [], []
        for e, h in rows:
            with torch.no_grad():
                pred = reconstruct(ar_tok, ar_backbone, affine, e).cpu()
            gn = normalize_activation(h[None], scale)
            pn = normalize_activation(pred[None], scale)
            cos.append(float(torch.nn.functional.cosine_similarity(pred[None], h[None])))
            fve.append(1 - float(((pn - gn) ** 2).mean()) / base.meannorm)
        results[k] = (cos, fve)
        print(
            f"{k:<12}{len(rows):>6}{st.mean(cos):>10.3f}{st.mean(fve):>10.3f}"
            f"{float(np.linalg.norm(golds, axis=1).mean()):>11.1f}{base.meannorm:>11.1f}"
        )

    import math

    print()
    pairs = [
        ("web_rand", "reasoning_rand", "domain: web -> reasoning, random positions"),
        ("web_rand", "web_block", "break position, within web"),
        ("reasoning_rand", "reasoning_sent", "sentence end, within reasoning"),
        ("reasoning_rand", "reasoning_block", "block boundary, within reasoning"),
        ("reasoning_sent", "reasoning_block", "sentence end -> block boundary"),
        ("web_rand", "reasoning_block", "combined: training dist -> what we probe"),
        ("reasoning_rand", "reasoning_gold", "gold sentence, within reasoning"),
        ("reasoning_block", "reasoning_gold", "block boundary -> gold sentence"),
    ]
    for left, right, label in pairs:
        if left not in results or right not in results:
            continue
        a, b = results[left][0], results[right][0]
        se = math.sqrt(st.variance(a) / len(a) + st.variance(b) / len(b))
        t = (st.mean(a) - st.mean(b)) / se
        print(
            f"{label:<46} Δcos {st.mean(a) - st.mean(b):+.3f}  "
            f"t {t:+5.1f}  p {math.erfc(abs(t) / math.sqrt(2)):.2g}"
        )


if __name__ == "__main__":
    main()
