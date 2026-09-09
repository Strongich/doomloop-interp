"""Per-token layer-20 states for one trace, with anchors and a similarity null.

Backs the token explorer: click a token, get its `h_l`, verbalize it, and see how
close it is to the states that matter — the one just before the model first turns
on itself, the one where the gold answer first appears, and the last one.

The whole trace's states come from ONE forward pass (the layer-20 hook sees the
full `[1, S, d]` tensor), so stepping through tokens costs nothing after load.

**The null matters more than the anchors.** Within-trace cosine between layer-20
states is ~0.88 for *arbitrary* pairs, so "the pre-doubt state is similar to the
answer state" is not a finding by itself — it is what any two states in one trace
look like. `baseline()` samples random pairs at a matched separation so a
similarity can be read as above or below chance rather than as a bare number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .loops import doubt_spans, gold_span
from .nla.arch import inner_transformer

# Random pairs drawn for the null. Enough to pin the mean to ~0.001.
BASELINE_PAIRS = 2000


def _chat_header(tokenizer: Any, question: str) -> str:
    """The prompt as the model saw it during generation, up to the assistant turn."""
    return str(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
    )


@dataclass
class Anchor:
    name: str
    token_index: int
    label: str


@dataclass
class TraceView:
    """Tokenized trace plus per-token labels, anchors and cached states."""

    question: str
    response: str
    gold: str
    ids: list[int]
    offsets: list[tuple[int, int]]
    labels: list[str]
    anchors: list[Anchor]
    n_header: int
    states: torch.Tensor | None = field(default=None, repr=False)

    def piece(self, index: int) -> str:
        start, end = self.offsets[index]
        return self.response[start:end]

    def tokens_for_display(self, start: int, stop: int) -> list[tuple[str, str | None]]:
        """(text, label) pairs for gr.HighlightedText over a window."""
        return [
            (self.piece(i), None if self.labels[i] == "plain" else self.labels[i])
            for i in range(start, min(stop, len(self.ids)))
        ]


def build_view(tokenizer: Any, question: str, response: str, gold: str) -> TraceView:
    enc = tokenizer(response, add_special_tokens=False, return_offsets_mapping=True)
    offsets = [(int(a), int(b)) for a, b in enc["offset_mapping"]]
    labels = ["plain"] * len(offsets)

    doubts = doubt_spans(response)
    for lo, hi in doubts:
        for i, (a, b) in enumerate(offsets):
            if a < hi and b > lo:
                labels[i] = "doubt"

    anchors: list[Anchor] = []
    if doubts:
        first_doubt_tok = next((i for i, (a, b) in enumerate(offsets) if b > doubts[0][0]), None)
        if first_doubt_tok:
            # The state BEFORE the marker: the one whose next-token choice was
            # to start doubting.
            anchors.append(Anchor("pre_doubt", first_doubt_tok - 1, "pre-doubt"))
        if len(doubts) > 2:
            after = next((i for i, (a, b) in enumerate(offsets) if b > doubts[2][1]), None)
            if after:
                anchors.append(Anchor("after_doubts", after, "after 3 doubts"))

    span = gold_span(response, gold)
    if span:
        for i, (a, b) in enumerate(offsets):
            if a < span[1] and b > span[0]:
                labels[i] = "gold"
        gold_tok = next((i for i, (a, b) in enumerate(offsets) if b > span[0]), None)
        if gold_tok is not None:
            anchors.append(Anchor("gold_first", gold_tok, "gold first stated"))

    anchors.append(Anchor("end", len(offsets) - 1, "last token"))

    header = _chat_header(tokenizer, question)
    n_header = len(tokenizer(header, add_special_tokens=False)["input_ids"])
    return TraceView(
        question=question,
        response=response,
        gold=gold,
        ids=list(enc["input_ids"]),
        offsets=offsets,
        labels=labels,
        anchors=sorted(anchors, key=lambda a: a.token_index),
        n_header=n_header,
    )


class StateCache:
    """Loads the target model once and caches one trace's states at a time."""

    def __init__(self, base: str, layer: int, device: str = "cuda") -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map=device)
        model.eval()
        trunk: Any = inner_transformer(model)
        self.trunk = trunk
        self._grab: dict[str, torch.Tensor] = {}
        self.trunk.layers[layer].register_forward_hook(self._hook)

    def _hook(self, _m: Any, _i: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        self._grab["h"] = hidden[0].detach()

    @torch.no_grad()
    def fill(self, view: TraceView) -> TraceView:
        if view.states is not None:
            return view
        header = _chat_header(self.tokenizer, view.question)
        enc = self.tokenizer(header + view.response, return_tensors="pt").to(
            self.trunk.device if hasattr(self.trunk, "device") else "cuda"
        )
        self.trunk(**enc)
        states = self._grab["h"][view.n_header :]
        # Keep on GPU in bf16: 32k x 2048 is 134 MB, and every click needs a
        # cosine against it.
        view.states = states[: len(view.ids)]
        return view


def cosine(states: torch.Tensor, i: int, j: int) -> float:
    a = states[i].float()
    b = states[j].float()
    return float(torch.nn.functional.cosine_similarity(a[None], b[None]).item())


def baseline(
    states: torch.Tensor, separation: int, pairs: int = BASELINE_PAIRS
) -> tuple[float, float]:
    """Mean and sd of cosine between random pairs ~`separation` tokens apart.

    Matched on separation because similarity decays with distance: comparing a
    500-token-apart anchor pair against a null of arbitrary pairs would show a
    difference that is only about distance.
    """
    n = states.shape[0]
    sep = max(1, min(separation, n - 1))
    lo = torch.randint(0, n - sep, (min(pairs, max(n - sep, 1)),), device=states.device)
    hi = lo + sep
    unit = torch.nn.functional.normalize(states.float(), dim=-1)
    sims = (unit[lo] * unit[hi]).sum(-1)
    return float(sims.mean()), float(sims.std())
