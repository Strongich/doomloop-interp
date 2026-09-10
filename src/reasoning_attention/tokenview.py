"""Per-token layer-20 states for one trace, with anchors and a similarity null.

Backs the token explorer: click a token, get its `h_l`, verbalize it, and see how
close it is to the states that matter — the one just before the model first turns
on itself, the one where the gold answer first appears, and the last one.

The whole trace's states come from ONE forward pass (the layer-20 hook sees the
full `[1, S, d]` tensor), so stepping through tokens costs nothing after load.

**The null matters more than the anchors.** Measured within one trace, arbitrary
layer-20 pairs sit at ~0.47 cosine and pairs a few tokens apart at ~0.54, so a
bare "0.5 between the pre-doubt and answer states" is not a finding — it is what
any two states in that trace look like. `baseline()` samples random pairs at a
matched separation, drawn from inside `<think>`, so a similarity reads as above
or below chance rather than as an uninterpretable number.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from .loops import doubt_spans, gold_span
from .nla.arch import inner_transformer

# Random pairs drawn for the null. Enough to pin the mean to ~0.001.
BASELINE_PAIRS = 2000


def _think_ranges(
    response: str, offsets: list[tuple[int, int]]
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Token span and char span of the <think> block.

    A derailed trace never emits `</think>` — it ran out of budget mid-thought —
    so an unterminated block runs to the end of the response. Treating a missing
    close tag as an empty block would leave exactly the traces this study is
    about with no null at all.
    """
    open_at = response.find("<think>")
    start_char = open_at + len("<think>") if open_at >= 0 else 0
    close_at = response.find("</think>", start_char)
    end_char = close_at if close_at >= 0 else len(response)

    def tok_at(char: int, default: int) -> int:
        for i, (_a, b) in enumerate(offsets):
            if b > char:
                return i
        return default

    start = tok_at(start_char, 0)
    end = tok_at(end_char, len(offsets)) if close_at >= 0 else len(offsets)
    return (start, max(end, start + 1)), (start_char, end_char)


def block_boundaries(
    text: str,
    region: tuple[int, int] | None = None,
    markers: Sequence[str] = ("wait",),
) -> list[tuple[int, int, str | None]]:
    """Char offsets of block ends that are followed by a block of self-doubt.

    The model paragraphs its reasoning on blank lines, so the last token before a
    `\n\n` is where it has finished a step and is choosing what to do next. Those
    boundaries where the *next* block opens with a doubt marker are the decision
    points this study wants: the model had a completed step in hand and chose to
    second-guess it rather than conclude.

    Returns (last_char_of_block, first_char_of_next_block, matched_marker), the
    marker being None when the next block opens with none of them. Only the
    first sentence of the next block is searched, so a marker deep inside a long
    block does not label a boundary hundreds of tokens earlier.

    Boundaries where the next block does NOT doubt are the control this analysis
    needs: they are the same kind of position (end of a completed reasoning
    block), measured the same way, differing only in what the model chose to do
    next. Comparing doubt boundaries against a null of arbitrary within-think
    pairs instead confounds the result with absolute position, since every
    boundary sits at a block edge and the anchor is pinned to the end of
    thinking.
    """
    lo, hi = region if region else (0, len(text))
    out: list[tuple[int, int, str | None]] = []
    pos = text.find("\n\n", lo)
    while pos != -1 and pos < hi:
        nxt = pos
        while nxt < len(text) and text[nxt] == "\n":
            nxt += 1
        sentence = re.split(r"(?<=[.!?])\s|\n", text[nxt : nxt + 400], maxsplit=1)[0].lower()
        if pos - 1 >= lo:
            hit = next((m for m in markers if m.lower() in sentence), None)
            out.append((pos - 1, nxt, hit))
        pos = text.find("\n\n", pos + 2)
    return out


def think_close_char(text: str) -> int | None:
    """Char offset of `</think>`, or None when the trace never closed it."""
    at = text.find("</think>")
    return at if at >= 0 else None


def think_last_content_char(text: str) -> int | None:
    """Char offset of the last non-whitespace character before `</think>`.

    `</think>` itself is a poor anchor: measured against random reasoning states
    it sits at ~0.25 cosine where those states sit at ~0.48 with each other, so
    `h_l` on the tag is dominated by "emit the closing tag" and lives off the
    reasoning manifold the null is drawn from. The last *content* token is the
    model's final reasoning state and stays on that manifold, so a similarity to
    it is comparable to the null.
    """
    at = text.find("</think>")
    if at < 0:
        return None
    pos = at - 1
    while pos >= 0 and text[pos].isspace():
        pos -= 1
    return pos if pos >= 0 else None


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
    # Token range of the <think> block, the region the null is drawn from.
    think: tuple[int, int] = (0, 0)
    # The same block as character offsets, for text-level scanning.
    think_chars: tuple[int, int] = (0, 0)
    states: torch.Tensor | None = field(default=None, repr=False)

    def char_to_token(self, char_pos: int) -> int | None:
        for i, (start, end) in enumerate(self.offsets):
            if start <= char_pos < end:
                return i
        return None

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

    think, think_chars = _think_ranges(response, offsets)

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
        think=think,
        think_chars=think_chars,
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
    states: torch.Tensor,
    separation: int,
    pairs: int = BASELINE_PAIRS,
    region: tuple[int, int] | None = None,
) -> tuple[float, float]:
    """Mean and sd of cosine between random pairs `separation` tokens apart.

    Matched on separation because similarity decays with distance: comparing a
    500-token-apart anchor pair against a null of arbitrary pairs would show a
    difference that is only about distance.

    `region` confines both endpoints to a token range — the `<think>` block. The
    reasoning inside <think> and the answer written after it have different
    statistics (the post-think text is a short committed restatement), so a null
    spanning both is a mixture, while every anchor we care about lives inside
    <think>. Drawing the null from the region the anchors come from makes z a
    statement about *those positions* rather than about think-vs-answer.
    """
    n = states.shape[0]
    lo_b, hi_b = region if region else (0, n)
    lo_b, hi_b = max(0, lo_b), min(n, hi_b)
    span = hi_b - lo_b
    sep = max(1, min(separation, max(span - 1, 1)))
    count = max(span - sep, 1)
    lo = lo_b + torch.randint(0, count, (min(pairs, count),), device=states.device)
    hi = lo + sep
    unit = torch.nn.functional.normalize(states.float(), dim=-1)
    sims = (unit[lo] * unit[hi]).sum(-1)
    if sims.numel() < 2:
        return float(sims.mean()), float("nan")
    return float(sims.mean()), float(sims.std())
