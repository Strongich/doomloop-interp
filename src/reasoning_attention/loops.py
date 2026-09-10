"""Locate the token at which a reasoning trace commits to repeating itself.

Ported from Liquid AI's Antidoom (<https://www.liquid.ai/blog/antidoom>,
`Liquid4All/antidoom`, `src/antidoom/{repetition,generate}.py`) at their published
thresholds. Their detector answers exactly the question this study needs and
answers it without a model: *which token is the onset of the loop?*

`metrics.has_doom_loop` scores a whole trace and cannot answer that — it is a
trace-level flag, not a position. Antidoom's is a position, which is what we need
to build a prefix and read `h_l` off its final token.

Two deliberate differences from their implementation:

- **Onset is `start + period`**, the first character of the SECOND occurrence, not
  the first. That is their choice and it is the right one: the model emitting a
  span once is ordinary reasoning; emitting it again is the loop. The residual
  stream we want is the one that *chose to repeat*.
- We resolve char->token with the fast tokenizer's `offset_mapping` instead of
  their hand-rolled BPE string surgery (`Ġ`/`Ċ` substitution, mojibake repair).
  They were reconstructing offsets from an OpenAI-style API's token strings; we
  have the real tokenizer, so offsets are exact and multi-byte UTF-8 split across
  two tokens cannot corrupt a position.

Detection is exact string matching, so a trace that loops *semantically* while
paraphrasing ("Wait, let me check" -> "Hold on, let me verify") scores nothing.
That is a property of the criterion, not a bug in the port.
"""

from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field
from typing import Any, Protocol

# Their published defaults, verbatim from configs/default.yaml.
MIN_REPEATS = 4
MIN_TOTAL_REPEATED = 60
MIN_PERIOD = 1
MAX_PERIOD = 1024
SAMPLE_LEN = 16
SAMPLE_INTERVAL = 128

# Words that, after sentence-ending punctuation, mark the *rhetorical* start of a
# repeated block rather than its lexical start. Their list, verbatim.
RESTART_WORDS = frozenset(
    """actually after also alternatively because but finally first given hmm however
    in let looking maybe now okay perhaps second since so the then therefore this
    thus wait""".split()
)


@dataclass(frozen=True)
class RepeatHit:
    """One detected verbatim loop."""

    start: int
    end: int
    period: int
    repeats: int
    snippet: str

    @property
    def repeat_start(self) -> int:
        """Char offset of the second occurrence — the loop's onset."""
        return self.start + self.period


def _verify_repetition_at(
    text: str,
    start_pos: int,
    period: int,
    min_repeats: int,
    min_total_repeated: int,
) -> RepeatHit | None:
    """Count exact repeats of `text[start_pos:start_pos+period]` around start_pos."""
    if period < 1 or start_pos < 0 or start_pos + period > len(text):
        return None

    pattern = text[start_pos : start_pos + period]
    reps = 0
    pos = start_pos
    while pos + period <= len(text) and text[pos : pos + period] == pattern:
        reps += 1
        pos += period
    end_pos = pos

    # Extend backwards too: the fingerprint scan lands on an arbitrary grid
    # point, which is usually mid-loop rather than at its start.
    pos = start_pos - period
    while pos >= 0 and text[pos : pos + period] == pattern:
        reps += 1
        start_pos = pos
        pos -= period

    if reps >= min_repeats and reps * period >= min_total_repeated:
        snippet = pattern if len(pattern) <= 100 else pattern[:100] + "..."
        return RepeatHit(start_pos, end_pos, period, reps, snippet)
    return None


def find_inner_repetition(
    text: str,
    *,
    min_repeats: int = MIN_REPEATS,
    max_period: int = MAX_PERIOD,
    min_period: int = MIN_PERIOD,
    min_total_repeated: int = MIN_TOTAL_REPEATED,
    sample_len: int = SAMPLE_LEN,
    sample_interval: int = SAMPLE_INTERVAL,
) -> RepeatHit | None:
    """Find a span that repeats >= `min_repeats` times over >= 60 chars.

    Fingerprints every `sample_interval` chars and looks for the same 16 chars
    elsewhere; the gap between occurrences is a candidate period, verified by
    exact expansion. Cheap, and it needs no model.

    Known false negative, theirs and ours: a loop shorter than ~`sample_interval
    + sample_len` chars can fall between two sample points and never be
    fingerprinted. Tightening `sample_interval` costs runtime linearly.
    """
    if not text or len(text) < min_total_repeated:
        return None

    n = len(text)
    for sample_pos in range(0, n - sample_len, sample_interval):
        fingerprint = text[sample_pos : sample_pos + sample_len]

        for other_pos, anchor in (
            (text.find(fingerprint, sample_pos + sample_len), sample_pos),
            (text.rfind(fingerprint, 0, sample_pos), None),
        ):
            if other_pos == -1:
                continue
            start = anchor if anchor is not None else other_pos
            candidate_period = abs(sample_pos - other_pos)
            if not (min_period <= candidate_period <= max_period):
                continue
            hit = _verify_repetition_at(
                text, start, candidate_period, min_repeats, min_total_repeated
            )
            if hit is not None:
                return hit
    return None


class _FastTokenizer(Protocol):
    def __call__(self, text: str, **kwargs: Any) -> Any: ...


@dataclass(frozen=True)
class TokenSpans:
    """Token ids plus their exact character spans in the source text."""

    ids: list[int]
    offsets: list[tuple[int, int]]
    text: str
    # Token start offsets, for bisecting. A trace with 2950 markers in it would
    # otherwise cost 2950 linear scans over ~40k tokens.
    _starts: list[int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_starts", [s for s, _ in self.offsets])

    def char_to_token(self, char_pos: int) -> int | None:
        idx = bisect.bisect_right(self._starts, char_pos) - 1
        if idx < 0:
            return None
        start, end = self.offsets[idx]
        return idx if start <= char_pos < end else None

    def key(self, idx: int) -> str:
        """Comparison key for period matching: the token's text, left-stripped.

        Left-stripping is theirs and it matters: byte-level BPE gives ` let` and
        `let` different ids, so an id-level period comparison misses a loop whose
        two occurrences differ only in leading whitespace.
        """
        start, end = self.offsets[idx]
        return self.text[start:end].lstrip()

    def is_boundary_only(self, idx: int) -> bool:
        start, end = self.offsets[idx]
        piece = self.text[start:end]
        return bool(piece) and not any(ch.isalnum() or ch == "_" for ch in piece)


def tokenize_with_spans(tokenizer: _FastTokenizer, text: str) -> TokenSpans:
    enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    return TokenSpans(
        ids=list(enc["input_ids"]),
        offsets=[(int(a), int(b)) for a, b in enc["offset_mapping"]],
        text=text,
    )


def _slide_period_left(spans: TokenSpans, seed_idx: int, period_tokens: int) -> int:
    """Walk the seed back while the period still holds, to reach the true start."""
    idx = seed_idx
    while idx > 0:
        prev = idx - 1
        if prev + period_tokens >= len(spans.ids):
            break
        if spans.key(prev) != spans.key(prev + period_tokens):
            break
        idx = prev
    return idx


def _skip_boundary_only(
    spans: TokenSpans, idx: int, *, max_tokens: int = 4, max_chars: int = 12
) -> int:
    """Advance past pure punctuation/whitespace so the onset is a readable token."""
    out = idx
    chars = 0
    while out < len(spans.ids) and out - idx < max_tokens:
        start, end = spans.offsets[out]
        chars += end - start
        if chars > max_chars or not spans.is_boundary_only(out):
            break
        out += 1
    return out


def _normalised_word(text: str) -> str:
    return re.sub(r"^[^\w_]+|[^\w_]+$", "", text.strip().lower())


def onset_token_index(spans: TokenSpans, hit: RepeatHit) -> int | None:
    """Token index of the loop's onset — the first readable token of repeat #2.

    Char offsets are not enough on their own: the fingerprint scan is aligned to a
    128-char grid, so `hit.start` is typically mid-loop. Re-deriving the period in
    *token* space and sliding it left recovers the actual boundary.
    """
    seed_idx = spans.char_to_token(hit.start)
    repeat_idx = spans.char_to_token(hit.repeat_start)
    if seed_idx is None or repeat_idx is None or repeat_idx <= seed_idx:
        return _fallback_onset(spans, hit)

    period_tokens = repeat_idx - seed_idx
    seed_idx = _slide_period_left(spans, seed_idx, period_tokens)
    target = seed_idx + period_tokens
    if target >= len(spans.ids):
        return None
    target = _skip_boundary_only(spans, target)
    if target >= len(spans.ids):
        return None

    # A block that repeats as "...answer. Wait, let me" has its lexical period
    # boundary mid-sentence but its rhetorical restart at "Wait". Prefer the
    # restart word, but only when the token before the boundary actually differs
    # between the two occurrences — i.e. the sentence end is genuinely the seam.
    word = _normalised_word(spans.text[slice(*spans.offsets[target])])
    if word and any(ch.isalnum() or ch == "_" for ch in word):
        punct_start = target + 1
        punct_end = _skip_boundary_only(spans, punct_start, max_tokens=3, max_chars=8)
        punct_text = "".join(
            spans.text[slice(*spans.offsets[i])] for i in range(punct_start, punct_end)
        )
        next_word = (
            _normalised_word(spans.text[slice(*spans.offsets[punct_end])])
            if punct_end < len(spans.ids)
            else ""
        )
        seed_prev, repeat_prev = seed_idx - 1, seed_idx + period_tokens - 1
        if (
            punct_end > punct_start
            and any(ch in punct_text for ch in ".!?")
            and seed_prev >= 0
            and repeat_prev < len(spans.ids)
            and spans.ids[seed_prev] != spans.ids[repeat_prev]
            and next_word in RESTART_WORDS
        ):
            target = punct_end

    while target < len(spans.ids):
        if spans.text[slice(*spans.offsets[target])].strip():
            return target
        target += 1
    return None


def _fallback_onset(spans: TokenSpans, hit: RepeatHit) -> int | None:
    """Char-space onset, used when the token-space period cannot be derived."""
    pos = hit.repeat_start
    while pos < hit.end and spans.text[pos].isspace():
        pos += 1
    if pos >= hit.end:
        return None
    idx = spans.char_to_token(pos)
    while idx is not None and idx < len(spans.ids):
        if spans.text[slice(*spans.offsets[idx])].strip():
            return idx
        idx += 1
    return None


# The self-doubt marker the study reads at. Case-sensitive by default: the model
# writes it capitalized when it interrupts itself, and lowercase "wait" is
# usually ordinary prose ("wait for", "waiting"), not a restart.
DEFAULT_MARKER = "Wait"


def marker_matches(
    text: str,
    marker: str = DEFAULT_MARKER,
    *,
    case_sensitive: bool = True,
) -> list[int]:
    """Character offsets where `marker` occurs as a whole word.

    Separate from `find_marker_onsets` because it needs no tokenizer: scanning
    the corpus for candidates is then cheap, and only the traces that survive
    get tokenized.
    """
    flags = 0 if case_sensitive else re.IGNORECASE
    return [m.start() for m in re.finditer(rf"\b{re.escape(marker)}\b", text, flags)]


def find_marker_onsets(
    spans: TokenSpans,
    marker: str = DEFAULT_MARKER,
    *,
    case_sensitive: bool = True,
) -> list[int]:
    """Token indices at which `marker` begins, in order.

    The returned index is the token that *starts* the marker, so the context
    ending just before it is the one whose next-token distribution chose to
    emit the marker. With byte-level BPE that token usually carries the leading
    space (" Wait"), and cutting at its start — not at the "W" — is what keeps
    the prefix a clean token boundary.

    Unlike `find_inner_repetition`, this is phrase matching, and per D33 a doubt
    marker is ordinary reasoning behaviour that appears in successful traces at
    similar density. It locates *positions to read*; it must not be used to
    label a trace as derailed. The label comes from the outcome.
    """
    out = []
    for start in marker_matches(spans.text, marker, case_sensitive=case_sensitive):
        idx = spans.char_to_token(start)
        if idx is not None:
            out.append(idx)
    return out


# Phrases that mark the model turning on itself. Used ONLY to locate positions:
# per D33 a doubt marker is ordinary reasoning behaviour, present in 99.9% of
# this corpus's traces, so its presence is never evidence that a trace derailed.
DOUBT_MARKERS = (
    "Wait",
    "Hmm",
    "Hold on",
    "Actually",
    "Alternatively",
    "But",
    "Maybe",
    "However",
    "Let me reconsider",
    "Let me double-check",
    "Let me verify",
    "That doesn't seem right",
    "That can't be right",
)
# Everything above except the one word, for testing whether a result about
# "Wait" is about self-doubt or about that specific token.
DOUBT_MARKERS_NON_WAIT = tuple(m for m in DOUBT_MARKERS if m.lower() != "wait")


def doubt_spans(text: str, markers: tuple[str, ...] = DOUBT_MARKERS) -> list[tuple[int, int]]:
    """Character spans of every self-doubt marker, merged and in order."""
    hits: list[tuple[int, int]] = []
    for marker in markers:
        for start in marker_matches(text, marker):
            hits.append((start, start + len(marker)))
    hits.sort()
    merged: list[tuple[int, int]] = []
    for start, end in hits:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def gold_span(text: str, gold: str) -> tuple[int, int] | None:
    """First occurrence of the gold answer as a standalone number/word.

    Word-bounded so gold "7" does not match the 7 in "70" — which would put the
    anchor hundreds of tokens too early and silently invalidate every similarity
    measured against it.

    The right-hand guard rejects a following "." only when a digit follows it, so
    "12.5" does not match gold 12 but a sentence-final "12." does. Rejecting "."
    outright looks equivalent and is not: it silently skips every occurrence that
    ends a sentence, which is most of them, and moves the anchor hundreds of
    tokens later or loses it entirely.
    """
    gold = gold.strip()
    if not gold:
        return None
    match = re.search(rf"(?<![\w.]){re.escape(gold)}(?!\w)(?!\.\d)", text)
    return (match.start(), match.end()) if match else None
