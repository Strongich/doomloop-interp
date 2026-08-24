"""Text-quality metrics for model outputs.

Currently: n-gram repetition, used to flag degenerate (looping) generations.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

# A generation whose 4-gram repetition exceeds this fraction is treated as
# degenerate (stuck in a loop).
REPETITION_THRESHOLD = 0.15

# "Doom loop" = some long n-gram repeats many times over the token stream.
DOOM_LOOP_N = 30
DOOM_LOOP_MIN_REPEATS = 20


def ngram_repetition_ratio(text: str, n: int = 4) -> float:
    """Fraction of n-grams that are duplicates."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i : i + n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeated = sum(c - 1 for c in counts.values())
    return repeated / len(ngrams)


def is_repetitive(text: str, n: int = 4, threshold: float = REPETITION_THRESHOLD) -> bool:
    """True if the text's n-gram repetition ratio exceeds `threshold` (default 15%)."""
    return ngram_repetition_ratio(text, n=n) > threshold


def has_doom_loop(
    tokens: Sequence[object],
    n: int = DOOM_LOOP_N,
    min_repeats: int = DOOM_LOOP_MIN_REPEATS,
) -> bool:
    """True if some `n`-token n-gram repeats at least `min_repeats` times.

    Operates over a TOKEN sequence (e.g. the model's generated token ids, or
    `tokenizer.tokenize(text)`) — not whitespace words. This catches the
    "doom loop" failure mode where a long span repeats verbatim many times.
    """
    seq = list(tokens)
    if len(seq) < n:
        return False
    counts = Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))
    return max(counts.values()) >= min_repeats


def is_degenerate(
    text: str,
    tokens: Sequence[object] | None = None,
    *,
    ratio_threshold: float = REPETITION_THRESHOLD,
    ratio_n: int = 4,
    doom_n: int = DOOM_LOOP_N,
    doom_min_repeats: int = DOOM_LOOP_MIN_REPEATS,
) -> bool:
    """True only if BOTH degeneracy signals fire:

    1. word-level 4-gram repetition ratio exceeds `ratio_threshold` (15%), AND
    2. the token stream contains a doom loop (a `doom_n`-token n-gram repeated
       at least `doom_min_repeats` times).

    `tokens` should be the model's token sequence; if omitted it falls back to
    whitespace words (`text.split()`), so the doom-loop check is word-based.
    """
    if tokens is None:
        tokens = text.split()
    return is_repetitive(text, n=ratio_n, threshold=ratio_threshold) and has_doom_loop(
        tokens, n=doom_n, min_repeats=doom_min_repeats
    )
