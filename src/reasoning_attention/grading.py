"""Did a reasoning trace reach the right answer, a wrong one, or none at all?

The doom-loop study contrasts traces that **reach a correct `\\boxed{}` answer**
with traces that **produce no answer at all** (budget exhausted inside
`<think>`). Those are three distinct outcomes, not two, and a bare
`verify(gold, pred) -> bool` collapses "no box was ever emitted" into the same
bucket as "boxed the wrong number". `grade()` keeps them apart, because which
bucket a trace lands in *is* the study's independent variable.

Extraction is `math_verify` + `latex2sympy2_extended`, ported from
`UNI/MASTERS/eval_mmk12_filtered.py`. That is symbolic equivalence, not string
comparison, so `\\dfrac{1}{2}`, `\\frac{1}{2}` and `0.5` all verify equal — which
a hand-rolled brace-matcher plus normalization table gets wrong in both
directions. `boxed_match_priority=0` and `try_extract_without_anchor=False`
together mean *only* a `\\boxed{}` counts: the grader never guesses an answer out
of trailing prose, so "no box" stays cleanly distinguishable.

**Gold answers are parsed boxed** (D58). `parse(gold)` on a bare LaTeX gold runs
the unanchored extractor, which on MATH-500 fails outright for 51 of 500 golds
(`\\pi`, `\\text{Evelyn}`, `p - q`) and, worse, silently TRUNCATES 57 more —
`3\\sqrt{13}` parses to `3`, `6-5i` to `6`. Truncation is the dangerous one: it
produces false positives, marking a response of `\\boxed{3}` correct for gold
`3\\sqrt{13}`. Wrapping the gold in `\\boxed{}` and parsing it through the SAME
config as the response makes gold round-trip against itself 500/500 on MATH-500
and 300/300 on GSM8K. GSM8K golds are bare integers that parsed correctly either
way, so this does not move any historical GSM8K number — verified by re-grading
all 3,096 rows of `data/prefix_online_vllm/`.

sympy can hang or blow the stack on adversarial input, hence the alarm. Note
`signal.setitimer` only works on the main thread — call `grade()` from the main
thread, or pass `timeout=None` to skip the guard.
"""

from __future__ import annotations

import signal
from dataclasses import dataclass
from typing import Any

# Seconds to allow one symbolic comparison. Their script used 5s.
VERIFY_TIMEOUT = 5.0


@dataclass(frozen=True)
class Grade:
    """One trace's outcome.

    `has_answer` is False only when no `\\boxed{}` was parseable, so
    `(has_answer, is_correct)` distinguishes the three cases the study needs:
    `(True, True)` recovered, `(True, False)` answered but wrong,
    `(False, False)` never answered.
    """

    has_answer: bool
    is_correct: bool
    # Why grading came out the way it did — "ok", "no_boxed_answer",
    # "gold_unparseable_fell_back_to_string", "timeout", or "error: <type>".
    status: str = "ok"


class _Timeout(Exception):
    pass


def _on_alarm(signum: int, frame: Any) -> None:
    raise _Timeout


def _extraction_config() -> list[Any]:
    """`math_verify` extraction restricted to `\\boxed{}` spans."""
    from latex2sympy2_extended import NormalizationConfig
    from math_verify import LatexExtractionConfig

    return [
        LatexExtractionConfig(
            normalization_config=NormalizationConfig(
                nits=False,
                malformed_operators=False,
                basic_latex=True,
                boxed="all",
                units=True,
            ),
            # 0 = highest: prefer a boxed span over anything else.
            boxed_match_priority=0,
            # Never infer an answer from unanchored trailing text. A trace that
            # rambled to a number without boxing it must count as unanswered.
            try_extract_without_anchor=False,
        )
    ]


def _parse_gold(gold: str) -> list[Any]:
    """Parse the gold answer through the same boxed extractor as the response.

    Order matters. The boxed form is tried FIRST because the unanchored parser
    silently truncates symbolic golds rather than declining them, so asking it
    first would hand back a confidently wrong answer for `3\\sqrt{13}`. The raw
    parse is kept as a fallback for a gold that is already a full LaTeX
    expression the boxed wrapper would break.
    """
    from math_verify import parse

    for candidate in ("\\boxed{" + gold + "}", gold):
        try:
            parsed = parse(
                candidate, extraction_config=_extraction_config(), extraction_mode="first_match"
            )
        except Exception:
            parsed = []
        if parsed:
            return list(parsed)
    return []


def _grade_inner(response: str, gold: str) -> Grade:
    from math_verify import parse, verify

    gold_parsed = _parse_gold(gold)

    if not gold_parsed:
        # GSM8K golds are bare integers, which `parse` sometimes declines. Fall
        # back to string equality rather than silently marking everything wrong.
        ok = response.strip().lower() == gold.strip().lower()
        return Grade(has_answer=ok, is_correct=ok, status="gold_unparseable_fell_back_to_string")

    answer_parsed = parse(
        response, extraction_config=_extraction_config(), extraction_mode="first_match"
    )
    if not answer_parsed:
        return Grade(has_answer=False, is_correct=False, status="no_boxed_answer")
    return Grade(has_answer=True, is_correct=bool(verify(gold_parsed, answer_parsed)))


def grade(response: str, gold: str, timeout: float | None = VERIFY_TIMEOUT) -> Grade:
    """Grade one model response against one gold answer.

    Never raises: a timeout or a sympy explosion grades as unanswered-and-wrong
    with the reason in `status`, so one pathological trace cannot kill a sweep
    over thousands. Check `status` when auditing, not just the booleans.
    """
    if timeout is None:
        try:
            return _grade_inner(response, gold)
        except Exception as exc:
            return Grade(False, False, status=f"error: {type(exc).__name__}")

    old = signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, timeout)
    try:
        return _grade_inner(response, gold)
    except _Timeout:
        return Grade(False, False, status="timeout")
    except Exception as exc:
        return Grade(False, False, status=f"error: {type(exc).__name__}")
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
