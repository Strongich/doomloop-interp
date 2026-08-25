"""The instruction prompt that makes the explainer model write a summary.

Taken from the reference repo's `stage2_api_explain._DEFAULT_INSTRUCTION`, which
is itself adapted from the prompt template in the NLA paper's appendix. Kept
verbatim apart from the tag name: the reference asks for `<analysis>` tags, and
so do we, so the extraction regex is unchanged.

The ~80-100 word budget and the 2-3 feature cap are deliberate: a response cut
off before its closing tag fails extraction and the row is dropped, so the
prompt constrains length rather than relying on the token limit.
"""

from __future__ import annotations

import re

ANALYSIS_OPEN = "<analysis>"
ANALYSIS_CLOSE = "</analysis>"

# Strict: both tags must be present. A truncated response fails this and the row
# is dropped — better than training on half a thought.
ANALYSIS_RE = re.compile(
    f"{re.escape(ANALYSIS_OPEN)}\\s*(.*?)\\s*{re.escape(ANALYSIS_CLOSE)}", re.DOTALL
)

# Minimum features required. The prompt asks for 2-3; fewer than 2 means the
# model ignored the format.
MIN_FEATURES = 2

# A thinking model (Qwen3 in its default mode) emits its chain of thought first,
# then the answer. Only the text after the closing tag is the explanation: the
# reasoning is scratch work, and it routinely quotes the <analysis> format while
# planning, so searching the whole response would extract a draft rather than the
# final answer.
THINK_CLOSE = "</think>"

EXPLAIN_INSTRUCTION = """A language model needs to predict what text comes next after a snippet which will be presented to you shortly. Identify the 2-3 most important features it would use for this prediction.
Focus on what the language model must be "thinking about" at the point where the provided text ends. You should not need to reference the fact that the text is truncated/incomplete/a prefix: the language model is causal, so only sees the prefix to what it predicts and this is implicit.
Order features by what is most important for predicting the next tokens. Each feature should consist of a concise ~10-20 word description. Feel free to include specific textual examples inline.

Feature types to consider (as inspiration, not a rigid checklist):
- Syntactic/structural constraints: "unclosed parenthesis requires matching close"
- Immediate semantic expectations: "list promised three items but only two given"
- Stylistic/register patterns: "formal academic tone maintained throughout"
- Narrative/argumentative momentum: "thesis stated, supporting evidence now expected"
- Domain/genre signals: "medical case history following SOAP format"
- Repetition/continuation patterns: "same phrase structure repeating with variations"

The final feature must describe the very end of the presented sequence: its role, what it's part of, and immediate constraints on what follows.

Format — IMPORTANT: keep to ~80-100 words total and ALWAYS close the tag:
<analysis>
[first feature — include specific examples when relevant]
[second feature]
[final feature: the last token, its role, immediate constraints]
</analysis>

Text to analyze:

<begin_text>{text}<end_text>"""

# API models reach for every list marker there is; we want plain paragraphs
# separated by blank lines.
_LIST_PREFIX_RE = re.compile(
    r"^\s*(?:"
    r"[-*•+–—]"  # bullet chars (incl. en/em dash)
    r"|\d+[.)]"  # 1. 1)
    r"|\(\d+\)"  # (1)
    r"|[a-zA-Z][.)]"  # a. a) A. A)
    r"|\([a-zA-Z]\)"  # (a) (A)
    r"|[ivxIVX]+[.)]"  # i. ii) IV.
    r")\s+"
)
_BOLD_WRAP_RE = re.compile(r"^\*\*(.+?)\*\*\s*")


def build_explain_prompt(text: str) -> str:
    """Fill the instruction template with the snippet to explain."""
    return EXPLAIN_INSTRUCTION.format(text=text)


def extract_and_clean(raw: str) -> str | None:
    """Pull the content inside `<analysis>` tags and strip list formatting.

    Any chain of thought is discarded first: only the text after the final
    `</think>` is considered. Returns paragraphs joined by blank lines, or None
    when the tags are absent (truncated or off-format) — the caller drops those.
    """
    if THINK_CLOSE in raw:
        raw = raw.rsplit(THINK_CLOSE, 1)[1]

    match = ANALYSIS_RE.search(raw)
    if match is None:
        return None

    cleaned: list[str] = []
    for line in match.group(1).split("\n"):
        line = _LIST_PREFIX_RE.sub("", line)
        line = _BOLD_WRAP_RE.sub(r"\1 ", line)  # **Header:** text -> Header: text
        line = line.strip().strip("*_")
        if line:
            cleaned.append(line)
    return "\n\n".join(cleaned)


def count_features(explanation: str) -> int:
    """How many blank-line-separated features the cleaned explanation holds."""
    return explanation.count("\n\n") + 1 if explanation else 0
