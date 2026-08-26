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

# The reference's prompt, kept for provenance and for reproducing their setup.
# It invites the surface-form leak in two places — "Feel free to include specific
# textual examples inline" and the final-feature instruction — and measured on our
# labels 97.4% of explanations quoted something, 82% of those quotes appeared
# verbatim in the context, and 69.8% fell in the last six words, i.e. exactly where
# h_l is read. See EXPLAIN_INSTRUCTION for the replacement and D31.
EXPLAIN_INSTRUCTION_V1 = """A language model needs to predict what text comes next after a snippet which will be presented to you shortly. Identify the 2-3 most important features it would use for this prediction.
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

# v2 — describe, never quote. Same task, same format, same length budget; the only
# change is that naming the text's own tokens is forbidden, so the AR has to
# reconstruct h_l from the described *role* of the ending rather than its identity.
EXPLAIN_INSTRUCTION = """A language model needs to predict what text comes next after a snippet which will be presented to you shortly. Identify the 2-3 most important features it would use for this prediction.
Focus on what the language model must be "thinking about" at the point where the provided text ends. You should not need to reference the fact that the text is truncated/incomplete/a prefix: the language model is causal, so only sees the prefix to what it predicts and this is implicit.
Order features by what is most important for predicting the next tokens. Each feature should consist of a concise ~10-20 word description.

CRITICAL — describe, do not quote. Do NOT reproduce any word, phrase, name, number, or punctuation mark that appears in the text, and do not use quotation marks at all. Refer to elements by their role or category (the subject noun, the opening clause, a proper name of an institution, a numeric quantity), never by their literal wording. An explanation that repeats the text's own words is worthless for this task.

Feature types to consider (as inspiration, not a rigid checklist):
- Syntactic/structural constraints: an unclosed bracket demanding its match
- Immediate semantic expectations: a list promising three items but supplying only two
- Stylistic/register patterns: a formal academic tone maintained throughout
- Narrative/argumentative momentum: a thesis stated, its supporting evidence now due
- Domain/genre signals: a medical case history following a standard reporting format
- Repetition/continuation patterns: a phrase structure recurring with variation

The final feature must characterise the very end of the sequence — the grammatical role that element plays, the construction it belongs to, and what kind of continuation it demands — again WITHOUT naming or quoting it.

Format — IMPORTANT: keep to ~80-100 words total and ALWAYS close the tag:
<analysis>
[first feature]
[second feature]
[final feature: the role of the ending, its construction, what must follow]
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


# --- surface-form leak detection -------------------------------------------
# A prompt instruction is not proof of compliance, so measure it. The AR reads
# h_l at the context's FINAL token, so an explanation that reproduces the tail
# verbatim hands it the answer: masking these spans cost 74% of the AR's FVE
# (0.542 -> 0.141 at step 130), which is the size of the shortcut.
_QUOTE_SPAN_RE = re.compile(r"[\"\u201c\u201d]([^\"\u201c\u201d]{1,80}?)[\"\u201c\u201d]")
LEAK_NGRAM = 5


def _shingles(text: str, n: int) -> set[str]:
    words = text.lower().split()
    return {" ".join(words[i : i + n]) for i in range(len(words) - n + 1)}


def verbatim_overlap(context: str, explanation: str, n: int = LEAK_NGRAM) -> bool:
    """True if `explanation` reproduces text from `context`.

    Two signals: a quoted span that occurs in the context, or any shared n-word
    shingle. n=5 is long enough that innocent function-word runs ("at the end of
    the") rarely collide, and short enough to catch a copied phrase.
    """
    for quoted in _QUOTE_SPAN_RE.findall(explanation):
        if quoted.strip() and quoted.strip().lower() in context.lower():
            return True
    return bool(_shingles(explanation, n) & _shingles(context, n))
