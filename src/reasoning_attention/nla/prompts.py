"""AV and AR prompt templates + the `<INJECT>` placeholder convention.

Recreated from the reference repo (natural_language_autoencoders):
  - AV/actor template + the `<concept>...</concept>` injection slot:
        nla/datagen/stage3_build.py:_DEFAULT_ACTOR_TEMPLATE
  - AR/critic template (suffix-anchored, no marker):
        nla/datagen/stage3_build.py:_DEFAULT_CRITIC_TEMPLATE
  - `<INJECT>` placeholder + explanation wrapping:
        nla/schema.py (INJECT_PLACEHOLDER, wrap/extract_explanation)

Flow: the AV template carries one placeholder slot. In the repo this is the
literal `<INJECT>` token in storage, swapped at load time for the real
single-token injection char (their ㊗); the model's embedding at that position
is then overwritten with the activation vector. We use our repurposed
placeholder token (`<|image_pad|>`, see NLAConfig.placeholder_token) directly.
"""

from __future__ import annotations

import re

# Literal placeholder string the AV template carries (repo convention). At
# render time it is replaced by the real single-token placeholder.
INJECT_PLACEHOLDER = "<INJECT>"

EXPLANATION_OPEN = "<explanation>"
EXPLANATION_CLOSE = "</explanation>"
_EXPLANATION_RE = re.compile(
    f"{re.escape(EXPLANATION_OPEN)}(.*?){re.escape(EXPLANATION_CLOSE)}",
    re.DOTALL,
)

# AV / actor template — verbatim from the repo, with the `{injection_char}` slot
# renamed `{placeholder}`. The injected activation lands at the single token
# inside <concept>...</concept>.
AV_TEMPLATE = """You are a meticulous AI researcher conducting an important investigation into activation vectors from a language model. Your overall task is to describe the semantic content of that activation vector.

We will pass the vector enclosed in <concept> tags into your context. You must then produce an explanation for the vector, enclosed within <explanation> tags. The explanation consists of 2-3 text snippets describing that vector.

Here is the vector:

<concept>{placeholder}</concept>

Please provide an explanation."""

# AR / critic template — verbatim from the repo. Ends with the fixed suffix
# `</text> <summary>`; the activation is read at the last-token position (no
# marker, suffix-anchored).
AR_TEMPLATE = "Summary of the following text: <text>{explanation}</text> <summary>"


def build_av_content(placeholder: str = INJECT_PLACEHOLDER) -> str:
    """Fill the AV template with the placeholder token (or the `<INJECT>` literal)."""
    return AV_TEMPLATE.format(placeholder=placeholder)


def build_av_messages(placeholder: str = INJECT_PLACEHOLDER) -> list[dict[str, str]]:
    """AV chat messages: a single user turn carrying the injection placeholder."""
    return [{"role": "user", "content": build_av_content(placeholder)}]


def build_ar_prompt(explanation: str) -> str:
    """Fill the AR/critic template with the explanation text.

    Returns the complete formatted string ending in the fixed `</text> <summary>`
    suffix — the activation is reconstructed at its last-token position.
    """
    return AR_TEMPLATE.format(explanation=explanation)


def wrap_explanation(text: str) -> str:
    """Wrap text in <explanation> tags — the AV's expected response format."""
    return f"{EXPLANATION_OPEN}\n{text}\n{EXPLANATION_CLOSE}"


def extract_explanation(response: str) -> str | None:
    """Pull the payload between <explanation> tags; None if absent."""
    match = _EXPLANATION_RE.search(response)
    return match.group(1).strip() if match else None
