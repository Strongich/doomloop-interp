"""NLA (Natural Language Activations) — AV + AR models and helpers."""

from reasoning_attention.nla.injection import inject_at_placeholder, normalize_activation
from reasoning_attention.nla.model import NLA, ARModel, AROutput
from reasoning_attention.nla.prompts import (
    AR_TEMPLATE,
    AV_TEMPLATE,
    INJECT_PLACEHOLDER,
    build_ar_prompt,
    build_av_messages,
    extract_explanation,
    wrap_explanation,
)

__all__ = [
    "NLA",
    "ARModel",
    "AROutput",
    "AV_TEMPLATE",
    "AR_TEMPLATE",
    "INJECT_PLACEHOLDER",
    "build_av_messages",
    "build_ar_prompt",
    "wrap_explanation",
    "extract_explanation",
    "inject_at_placeholder",
    "normalize_activation",
]
