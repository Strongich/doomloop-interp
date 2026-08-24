"""Model section: download + (transformers) initialization for Qwen3-1.7B."""

from reasoning_attention.model.download import download_model
from reasoning_attention.model.loader import build_thinking_prompt, load_model

__all__ = ["download_model", "load_model", "build_thinking_prompt"]
