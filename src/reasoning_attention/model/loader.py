"""Initialize Qwen3-1.7B with transformers, in thinking mode.

This is the plain-transformers path (useful for inspection / single-shot
generation / attention work). For high-throughput serving use the vLLM section
in `reasoning_attention.serving`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reasoning_attention.config import ModelConfig


@dataclass
class LoadedModel:
    """A loaded model + its tokenizer."""

    model: Any  # transformers PreTrainedModel
    tokenizer: Any  # transformers PreTrainedTokenizerBase
    config: ModelConfig


def load_model(config: ModelConfig | None = None) -> LoadedModel:
    """Load the tokenizer and model.

    Uses bf16 and device_map="auto" so it lands on the GPU when available.
    """
    config = config or ModelConfig()
    source = config.local_dir or config.model_id

    tokenizer = AutoTokenizer.from_pretrained(source, revision=config.revision)
    model = AutoModelForCausalLM.from_pretrained(
        source,
        revision=config.revision,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    return LoadedModel(model=model, tokenizer=tokenizer, config=config)


def build_thinking_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    enable_thinking: bool = True,
) -> str:
    """Render chat `messages` into a prompt string with Qwen3 thinking mode.

    Qwen3 toggles its reasoning block via the chat template's `enable_thinking`
    flag. With it on, the model emits a `<think>...</think>` section before the
    final answer.
    """
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
