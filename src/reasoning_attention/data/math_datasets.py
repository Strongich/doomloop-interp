"""Preparation functions for the three math eval datasets.

All three are normalized to the SAME schema so they're interchangeable downstream:

    question : str   — the problem statement
    answer   : str   — the gold FINAL answer (for grading; never shown to the model)
    source   : str   — dataset id, e.g. "gsm8k"
    subset   : str   — config/subset name, e.g. "main" / "AIME2025-I" / "default"
    split    : str   — "train" / "test"
    messages : list  — chat messages for the model: a SINGLE user turn (the
                       question). The answer is deliberately NOT added as an
                       assistant turn — we want the model to generate it.

Datasets (structure confirmed via the HF Dataset Viewer):
  - openai/gsm8k        config "main", splits train+test. `answer` is a CoT that
                        ends in "#### <final>"; we keep only the final value.
  - opencompass/AIME2025 configs AIME2025-I and AIME2025-II (all subsets), split
                        test. `answer` is the final integer.
  - math-ai/amc23       config "default", split test. `answer` is the final value.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

# Registry of available prepared datasets: name -> loader.
DATASETS: dict[str, str] = {
    "gsm8k": "openai/gsm8k",
    "aime2025": "opencompass/AIME2025",
    "amc23": "math-ai/amc23",
}


# ─────────────────────────────────────────────────────────────────────────
# Chat formatting — user turn only, answer never included.
# ─────────────────────────────────────────────────────────────────────────
def build_messages(question: str) -> list[dict[str, str]]:
    """Build the chat messages for one problem: a single user turn.

    The gold answer is intentionally excluded — there is no assistant turn — so
    the model is prompted to *generate* the solution, not shown it.
    """
    return [{"role": "user", "content": question.strip()}]


def render_prompt(tokenizer: Any, question: str, enable_thinking: bool = True) -> str:
    """Render `build_messages(question)` into a prompt string via the tokenizer's
    chat template, with Qwen3 thinking mode and a trailing generation prompt.
    """
    return tokenizer.apply_chat_template(
        build_messages(question),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


# ─────────────────────────────────────────────────────────────────────────
# Per-dataset answer normalization.
# ─────────────────────────────────────────────────────────────────────────
def _gsm8k_final_answer(answer: str) -> str:
    """GSM8K gold answers are CoT ending in `#### <final>`; keep the final value."""
    if "####" in answer:
        answer = answer.rsplit("####", 1)[-1]
    return answer.strip().replace(",", "")


def _normalize(
    ds: Dataset,
    *,
    source: str,
    subset: str,
    split: str,
    answer_transform: Callable[[str], str] | None = None,
) -> Dataset:
    """Map an arbitrary dataset onto the common schema."""

    def _row(example: dict[str, Any]) -> dict[str, Any]:
        question = str(example["question"])
        raw_answer = str(example["answer"])
        answer = answer_transform(raw_answer) if answer_transform else raw_answer.strip()
        return {
            "question": question,
            "answer": answer,
            "source": source,
            "subset": subset,
            "split": split,
            "messages": build_messages(question),
        }

    # Drop original columns so every prepared dataset has identical schema.
    return ds.map(_row, remove_columns=ds.column_names)


# ─────────────────────────────────────────────────────────────────────────
# Dataset preparation functions.
# ─────────────────────────────────────────────────────────────────────────
def prepare_gsm8k() -> Dataset:
    """openai/gsm8k, config `main`: train + test combined under one subset."""
    parts = []
    for split in ("train", "test"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        parts.append(
            _normalize(
                ds,
                source="gsm8k",
                subset="main",
                split=split,
                answer_transform=_gsm8k_final_answer,
            )
        )
    return concatenate_datasets(parts)


def prepare_aime2025() -> Dataset:
    """opencompass/AIME2025: all subsets (AIME2025-I and AIME2025-II), test split."""
    parts = []
    for subset in ("AIME2025-I", "AIME2025-II"):
        ds = load_dataset("opencompass/AIME2025", subset, split="test")
        parts.append(_normalize(ds, source="aime2025", subset=subset, split="test"))
    return concatenate_datasets(parts)


def prepare_amc23() -> Dataset:
    """math-ai/amc23: config `default`, single test split."""
    ds = load_dataset("math-ai/amc23", "default", split="test")
    return _normalize(ds, source="amc23", subset="default", split="test")


def load_all() -> dict[str, Dataset]:
    """Prepare all three datasets, keyed by short name (see `DATASETS`)."""
    return {
        "gsm8k": prepare_gsm8k(),
        "aime2025": prepare_aime2025(),
        "amc23": prepare_amc23(),
    }


def load_one(name: str) -> Dataset:
    """Prepare a single dataset by short name."""
    preparers = {
        "gsm8k": prepare_gsm8k,
        "aime2025": prepare_aime2025,
        "amc23": prepare_amc23,
    }
    if name not in preparers:
        raise KeyError(f"unknown dataset {name!r}; choose from {sorted(preparers)}")
    return preparers[name]()
