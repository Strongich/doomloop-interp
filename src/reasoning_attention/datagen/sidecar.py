"""The dataset sidecar — `{parquet}.nla_meta.yaml`.

The sidecar is the contract between datagen and training. Token ids, prompt
templates, the injection scale, `d_model`, the extraction layer: training reads
them from here and asserts them against the live tokenizer at startup instead of
hardcoding. If a future run changes the layer or the placeholder token, the
training side finds out from the sidecar rather than by silently mistraining.

`norm` is always `"none"` out of datagen — vectors are stored raw and
normalization is a training-time decision.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from typing import Any

SIDECAR_SUFFIX = ".nla_meta.yaml"


@dataclass
class ExtractionMeta:
    """What was extracted, from where, and at what layer."""

    base_model: str
    d_model: int
    layer_index: int
    hidden_states_index: int
    corpus: str
    corpus_config: str | None
    corpus_split: str
    corpus_start: int
    n_documents: int
    positions_per_doc: int
    max_context_tokens: int
    min_position: int
    # Always "none" from datagen — normalization happens at injection/loss time.
    norm: str = "none"


@dataclass
class ExplainerMeta:
    """Which model wrote the explanations, and with what prompt."""

    model: str
    reasoning_effort: str
    max_output_tokens: int
    instruction_prompt: str


@dataclass
class TokenMeta:
    """The placeholder token training must inject at, and the AR's tail."""

    placeholder_token: str
    placeholder_token_id: int
    # Expected trailing token ids of the AR prompt. Training verifies the tail
    # matches, then extracts the activation at `tokens[-1]`.
    ar_suffix_ids: list[int] = field(default_factory=list)


@dataclass
class DatasetMeta:
    """Top-level sidecar record."""

    dataset_id: str
    stage: str  # base | av_half | ar_half | av_sft | ar_sft
    row_count: int
    n_documents: int
    extraction: ExtractionMeta
    created_by: str
    tokens: TokenMeta | None = None
    explainer: ExplainerMeta | None = None
    prompt_templates: dict[str, str] = field(default_factory=dict)
    parent_datasets: list[str] = field(default_factory=list)


def _to_plain(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _to_plain(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    return obj


def sidecar_path(parquet_path: str) -> str:
    return f"{parquet_path}{SIDECAR_SUFFIX}"


def write_sidecar(parquet_path: str, meta: DatasetMeta) -> str:
    """Write the sidecar next to its parquet; returns the path.

    Serialized as JSON, which is valid YAML — no yaml dependency, and multi-line
    prompt templates round-trip exactly instead of being reflowed by a dumper.
    """
    path = sidecar_path(parquet_path)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_to_plain(meta), fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    return path


def read_sidecar(parquet_path: str) -> dict[str, Any]:
    """Read a sidecar back as a plain dict."""
    with open(sidecar_path(parquet_path), encoding="utf-8") as fh:
        loaded: dict[str, Any] = json.load(fh)
    return loaded
