"""Datasets and collators for the AV / AR warm-start parquets.

Both datasets read the stage-3 output of `datagen/` and hold their activation
vectors in one contiguous float32 array rather than as per-row Python lists: at
250k x 2048 the list-of-lists form costs several GB of object overhead for data
that is already a matrix on disk.

**Normalization happens here, not in datagen.** Vectors are stored raw; both
sides scale them: the AV to `NLAConfig.injection_scale` (1000, matching the
activation distribution), the AR to `NLAConfig.mse_scale` (sqrt(d_model) ~ 45.3,
keeping the loss O(1)). The AV
injects the scaled vector, and the AR regresses onto the *same* scaled vector, so
the two models agree on what "the activation" means. Regressing onto raw
~900-norm vectors instead would put the MSE on a wildly different scale from
everything else and make the loss unreadable across runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from reasoning_attention.config import NLAConfig
from reasoning_attention.nla.injection import normalize_activation


def load_activations(path: str, limit: int | None = None) -> tuple[Any, np.ndarray]:
    """Read a stage-3 parquet; returns (table, activations[N, d_model] float32)."""
    table = pq.read_table(path)
    if limit is not None:
        table = table.slice(0, limit)
    flat = table.column("activation_vector").combine_chunks().values.to_numpy()
    # copy=True: the arrow buffer is read-only, and torch.from_numpy on a
    # non-writable array is undefined behaviour rather than an error.
    activations = np.array(flat, dtype=np.float32, copy=True).reshape(table.num_rows, -1)
    return table, activations


@dataclass
class AVBatch:
    """One AV batch: token ids, labels masked to the response, and the vectors."""

    input_ids: torch.Tensor  # [B, T]
    attention_mask: torch.Tensor  # [B, T]
    labels: torch.Tensor  # [B, T], -100 everywhere but the response
    activations: torch.Tensor  # [B, d_model], already scaled


@dataclass
class ARBatch:
    """One AR batch: the summary prompt and the activation to regress onto."""

    input_ids: torch.Tensor  # [B, T]
    attention_mask: torch.Tensor  # [B, T]
    targets: torch.Tensor  # [B, d_model], already scaled
    last_index: torch.Tensor  # [B], position of each row's final real token


def shuffled_activations(activations: np.ndarray, seed: int) -> np.ndarray:
    """Break the text<->activation pairing — the reference's random control.

    Every explanation keeps its text but is paired with some other row's vector,
    so any score above chance must come from something generic (the prior over
    explanation-shaped text, the mean activation direction) rather than from
    actually reading the vector. The reference runs this as `critic_rand` and
    measured 0.922 against a 0.938 baseline, i.e. FVE ~= 0, and as the AV's
    "random baseline" where the real run sat ~0.21 lower in loss.

    The permutation reuses the same multiset of vectors, so `predict_mean_baseline`
    is bit-identical between the real and control runs and the two FVEs are
    directly comparable. That is the whole point of the control: only the pairing
    changes.
    """
    rng = np.random.default_rng(seed)
    return activations[rng.permutation(len(activations))]


class AVDataset(Dataset[dict[str, Any]]):
    """`h -> s`: prompt (with the placeholder) + response, activation attached.

    The stored prompt carries the literal `<INJECT>`; it is swapped for the real
    placeholder token here, so a dataset built before the placeholder was chosen
    still trains correctly.

    The prompt is wrapped in the **chat template** with the response as the
    assistant turn, and only the response tokens are supervised. This mirrors the
    reference repo (its mask generator templates the messages and masks all but
    the assistant turn) and, more importantly, matches `NLA.verbalize()`, which
    calls `apply_chat_template(..., add_generation_prompt=True)` at inference.
    Training on the bare prompt string would leave the model to meet the
    `<|im_start|>assistant` scaffolding for the first time at eval.
    """

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        config: NLAConfig | None = None,
        max_length: int = 1024,
        limit: int | None = None,
        enable_thinking: bool = False,
        shuffle_seed: int | None = None,
    ) -> None:
        self.config = config or NLAConfig()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.enable_thinking = enable_thinking
        table, self.activations = load_activations(path, limit)
        if shuffle_seed is not None:
            self.activations = shuffled_activations(self.activations, shuffle_seed)
        self.prompts = [
            p.replace("<INJECT>", self.config.placeholder_token)
            for p in table.column("prompt").to_pylist()
        ]
        self.responses = table.column("response").to_pylist()
        self.scale = self.config.resolve_injection_scale(self.activations.shape[1])

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        prompt_ids = self.tokenizer(self.prompts[index], add_special_tokens=False)["input_ids"]
        response_ids = self.tokenizer(self.responses[index], add_special_tokens=False)["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            response_ids = response_ids + [self.tokenizer.eos_token_id]
        input_ids = (prompt_ids + response_ids)[: self.max_length]
        # Prompt tokens are context, not targets: the AV is being taught to emit
        # the summary given the injected vector, not to reproduce the template.
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]
        assert self.config.placeholder_token_id in input_ids, (
            f"row {index}: placeholder token {self.config.placeholder_token_id} absent after "
            f"tokenization — injection would silently no-op and the AV would train on the "
            f"literal template instead of the activation"
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "activation": self.activations[index],
        }


class ARDataset(Dataset[dict[str, Any]]):
    """`s -> h`: the suffix-anchored summary prompt, activation as the target."""

    def __init__(
        self,
        path: str,
        tokenizer: Any,
        config: NLAConfig | None = None,
        max_length: int = 1024,
        limit: int | None = None,
        shuffle_seed: int | None = None,
    ) -> None:
        self.config = config or NLAConfig()
        self.tokenizer = tokenizer
        self.max_length = max_length
        table, self.activations = load_activations(path, limit)
        if shuffle_seed is not None:
            self.activations = shuffled_activations(self.activations, shuffle_seed)
        self.prompts = table.column("prompt").to_pylist()
        # mse_scale, NOT injection_scale: the AR never injects, it *predicts* the
        # vector. This scale only sets the units of the loss, and sqrt(d) is what
        # keeps it O(1) and comparable to the reference's 0.938 baseline.
        self.scale = self.config.resolve_mse_scale(self.activations.shape[1])

    def __len__(self) -> int:
        return len(self.prompts)

    def __getitem__(self, index: int) -> dict[str, Any]:
        # `add_special_tokens=True` matches the extractor that produced the gold
        # activations. The reference repo flags this explicitly: without it the
        # backbone runs out of distribution and its layer-l means drift from the
        # gold regime (they measured init cos(mu_backbone, mu_gold) ~0 vs ~0.9+).
        # Qwen3 has bos_token=None so it is a no-op here, but it is the correct
        # setting and stays correct if the target model changes.
        #
        # Left-truncate: the activation is read at the LAST token, so the tail
        # (`</text> <summary>`) must survive. Cutting from the right would drop
        # the anchor and read the wrong position.
        ids = self.tokenizer(self.prompts[index], add_special_tokens=True)["input_ids"]
        return {
            "input_ids": ids[-self.max_length :],
            "activation": self.activations[index],
        }


def _pad(
    sequences: list[list[int]], pad_value: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad to the batch max; returns (ids, attention_mask, last_index)."""
    width = max(len(s) for s in sequences)
    ids = torch.full((len(sequences), width), pad_value, dtype=torch.long)
    mask = torch.zeros((len(sequences), width), dtype=torch.long)
    last = torch.zeros(len(sequences), dtype=torch.long)
    for row, seq in enumerate(sequences):
        ids[row, : len(seq)] = torch.tensor(seq, dtype=torch.long)
        mask[row, : len(seq)] = 1
        last[row] = len(seq) - 1
    return ids, mask, last


class AVCollator:
    """Pads AV rows and scales their activations to the injection norm."""

    def __init__(self, pad_token_id: int, scale: float | None) -> None:
        self.pad_token_id = pad_token_id
        self.scale = scale

    def __call__(self, items: list[dict[str, Any]]) -> AVBatch:
        ids, mask, _ = _pad([i["input_ids"] for i in items], self.pad_token_id)
        labels, _, _ = _pad([i["labels"] for i in items], -100)
        raw = torch.from_numpy(np.stack([i["activation"] for i in items]))
        return AVBatch(
            input_ids=ids,
            attention_mask=mask,
            labels=labels,
            activations=normalize_activation(raw, self.scale),
        )


class ARCollator:
    """Pads AR rows, tracking each row's final-token index for extraction."""

    def __init__(self, pad_token_id: int, scale: float | None) -> None:
        self.pad_token_id = pad_token_id
        self.scale = scale

    def __call__(self, items: list[dict[str, Any]]) -> ARBatch:
        ids, mask, last = _pad([i["input_ids"] for i in items], self.pad_token_id)
        raw = torch.from_numpy(np.stack([i["activation"] for i in items]))
        return ARBatch(
            input_ids=ids,
            attention_mask=mask,
            targets=normalize_activation(raw, self.scale),
            last_index=last,
        )


def predict_mean_baseline(
    activations: np.ndarray, scale: float | None, sample_size: int = 20_000
) -> float:
    """The loss a constant predictor achieves — the FVE denominator.

    This is the reference repo's `load_predict_mean_baselines`, not a plain
    variance. Because the AR loss normalizes the *prediction* as well as the
    target, the best a signal-free model can do is emit one constant direction,
    so the honest denominator is `MSE(normalize(mean_gold), normalize(gold))`
    averaged over the set — the achievable floor, not the raw spread.

    Their measured numbers: baseline 0.938, trained critic 0.586, so
    `FVE = 1 - 0.586/0.938 = 37.5%`. Their shuffled-target control scored 0.922,
    i.e. essentially the baseline — exactly what a model with no real signal
    should get.
    """
    rows = torch.from_numpy(activations[:sample_size])
    scaled = normalize_activation(rows, scale).float()
    constant = normalize_activation(scaled.mean(dim=0, keepdim=True), scale).float()
    return float(torch.mean((constant - scaled) ** 2).item())
