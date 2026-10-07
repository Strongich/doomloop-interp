"""Decode-time doubt-word ban for vLLM 0.22's V1 sampler (EXPERIMENT-reviewer-baselines.md §B2).

A custom V1 logits processor, loaded with `LLM(logits_processors=[BAN_PROCESSOR])`. It is a
no-op for every request that does not opt in, so ban and steering arms share one engine.
A request opts in with

    extra_args={"ban": {"file": "data/baselines/ban_tokens.json", "mode": "all" | "para",
                        "delay": 256}}

- `all`: logit -inf for every id of V_ban at every generated token (B1, NoWait-style).
- `para`: logit -inf for V_ban only when the previous generated token is a paragraph
  boundary (>= 2 newlines) inside `<think>` whose one-based generated index g satisfies
  `g >= delay` (B2). That is exactly N's site rule (`vllm_steering.boundary_positions`):
  N edits the boundary token's residual, which acts on the prediction of the next token;
  the ban forbids words in that same prediction.

The rule is a pure function of the request's own output tokens, so a KV-preemption
recompute replays the same decisions. Applied before penalties, temperature, top-k and
top-p (vLLM's non-argmax-invariant slot), so the sampler renormalizes over what is left.
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import torch
from vllm import SamplingParams
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.builtin import process_dict_updates

BAN_PROCESSOR = "reasoning_attention.serving.vllm_ban:DoubtBanLogitsProcessor"
MODES = ("all", "para")


def load_ban_file(path: str) -> tuple[list[int], np.ndarray, int]:
    """(V_ban ids, boundary mask over the vocabulary, `</think>` id)."""
    blob = json.loads(open(path).read())
    ban = [int(x["id"]) for x in blob["ban"]]
    bound = blob["boundary_ids"]
    mask = np.zeros(max(max(bound), max(ban)) + 1, dtype=bool)
    mask[bound] = True
    return ban, mask, int(blob["close_id"])


def para_ban_active(out: list[int], delay: int, boundary: np.ndarray, closed: bool) -> bool:
    """B2's rule for the token about to be sampled, given the generated tokens so far."""
    n = len(out)
    if closed or n == 0 or n < delay:
        return False
    last = out[-1]
    return last < len(boundary) and bool(boundary[last])


class _ReqState:
    __slots__ = ("mode", "delay", "out", "file", "scanned", "closed")

    def __init__(self, mode: str, delay: int, out: list[int], file: str) -> None:
        self.mode, self.delay, self.out, self.file = mode, delay, out, file
        self.scanned, self.closed = 0, False


class DoubtBanLogitsProcessor(LogitsProcessor):
    @classmethod
    def validate_params(cls, sampling_params: SamplingParams) -> None:
        cfg = (sampling_params.extra_args or {}).get("ban")
        if cfg is None:
            return
        if cfg.get("mode") not in MODES or not cfg.get("file"):
            raise ValueError(f"Bad ban config {cfg!r}")
        if int(cfg.get("delay", 0)) < 0:
            raise ValueError("ban delay must be >= 0")

    def __init__(self, vllm_config: Any, device: torch.device, is_pin_memory: bool) -> None:
        self.device = device
        self.reqs: dict[int, _ReqState] = {}
        self.files: dict[str, tuple[torch.Tensor, np.ndarray, int]] = {}

    def is_argmax_invariant(self) -> bool:
        return False

    def _new_state(
        self, params: SamplingParams, _prompt: list[int] | None, out: list[int]
    ) -> _ReqState | None:
        cfg = (params.extra_args or {}).get("ban")
        if cfg is None:
            return None
        if cfg["file"] not in self.files:
            ban, mask, close = load_ban_file(cfg["file"])
            self.files[cfg["file"]] = (
                torch.tensor(ban, dtype=torch.long, device=self.device),
                mask,
                close,
            )
        return _ReqState(cfg["mode"], int(cfg.get("delay", 0)), out, cfg["file"])

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        process_dict_updates(self.reqs, batch_update, self._new_state)

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        if not self.reqs:
            return logits
        rows: dict[str, list[int]] = {}
        for idx, st in self.reqs.items():
            _, boundary, close = self.files[st.file]
            if st.mode == "all":
                rows.setdefault(st.file, []).append(idx)
                continue
            if not st.closed and len(st.out) > st.scanned:
                st.closed = close in st.out[st.scanned :]
                st.scanned = len(st.out)
            if para_ban_active(st.out, st.delay, boundary, st.closed):
                rows.setdefault(st.file, []).append(idx)
        for file, picked in rows.items():
            ban = self.files[file][0]
            r = torch.tensor(picked, dtype=torch.long, device=logits.device)
            logits[r.unsqueeze(1), ban.unsqueeze(0)] = float("-inf")
        return logits
