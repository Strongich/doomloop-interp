"""Norm-relative decoder-output steering for the installed vLLM 0.22 worker.

Single-GPU, synchronous scheduling, eager execution only. Hooks are installed
AFTER vLLM's memory profiling. Prefix caching is disabled by the engine factory:
changing the direction must never reuse intervention-dependent KV entries.
"""

from __future__ import annotations

import os
from importlib.metadata import version
from typing import Any

import numpy as np
import torch

SUPPORTED_VLLM = "0.22.0"


def boundary_positions(
    tokens: Any,
    prompt_length: int,
    start: int,
    count: int,
    boundary_ids: np.ndarray,
    close_id: int,
    thinking: bool = True,
    start_delay: int = 0,
) -> list[int]:
    """Absolute consumed-token sites; excludes prompt and all tokens after close.

    Recomputing generated tokens after KV preemption must replay the same edits.
    The final sampled token is not a site unless it is subsequently consumed.

    `start_delay` withholds the intervention for the first generated thinking
    tokens. For a consumed token at absolute position `p`, its one-based
    generated index is `g = p - prompt_length + 1`, and a site needs
    `g >= start_delay`. The prompt is never counted, so `start_delay=0` and `1`
    both mean "from the first generated boundary onward" and are the same policy.

    Note that for Qwen3's thinking template the `<think>` opener is GENERATED,
    not prompted, so it occupies g=1 and the count includes it. It is never a
    site itself: it is followed by a single newline, not the double newline a
    boundary token carries.
    """
    if not thinking:
        return []
    end = start + count
    first = max(start, prompt_length + max(0, start_delay - 1))
    if first >= end:
        return []
    history = np.asarray(tokens[prompt_length:end])
    closes = np.flatnonzero(history == close_id)
    if len(closes):
        end = min(end, prompt_length + int(closes[0]))
    if first >= end:
        return []
    selected = np.flatnonzero(boundary_ids[np.asarray(tokens[first:end])])
    return (selected + first).tolist()


def edit_decoder_output(
    hidden: torch.Tensor,
    residual: torch.Tensor,
    indices: torch.Tensor,
    unit: torch.Tensor,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Edit the full residual sum, preserving untouched vLLM tensor pairs exactly."""
    if indices.numel() == 0 or alpha == 0:
        return hidden, residual
    h = hidden.index_select(0, indices) + residual.index_select(0, indices)
    push = (alpha * h.float().norm(dim=-1, keepdim=True)) * unit
    edited = h + push.to(h.dtype)
    hidden = hidden.index_copy(0, indices, edited)
    residual = residual.index_fill(0, indices, 0)
    return hidden, residual


class SteeringWorkerExtension:
    """RPC mixin; vLLM supplies model_runner/get_model on the worker instance."""

    def install_reasoning_steering(
        self: Any, layer: int, boundary_ids: list[int], close_id: int
    ) -> dict[str, Any]:
        if version("vllm") != SUPPORTED_VLLM:
            raise RuntimeError(f"Steering validated for vLLM {SUPPORTED_VLLM} only")
        runner = self.model_runner
        if type(runner).__module__ != "vllm.v1.worker.gpu_model_runner":
            raise RuntimeError("Steering requires the V1 GPUModelRunner in vLLM 0.22")
        cfg = runner.vllm_config
        if not cfg.model_config.enforce_eager or runner.use_async_scheduling:
            raise RuntimeError("Steering requires enforce_eager=True, async_scheduling=False")
        if cfg.cache_config.enable_prefix_caching or cfg.speculative_config is not None:
            raise RuntimeError("Disable prefix caching and speculative decoding for steering")
        if (
            cfg.parallel_config.tensor_parallel_size != 1
            or cfg.parallel_config.pipeline_parallel_size != 1
        ):
            raise RuntimeError("This steering backend supports a single GPU only")
        if hasattr(self, "_reasoning_hook"):
            raise RuntimeError("Steering hook already installed")
        model = self.get_model()
        if type(model).__name__ != "Qwen3ForCausalLM":
            raise RuntimeError(f"Expected dense Qwen3ForCausalLM, got {type(model).__name__}")
        self._reasoning_boundaries = np.zeros(model.config.vocab_size, dtype=bool)
        self._reasoning_boundaries[boundary_ids] = True
        self._reasoning_close = close_id
        self._reasoning_unit = None
        self._reasoning_alpha = 0.0
        self._reasoning_thinking = True
        self._reasoning_delay = 0
        self._reasoning_stats: dict[str, Any] = {}
        self._reasoning_calls = 0

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden, residual = output
            batch = runner.input_batch
            query = runner.query_start_loc.np
            if int(query[batch.num_reqs]) != hidden.shape[0]:
                raise RuntimeError("Unexpected padded/microbatched forward in eager steering")
            indices: list[int] = []
            for row, req_id in enumerate(batch.req_ids):
                start = int(batch.num_computed_tokens_cpu[row])
                count = int(query[row + 1] - query[row])
                prompt_len = int(batch.num_prompt_tokens[row])
                sites = boundary_positions(
                    batch.token_ids_cpu[row],
                    prompt_len,
                    start,
                    count,
                    self._reasoning_boundaries,
                    self._reasoning_close,
                    self._reasoning_thinking,
                    self._reasoning_delay,
                )
                params = runner.requests[req_id].sampling_params
                extra = params.extra_args or {}
                audit_id = extra.get("steering_audit_id", req_id)
                stats = self._reasoning_stats.setdefault(
                    audit_id, {"sites": set(), "injections": set(), "applications": 0}
                )
                stats["sites"].update(sites)
                if self._reasoning_unit is not None and self._reasoning_alpha != 0:
                    stats["injections"].update(sites)
                    stats["applications"] += len(sites)
                    indices.extend(int(query[row]) + pos - start for pos in sites)
            self._reasoning_calls += 1
            if not indices:
                return output
            return edit_decoder_output(
                hidden,
                residual,
                torch.tensor(indices, device=hidden.device, dtype=torch.long),
                self._reasoning_unit,
                self._reasoning_alpha,
            )

        self._reasoning_hook = model.model.layers[layer].register_forward_hook(hook)
        return {"layer": layer, "vllm": version("vllm"), "model": type(model).__name__}

    def configure_reasoning_steering(
        self: Any,
        unit: list[float] | None,
        alpha: float,
        thinking: bool = True,
        start_delay: int = 0,
    ) -> None:
        """Call between completed generate() calls, never while requests are active.

        `start_delay` is part of the policy, not the installation, so one engine
        serves every delay in a sweep -- but it is fixed for a whole batch. Vary
        it per request only by batching each policy separately.
        """
        self._reasoning_unit = None
        if unit is not None:
            u = torch.tensor(unit, dtype=torch.float32, device=self.device)
            expected = self.get_model().config.hidden_size
            if u.shape != (expected,) or not bool(torch.isfinite(u).all()):
                raise ValueError("Invalid steering vector shape/values")
            if not torch.isclose(u.norm(), torch.ones((), device=u.device), atol=1e-3):
                raise ValueError("Expected an already-normalized unit direction")
            self._reasoning_unit = u
        if not np.isfinite(alpha):
            raise ValueError("alpha must be finite")
        self._reasoning_alpha = alpha
        self._reasoning_thinking = thinking
        if int(start_delay) != start_delay or start_delay < 0:
            raise ValueError("start_delay must be a non-negative integer token count")
        self._reasoning_delay = int(start_delay)
        self._reasoning_stats = {}
        self._reasoning_calls = 0

    def reasoning_steering_stats(self: Any) -> dict[str, Any]:
        return {
            "calls": self._reasoning_calls,
            "requests": {
                req_id: {
                    "sites": sorted(s["sites"]),
                    "injections": sorted(s["injections"]),
                    "applications": s["applications"],
                }
                for req_id, s in self._reasoning_stats.items()
            },
        }


def build_steering_llm(
    model_id: str,
    layer: int,
    boundary_ids: list[int],
    close_id: int,
    max_model_len: int,
    max_num_seqs: int = 32,
    gpu_memory_utilization: float = 0.85,
    max_num_batched_tokens: int = 2048,
) -> Any:
    """Build a version-checked engine; all experiment arms share this engine."""
    # 0.22 defaults Qwen3 to its V2 runner, which has a different metadata API.
    # Select the bundled V1 runner; do not replace/downgrade the installed package.
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "0"
    os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
    from vllm import LLM

    if version("vllm") != SUPPORTED_VLLM:
        raise RuntimeError(
            f"Expected installed vLLM {SUPPORTED_VLLM}; no packages are auto-updated"
        )
    llm = LLM(
        model=model_id,
        dtype="bfloat16",
        tensor_parallel_size=1,
        enforce_eager=True,
        async_scheduling=False,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        gpu_memory_utilization=gpu_memory_utilization,
        worker_extension_cls=("reasoning_attention.serving.vllm_steering.SteeringWorkerExtension"),
    )
    result = llm.collective_rpc("install_reasoning_steering", args=(layer, boundary_ids, close_id))
    if len(result) != 1 or result[0]["layer"] != layer:
        raise RuntimeError(f"Steering installation failed: {result}")
    return llm
