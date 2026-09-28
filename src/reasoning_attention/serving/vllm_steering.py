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
    repeat_k: int = 0,
    repeat_minlen: int = 8,
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
    sites = (selected + first).tolist()
    if repeat_k and sites:
        cut = repeat_trigger(tokens, prompt_length, end, boundary_ids, repeat_k, repeat_minlen)
        if cut is not None:
            sites = [p for p in sites if p < cut]
    return sites


def repeat_trigger(
    tokens: Any, prompt_length: int, end: int, boundary_ids: np.ndarray, k: int, minlen: int
) -> int | None:
    """Absolute position of the first paragraph-boundary token that completes a paragraph
    (>= minlen tokens, boundary included) occurring for the k-th time in the generated
    history before `end`. Steering is withheld at that site and every later one.

    A loop guard (EXPERIMENT-transfer-8b-log.md): on Qwen3-8B, suppressing doubt at every
    boundary sometimes also suppresses closing </think>, and the model restates its
    answer verbatim until the budget. Stateless over the token history, so a KV-preemption
    recompute makes the same decision.
    """
    gen = np.asarray(tokens[prompt_length:end])
    counts: dict[bytes, int] = {}
    start = 0
    for i in np.flatnonzero(boundary_ids[gen]).tolist():
        if i + 1 - start >= minlen:
            key = gen[start : i + 1].tobytes()
            counts[key] = counts.get(key, 0) + 1
            if counts[key] >= k:
                return prompt_length + i
        start = i + 1
    return None


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
        # Pipeline parallelism is supported: every PP rank holds the full layer list,
        # with PPMissingLayer outside its own [start, end) range, so exactly one rank
        # owns the steered block and installs the hook. Tensor parallelism is not
        # validated here and stays refused.
        if cfg.parallel_config.tensor_parallel_size != 1:
            raise RuntimeError("This steering backend does not support tensor parallelism")
        if hasattr(self, "_reasoning_hook"):
            raise RuntimeError("Steering hook already installed")
        model = self.get_model()
        self._reasoning_owner = False
        if type(model.model.layers[layer]).__name__ == "PPMissingLayer":
            return {
                "layer": layer,
                "vllm": version("vllm"),
                "model": type(model).__name__,
                "owner": False,
            }
        # Qwen3-MoE decoder layers return the same (mlp_out, residual) pair as the dense
        # ones (vllm 0.22 qwen3_moe.py), so the full block output is still their sum.
        if type(model).__name__ not in ("Qwen3ForCausalLM", "Qwen3MoeForCausalLM"):
            raise RuntimeError(f"Expected Qwen3/Qwen3-MoE CausalLM, got {type(model).__name__}")
        self._reasoning_boundaries = np.zeros(model.config.vocab_size, dtype=bool)
        self._reasoning_boundaries[boundary_ids] = True
        self._reasoning_close = close_id
        self._reasoning_unit = None
        self._reasoning_alpha = 0.0
        self._reasoning_thinking = True
        self._reasoning_delay = 0
        self._reasoning_repeat_k = 0
        self._reasoning_repeat_minlen = 8
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
                    self._reasoning_repeat_k,
                    self._reasoning_repeat_minlen,
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
        self._reasoning_owner = True
        return {
            "layer": layer,
            "vllm": version("vllm"),
            "model": type(model).__name__,
            "owner": True,
        }

    def configure_reasoning_steering(
        self: Any,
        unit: list[float] | None,
        alpha: float,
        thinking: bool = True,
        start_delay: int = 0,
        repeat_k: int = 0,
        repeat_minlen: int = 8,
    ) -> None:
        """Call between completed generate() calls, never while requests are active.

        `start_delay` is part of the policy, not the installation, so one engine
        serves every delay in a sweep -- but it is fixed for a whole batch. Vary
        it per request only by batching each policy separately.
        """
        if not getattr(self, "_reasoning_owner", False):
            return  # a pipeline rank that does not hold the steered block
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
        self._reasoning_repeat_k = int(repeat_k)
        self._reasoning_repeat_minlen = int(repeat_minlen)
        self._reasoning_stats = {}
        self._reasoning_calls = 0

    def reasoning_steering_stats(self: Any) -> dict[str, Any] | None:
        if not getattr(self, "_reasoning_owner", False):
            return None
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


class CaptureWorkerExtension(SteeringWorkerExtension):
    """Adds prefill-time capture of full block outputs (paired-activation extraction).

    A request opts in with `extra_args={"capture_id": key, "capture_positions": [...]}`
    (absolute prompt positions). The owning pipeline rank stores `hidden + residual` at
    those positions for every installed layer; `pop_captures()` returns and clears them.
    Use prefill-only requests (max_tokens=1) with prefix caching off.
    """

    def install_capture(self: Any, layers: list[int]) -> dict[str, Any]:
        runner = self.model_runner
        model = self.get_model()
        self._captures: dict[str, dict[int, dict[int, torch.Tensor]]] = {}
        owned = [
            layer
            for layer in layers
            if type(model.model.layers[layer]).__name__ != "PPMissingLayer"
        ]

        def make_hook(layer: int) -> Any:
            def hook(_module: Any, _inputs: Any, output: Any) -> Any:
                hidden, residual = output
                batch = runner.input_batch
                query = runner.query_start_loc.np
                for row, req_id in enumerate(batch.req_ids):
                    extra = runner.requests[req_id].sampling_params.extra_args or {}
                    want = extra.get("capture_positions")
                    if not want:
                        continue
                    start = int(batch.num_computed_tokens_cpu[row])
                    count = int(query[row + 1] - query[row])
                    hit = [p for p in want if start <= p < start + count]
                    if not hit:
                        continue
                    idx = torch.tensor(
                        [int(query[row]) + p - start for p in hit], device=hidden.device
                    )
                    full = hidden.index_select(0, idx) + residual.index_select(0, idx)
                    store = self._captures.setdefault(extra["capture_id"], {}).setdefault(layer, {})
                    for p, v in zip(hit, full.to(torch.bfloat16).cpu(), strict=True):
                        store[p] = v
                return output

            return hook

        for layer in owned:
            model.model.layers[layer].register_forward_hook(make_hook(layer))
        return {"owned": owned}

    def pop_captures(self: Any) -> dict[str, dict[int, dict[int, list[float]]]]:
        # Plain float lists: RPC serialization does not round-trip torch tensors.
        out, self._captures = getattr(self, "_captures", {}), {}
        return {
            cid: {
                layer: {p: v.float().tolist() for p, v in pos.items()} for layer, pos in d.items()
            }
            for cid, d in out.items()
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
    pipeline_parallel_size: int = 1,
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
        pipeline_parallel_size=pipeline_parallel_size,
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
    owners = [r for r in result if r["owner"]]
    if len(owners) != 1 or owners[0]["layer"] != layer or len(result) != pipeline_parallel_size:
        raise RuntimeError(f"Steering installation failed: {result}")
    return llm


def steering_stats(llm: Any) -> dict[str, Any]:
    """Audit stats from the one worker that owns the steered block."""
    stats = [s for s in llm.collective_rpc("reasoning_steering_stats") if s is not None]
    if len(stats) != 1:
        raise RuntimeError(f"Expected stats from exactly one pipeline rank, got {len(stats)}")
    return stats[0]
