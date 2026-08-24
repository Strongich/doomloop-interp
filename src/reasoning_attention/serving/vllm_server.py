"""vLLM serving point.

`build_llm()` returns a configured vLLM `LLM` instance tuned for max throughput.
The engine itself is thinking-mode agnostic; thinking mode is a per-request
concern, enabled by passing `chat_template_kwargs={"enable_thinking": True}` to
`llm.chat(...)` (see `chat_thinking` below).
"""

from __future__ import annotations

from typing import Any

from reasoning_attention.config import VLLMConfig


def build_llm(config: VLLMConfig | None = None) -> Any:
    """Build and return a vLLM `LLM` instance configured for max throughput.

    All throughput knobs come from `VLLMConfig` (spec defaults). Importing vLLM
    is deferred to call time so the rest of the package stays importable without
    a CUDA-capable environment.
    """
    from vllm import LLM

    config = config or VLLMConfig()

    return LLM(
        model=config.model_id,
        trust_remote_code=config.trust_remote_code,
        gpu_memory_utilization=config.gpu_memory_utilization,
        max_num_batched_tokens=config.max_num_batched_tokens,
        max_num_seqs=config.max_num_seqs,
        max_model_len=config.max_model_len,
        enable_prefix_caching=config.enable_prefix_caching,
        tensor_parallel_size=config.tensor_parallel_size,
    )


def build_sampling_params(config: VLLMConfig | None = None, **overrides: Any) -> Any:
    """Build vLLM `SamplingParams` using Qwen's thinking-mode defaults.

    `overrides` (e.g. `max_tokens=2048`) take precedence over the defaults.
    """
    from vllm import SamplingParams

    config = config or VLLMConfig()
    s = config.sampling
    params: dict[str, Any] = dict(
        temperature=s.temperature,
        top_p=s.top_p,
        top_k=s.top_k,
        max_tokens=s.max_tokens,
    )
    params.update(overrides)
    return SamplingParams(**params)


def chat_thinking(
    llm: Any,
    messages: list[dict[str, str]],
    config: VLLMConfig | None = None,
    enable_thinking: bool = True,
    **sampling_overrides: Any,
) -> Any:
    """Run `llm.chat` with Qwen3 thinking mode toggled on.

    Returns the list of vLLM `RequestOutput`. `messages` is a single
    conversation; wrap in a list of conversations for batched throughput.
    """
    config = config or VLLMConfig()
    sampling_params = build_sampling_params(config, **sampling_overrides)
    return llm.chat(
        messages,
        sampling_params=sampling_params,
        chat_template_kwargs={"enable_thinking": enable_thinking},
    )


if __name__ == "__main__":
    engine = build_llm()
    outputs = chat_thinking(
        engine,
        [{"role": "user", "content": "How many r's are in 'strawberry'?"}],
    )
    for out in outputs:
        print(out.outputs[0].text)
