"""Completion backends for the explanation stage.

`CompletionProvider` is the pluggable interface, mirroring the reference repo:
hand it a batch of fully-formed prompts, get back one completion per prompt (or
None for a prompt that exhausted its retries). Concurrency, retries, rate
limits, and auth are the provider's problem.

`OpenAIProvider` targets the Responses API with a reasoning-effort setting,
which is what `gpt-5.6-luna` expects. A None return is a per-prompt gave-up
signal: the explain stage drops that row and logs the count, so one stubborn 429
in a chunk of 512 doesn't waste the other 511 calls.
"""

from __future__ import annotations

import asyncio
import os
from abc import ABC, abstractmethod
from typing import Any

import openai
from dotenv import load_dotenv
from openai import omit
from openai.types.shared_params import Reasoning

from reasoning_attention.config import ExplainerConfig


class CompletionProvider(ABC):
    """Submit a batch of prompts, get a batch of completions back."""

    @abstractmethod
    def complete(self, prompts: list[str]) -> list[str | None]:
        """Map `prompts[i] -> completion[i]`, or None where retries ran out."""


def load_api_key(env_file: str | None = ".env") -> str:
    """Read OPENAI_API_KEY, loading `env_file` first if it exists.

    Raises RuntimeError with an actionable message rather than letting the SDK
    fail later with an opaque auth error.
    """
    if env_file and os.path.exists(env_file):
        load_dotenv(env_file)
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            f"OPENAI_API_KEY is not set and was not found in {env_file!r}. "
            f"Copy the key into that file as `OPENAI_API_KEY=sk-...` or export it."
        )
    return key


class OpenAIProvider(CompletionProvider):
    """OpenAI Responses API with bounded async concurrency.

    Reasoning models reject `temperature`/`top_p`, so neither is sent —
    `reasoning_effort` is the quality knob. `max_output_tokens` covers reasoning
    tokens *plus* the visible answer; when the cap is hit the response comes back
    `incomplete` and this returns None, because a summary truncated before its
    closing `</analysis>` tag would be dropped downstream anyway.

    Only rate-limit / connection / 5xx errors degrade to None. Anything else
    (auth, bad request, unknown model) raises — those are bugs or config errors,
    not transients, and silently dropping every row would look like a data
    problem instead of a broken key.

    Calls `asyncio.run()`, so don't invoke it from inside a running event loop.
    """

    # Exceptions we degrade to None on instead of killing the batch.
    _TOLERATED = (
        openai.RateLimitError,
        openai.APIConnectionError,
        openai.APITimeoutError,
        openai.InternalServerError,
    )

    def __init__(self, config: ExplainerConfig | None = None) -> None:
        self.config = config or ExplainerConfig()
        # A local server needs no real key, but the SDK insists on one.
        if self.config.base_url:
            key = os.environ.get("OPENAI_API_KEY") or "local"
        else:
            key = load_api_key(self.config.env_file)
        self.client = openai.AsyncOpenAI(
            api_key=key,
            base_url=self.config.base_url,
            max_retries=self.config.max_retries,
        )
        self.api_kind = self.config.api_kind
        # Surfaced as plain attributes so the sidecar can record them.
        self.model = self.config.model
        # Typed so mypy can match the Responses `reasoning=` overload; the SDK
        # validates the effort string server-side.
        self.reasoning: Reasoning = {"effort": self.config.reasoning_effort}  # type: ignore[typeddict-item]
        self.reasoning_effort = self.config.reasoning_effort
        self.max_output_tokens = self.config.max_output_tokens
        self.concurrency = self.config.concurrency
        # Local-only generation knobs; see ExplainerConfig for why they matter.
        self.chat_reasoning_effort = self.config.chat_reasoning_effort
        self.chat_enable_thinking = self.config.chat_enable_thinking

    async def _one_chat(self, prompt: str) -> str | None:
        """Plain /v1/chat/completions — what vLLM and SGLang expose.

        No `reasoning` parameter: local servers reject unknown fields, and the
        effort knob does not exist there. A response cut off by the token cap is
        dropped for the same reason as the hosted path — no closing tag is coming.
        """
        # `omit` (not None) is how the SDK leaves a parameter out of the payload —
        # sending an explicit null makes vLLM reject the request.
        effort: Any = self.chat_reasoning_effort if self.chat_reasoning_effort else omit
        extra_body: dict[str, object] | None = None
        if self.chat_enable_thinking is not None:
            extra_body = {"chat_template_kwargs": {"enable_thinking": self.chat_enable_thinking}}
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_output_tokens,
            reasoning_effort=effort,
            extra_body=extra_body,
        )
        choice = resp.choices[0]
        if choice.finish_reason == "length":
            return None
        return (choice.message.content or "").strip() or None

    async def _one(self, sem: asyncio.Semaphore, prompt: str) -> str | None:
        if self.api_kind == "chat":
            async with sem:
                return await self._one_chat(prompt)
        async with sem:
            resp = await self.client.responses.create(
                model=self.model,
                input=prompt,
                reasoning=self.reasoning,
                max_output_tokens=self.max_output_tokens,
            )
        # Ran out of budget mid-thought: no closing tag is coming.
        if resp.status == "incomplete":
            return None
        assert resp.status == "completed", (
            f"unexpected response status={resp.status!r} (want completed/incomplete)"
        )
        text = (resp.output_text or "").strip()
        # A reasoning model can spend the whole budget thinking and emit nothing.
        return text or None

    def complete(self, prompts: list[str]) -> list[str | None]:
        if not prompts:
            return []

        async def _run() -> list[str | None | BaseException]:
            sem = asyncio.Semaphore(self.concurrency)
            return await asyncio.gather(
                *(self._one(sem, p) for p in prompts),
                return_exceptions=True,
            )

        raw = asyncio.run(_run())
        out: list[str | None] = []
        n_empty = 0
        n_failed = 0
        for i, result in enumerate(raw):
            if isinstance(result, str):
                out.append(result)
            elif result is None:
                n_empty += 1
                out.append(None)
            elif isinstance(result, self._TOLERATED):
                n_failed += 1
                out.append(None)
            elif isinstance(result, BaseException):
                # Auth / bad request / unknown model — a bug, not a transient.
                raise result
            else:
                raise AssertionError(
                    f"gather returned unexpected type at [{i}]: {type(result).__name__}"
                )
        if n_empty or n_failed:
            print(
                f"  [OpenAIProvider] dropped {n_empty} empty/truncated + "
                f"{n_failed} retry-exhausted of {len(prompts)}"
            )
        return out
