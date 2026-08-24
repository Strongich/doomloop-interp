"""Central configuration for the model and the vLLM serving point.

Defaults are tuned for `Qwen/Qwen3-1.7B` in thinking mode on a single Blackwell
GPU. Every field is overridable so callers can tweak one value without losing
the rest of the tuned defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv

# Path of the dotenv file holding credentials (OPENAI_API_KEY, HF_TOKEN).
# Gitignored, and on the cluster it lives on the PVC so it survives pod restarts.
ENV_FILE = ".env"


def load_project_env(env_file: str = ENV_FILE) -> None:
    """Load `.env` into the process environment, without overriding real exports.

    Called by every CLI entry point rather than left to the shell: a login shell
    sources `.zshrc`, but `ssh host 'cmd'`, tmux send-keys, and cron do not, so a
    token exported only in `.zshrc` is invisible exactly when a long job needs it.
    `override=False` keeps an explicit `HF_TOKEN=... cmd` winning over the file.
    """
    if os.path.exists(env_file):
        load_dotenv(env_file, override=False)


# Hub repo id for the target model.
MODEL_ID = "Qwen/Qwen3-1.7B"

# Model that *authors* the natural-language explanations for the SFT warm-start.
# An external API model, not a local one and not the target: it only writes text,
# is never injected into, and its activations are never extracted. Supersedes the
# earlier local-MoE plan (see implementation-notes D13).
EXPLAINER_MODEL_ID = "gpt-5.6-luna"

# Corpus for the warm-start set. Text lives in the `content` column; the dataset
# has a single config ("default") with `en` and `zh` splits.
CORPUS_ID = "openbmb/Ultra-FineWeb"
CORPUS_CONFIG = "default"
CORPUS_SPLIT = "en"
CORPUS_TEXT_COLUMN = "content"

# Second corpus, used only for the Stage-2 RL prompt set: real chat traffic, so
# the AV sees activations from conversational text and not just web prose.
CHAT_CORPUS_ID = "allenai/WildChat-1M"
CHAT_CORPUS_CONFIG = "default"
CHAT_CORPUS_SPLIT = "train"
CHAT_CORPUS_COLUMN = "conversation"

# Architecture facts for Qwen3-1.7B (used by the NLA extraction code).
NUM_HIDDEN_LAYERS = 28
D_MODEL = 2048


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for loading / preparing the model."""

    model_id: str = MODEL_ID
    # Where `hf download` puts the snapshot. None -> default HF cache.
    local_dir: str | None = None
    revision: str = "main"
    # Qwen3 thinking mode toggle, applied at chat-template time.
    enable_thinking: bool = True


@dataclass(frozen=True)
class SamplingDefaults:
    """Qwen-recommended sampling settings for **thinking** mode.

    Source: Qwen3 model card. Greedy decoding is explicitly discouraged in
    thinking mode, so we keep a small temperature.
    """

    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 8192


@dataclass(frozen=True)
class ExplainerConfig:
    """The external API model that writes the warm-start explanations.

    `gpt-5.6-luna` at high reasoning effort authors the `(context -> summary)`
    text used to SFT both halves of the NLA. It runs offline, produces text only,
    and is never the injection target.

    Reasoning models reject `temperature`/`top_p`, so neither is exposed here —
    `reasoning_effort` is the only quality knob. `max_output_tokens` must leave
    room for reasoning tokens *plus* the visible answer, so it is set well above
    the ~100-word explanation budget; a response truncated before its closing
    tag fails extraction and the row is dropped.
    """

    model: str = EXPLAINER_MODEL_ID
    reasoning_effort: str = "high"
    max_output_tokens: int = 4096
    # Bounded in-flight requests. Raise once the account's rate limit is known.
    concurrency: int = 32
    # SDK-level transport retries (429/5xx, exponential backoff with jitter).
    max_retries: int = 10
    # Path to the dotenv file holding OPENAI_API_KEY. None -> rely on the
    # ambient environment.
    env_file: str | None = ".env"


@dataclass(frozen=True)
class WarmStartDataConfig:
    """The Stage-1 (SFT warm-start) dataset recipe, from the NLA paper.

    100k documents x 5 random positions = ~500k `(context, summary, h_l)`
    examples, split **evenly by document** into two disjoint halves: the AV
    trains `h -> s` on one, the AR trains `s -> h` on the other. A given pair is
    never seen by both models. Splitting by document (not by row) is what makes
    the halves disjoint at all — all 5 positions of a document travel together,
    so no context leaks across the boundary.
    """

    corpus: str = CORPUS_ID
    corpus_config: str | None = CORPUS_CONFIG
    corpus_split: str = CORPUS_SPLIT
    text_column: str = CORPUS_TEXT_COLUMN
    # Document slice: [corpus_start, corpus_start + n_documents).
    corpus_start: int = 0
    n_documents: int = 100_000
    positions_per_doc: int = 5
    # Documents are truncated to this many tokens before extraction, so the
    # sampled position always has <= 4096 tokens of left-context.
    max_context_tokens: int = 4096
    # Positions below this index have too little left-context to be meaningful.
    min_position: int = 50
    # Even document-level split: AV half / AR half. Must sum to 1.0.
    av_fraction: float = 0.5
    ar_fraction: float = 0.5
    seed: int = 42

    @property
    def expected_pairs(self) -> int:
        """Upper bound on generated pairs (short docs yield fewer)."""
        return self.n_documents * self.positions_per_doc


@dataclass(frozen=True)
class RLDataConfig:
    """The Stage-2 (GRPO) prompt set — deliberately smaller than the paper's.

    The paper draws 500k activations per source. We use **200k per source**
    (40k documents x 5 positions), for 400k RL prompts total, split evenly
    between web prose (Ultra-FineWeb) and chat traffic (WildChat). RL needs no
    summaries: the AV generates the explanation and the AR scores it, so this set
    carries prompts + activations only, and costs no API spend.

    Two sources rather than one because the AV is graded on activations it has to
    verbalize, and web prose alone is a narrow slice of what the residual stream
    at layer 20 ever holds. WildChat positions are sampled the same way — random
    token positions in the chat-templated conversation.
    """

    n_documents_per_source: int = 40_000
    positions_per_doc: int = 5
    # Web prose half.
    web_corpus: str = CORPUS_ID
    web_corpus_config: str | None = CORPUS_CONFIG
    web_corpus_split: str = CORPUS_SPLIT
    web_text_column: str = CORPUS_TEXT_COLUMN
    # Chat half.
    chat_corpus: str = CHAT_CORPUS_ID
    chat_corpus_config: str | None = CHAT_CORPUS_CONFIG
    chat_corpus_split: str = CHAT_CORPUS_SPLIT
    chat_text_column: str = CHAT_CORPUS_COLUMN
    # Start the web slice past the warm-start slice so RL activations come from
    # documents the SFT stage never saw.
    web_corpus_start: int = 100_000
    chat_corpus_start: int = 0
    max_context_tokens: int = 4096
    min_position: int = 50
    seed: int = 43  # distinct from the warm-start seed

    @property
    def activations_per_source(self) -> int:
        return self.n_documents_per_source * self.positions_per_doc

    @property
    def total_activations(self) -> int:
        return 2 * self.activations_per_source


@dataclass(frozen=True)
class NLAConfig:
    """Natural-Language Activations setup (transformer-circuits.pub/2026/nla).

    Defines which activation we extract from the target model and how it is
    normalized before being injected as the special token's embedding.
    """

    model_id: str = MODEL_ID

    # --- What activation ---
    # Residual stream = the *output of decoder layer l* (hidden_states), NOT
    # post-LayerNorm and NOT an attention/MLP sub-output. With HF
    # `output_hidden_states=True`, hidden_states is a tuple of length
    # num_layers+1 where index 0 is the embedding and index i is the output of
    # layer i-1. So the residual stream after layer l == hidden_states[l + 1].
    extraction_layer: int = 20  # 0-indexed; ~71% depth of 28 layers
    # Token position the activation is read from: final token of the snippet.
    token_position: int = -1

    # --- How it's normalized before injection ---
    # The activation is rescaled to a target L2-norm before it overwrites the
    # placeholder's embedding (reference repo's `normalize_activation`):
    #   "sqrt_d_model"        -> sqrt(d_model), the "ambient residual-stream scale"
    #                            (default; matches the repo's resolve_target_scale)
    #   None / "raw" / "none" -> inject raw, magnitude preserved
    #   float                 -> scale to exactly that L2-norm
    injection_scale: float | str | None = "sqrt_d_model"

    # --- Placeholder token (task 1) ---
    # We do NOT add a new token to the vocabulary. The "special token" is only a
    # structural slot in the prompt whose embedding we overwrite with the scaled
    # activation, so its lexical identity is irrelevant. We repurpose an existing
    # token that the text-only model never emits: `<|image_pad|>` (id 151655), a
    # multimodal padding placeholder unused by Qwen3-1.7B in text mode. It's a
    # single token with a real embedding row, and safe from colliding with normal
    # generation.
    placeholder_token: str = "<|image_pad|>"
    placeholder_token_id: int = 151655

    # --- AR (reconstructor) ---
    # Learned affine map applied to the AR's layer-l activation:
    #   activation_pred = A @ h (+ b).
    # The reference repo (natural_language_autoencoders) uses a bias-FREE linear
    # value head, so we default to no bias to match it; the NLA blog writes the
    # map as "A @ x + b", so this is exposed as a knob.
    ar_affine_bias: bool = False

    def resolve_injection_scale(self, d_model: int = D_MODEL) -> float | None:
        """Turn `injection_scale` into a concrete target L2-norm (or None=raw).

        Mirrors the reference repo's `resolve_target_scale`.
        """
        raw = self.injection_scale
        if raw is None or raw in ("raw", "none"):
            return None
        if raw == "sqrt_d_model":
            return float(d_model) ** 0.5
        return float(raw)

    @property
    def hidden_states_index(self) -> int:
        """Index into HF `outputs.hidden_states` for the layer-l residual stream."""
        return self.extraction_layer + 1

    @property
    def ar_num_layers(self) -> int:
        """How many leading layers the AR model keeps (first l layers + layer l)."""
        return self.extraction_layer + 1


@dataclass(frozen=True)
class VLLMConfig:
    """Throughput-oriented vLLM engine configuration.

    The numeric fields come straight from the spec and are passed verbatim to
    `vllm.LLM(...)`.
    """

    model_id: str = MODEL_ID
    trust_remote_code: bool = True
    gpu_memory_utilization: float = 0.95
    max_num_batched_tokens: int = 24576
    max_num_seqs: int = 128
    max_model_len: int = 16384
    enable_prefix_caching: bool = True
    tensor_parallel_size: int = 1

    sampling: SamplingDefaults = field(default_factory=SamplingDefaults)
