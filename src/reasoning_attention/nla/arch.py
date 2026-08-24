"""Architecture helpers for building the AR (truncated) model.

Distilled from the reference repo (natural_language_autoencoders/nla/models.py
and arch_adapters.py), specialized for single-GPU Qwen3 — no FSDP, no multimodal
wrappers, no Megatron. Only the bits we need to truncate the backbone correctly.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

# Per-layer config arrays that transformers (>=4.50) validates to have length
# == num_hidden_layers (configuration_utils.py:layer_type_validation). When we
# truncate the layer count we must slice these to match or config init raises.
_PER_LAYER_ARRAY_ATTRS = ("layer_types", "sliding_window_pattern", "no_rope_layers")


def truncate_config_layers(config: Any, num_layers: int) -> None:
    """Set num_hidden_layers AND slice per-layer arrays to match.

    Must be called on the HF config *before* `from_pretrained`, so the weight
    loader only reads `num_layers` blocks. Do NOT slice `nn.ModuleList`
    post-hoc — the reference repo notes that breaks FSDP sharding, and it also
    leaves the config lying about its own depth.
    """
    config.num_hidden_layers = num_layers
    for attr in _PER_LAYER_ARRAY_ATTRS:
        value = getattr(config, attr, None)
        if isinstance(value, (list, tuple)) and len(value) > num_layers:
            setattr(config, attr, type(value)(value[:num_layers]))


# How deep to walk looking for the module that owns `.layers`. A bare causal LM
# is 1 hop; a PEFT-wrapped one is 3 (PeftModel -> LoraModel -> CausalLM -> Model).
_MAX_UNWRAP_DEPTH = 6


def inner_transformer(backbone: Any) -> nn.Module:
    """Return the inner transformer — the module holding `.layers` and `.norm`.

    Qwen/Llama/Mistral reach it at `.model`, GPT-2/Falcon at `.transformer`, but a
    single hop is not enough once the backbone is wrapped: PEFT nests it as
    `PeftModel.base_model.model.model`, and taking `.model` once lands on the
    causal LM, which has no `.layers`. So descend until a module actually owns
    `.layers` rather than assuming a fixed depth.

    Descending is safe for LoRA: PEFT replaces target submodules in place, so the
    adapters live inside this module tree and still run when it is called
    directly.
    """
    node = backbone
    for _ in range(_MAX_UNWRAP_DEPTH):
        if hasattr(node, "layers"):
            assert isinstance(node, nn.Module)
            return node
        if hasattr(node, "model"):
            node = node.model
        elif hasattr(node, "transformer"):
            node = node.transformer
        else:
            break
    raise AssertionError(
        f"could not find the inner transformer under {type(backbone).__name__} "
        f"(walked .model/.transformer up to {_MAX_UNWRAP_DEPTH} levels looking for "
        f".layers) — extend inner_transformer() for this architecture"
    )


def strip_lm_head(backbone: Any) -> None:
    """Replace the lm_head with Identity — the AR never produces logits."""
    if hasattr(backbone, "lm_head"):
        backbone.lm_head = nn.Identity()


def strip_final_norm(inner: nn.Module) -> str:
    """Replace the final layernorm with Identity, returning the attr name.

    The AR feeds the *raw* residual stream (output of block l) to its affine
    map, so the final RMSNorm must not be applied — that's what makes
    `last_hidden_state` equal the activation the AV extracted.
    """
    for attr in ("norm", "final_layernorm", "ln_f"):
        if hasattr(inner, attr):
            setattr(inner, attr, nn.Identity())
            return attr
    raise AssertionError(
        f"could not find a final layernorm on {type(inner).__name__} — "
        f"extend strip_final_norm() with the right attribute name"
    )
