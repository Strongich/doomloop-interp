"""NLA: the two models — AV (full target) + AR (truncated reconstructor).

Mirrors the split in the reference repo (natural_language_autoencoders):

  - AV  ("activation verbalizer", their *actor*): the FULL Qwen3 causal LM. It
    takes a fixed prompt with one placeholder token whose embedding is replaced
    by an injected activation, and generates a plain-text explanation.

  - AR  ("activation reconstructor", their *critic*): Qwen3 truncated to its
    first (l+1) decoder blocks, with the final RMSNorm and lm_head stripped, plus
    a learned affine map. It reads the explanation and predicts the layer-l
    activation; training minimizes MSE against the AV activation.

For now this only *constructs* the two models. Activation extraction, the
placeholder injection, the affine-map translation and the MSE training loop come
later — this is just the AV/AR scaffold the rest hangs off.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from reasoning_attention.config import NLAConfig
from reasoning_attention.nla.arch import (
    inner_transformer,
    strip_final_norm,
    strip_lm_head,
    truncate_config_layers,
)
from reasoning_attention.nla.injection import inject_at_placeholder, normalize_activation
from reasoning_attention.nla.prompts import build_av_messages, extract_explanation


@dataclass
class AROutput:
    """Output of the AR forward pass."""

    activation_pred: torch.Tensor  # [B, T, d_model] — affine map applied to residual
    residual_stream: torch.Tensor  # [B, T, d_model] — raw layer-l residual (pre-affine)


class ARModel(nn.Module):
    """Qwen3 truncated to its first (l+1) blocks + a learned affine map.

    The backbone's `inner_transformer` keeps blocks 0..l with final-norm replaced
    by Identity, so `last_hidden_state` is the *raw residual stream at the output
    of block l* — the same quantity the AV side extracts. The affine map
    (`activation_pred = A @ h (+ b)`) translates the AR context to the AV context.
    """

    def __init__(
        self, backbone: Any, d_model: int, bias: bool = False, identity_init: bool = True
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.affine = nn.Linear(d_model, d_model, bias=bias)
        if identity_init:
            self._identity_init_affine()
        self._align_affine_placement()

    def _identity_init_affine(self) -> None:
        """Start the affine map at the identity, not at PyTorch's default.

        The reference repo flags this as critical (TRAINING_NOTES.md, "Critical:
        identity-init `value_head`"): `nn.Linear`'s kaiming_uniform default scales
        the backbone's output norm by ~1/sqrt(3), so at step 0 their
        `pred_norm ~= 48` against `backbone_norm ~= 83`. With the identity,
        `pred_norm ~= backbone_norm` and their initial loss drops 1.94 -> 1.61
        (~17% better starting direction match).

        The identity is the right prior anyway: the AR reads the same residual
        stream the AV wrote, so "change nothing" is a better starting guess than a
        random rotation.
        """
        with torch.no_grad():
            self.affine.weight.copy_(torch.eye(self.affine.out_features))
            if self.affine.bias is not None:
                self.affine.bias.zero_()

    def _align_affine_placement(self) -> None:
        """Move the affine map onto the backbone's device/dtype.

        `nn.Linear` initializes on CPU/fp32; the backbone was placed (and cast)
        by `from_pretrained`. Align so the forward doesn't bounce device/dtype.
        Skipped if the backbone is on `meta` (nothing materialized yet).
        """
        inner: Any = inner_transformer(self.backbone)
        last_param = next(inner.layers[-1].parameters())
        if not last_param.is_meta:
            self.affine.to(device=last_param.device, dtype=last_param.dtype)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> AROutput:
        out = inner_transformer(self.backbone)(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            **kwargs,
        )
        h = out.last_hidden_state  # [B, T, d_model], raw residual (norm == Identity)
        # The affine may be held in a different dtype than the backbone — training
        # keeps it in fp32 for numerical headroom while the blocks run bf16 — so
        # cast into its dtype rather than assuming they match.
        affine_dtype = self.affine.weight.dtype
        return AROutput(
            activation_pred=self.affine(h.to(affine_dtype)),
            residual_stream=h,
        )


class NLA:
    """Holds the AV (full target model) and AR (truncated reconstructor).

    Build with `NLA.from_pretrained()`. Everything beyond construction
    (extraction / injection / training) is added later.
    """

    def __init__(
        self,
        config: NLAConfig,
        tokenizer: Any,
        av_model: Any,
        ar_model: ARModel,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.av = av_model  # full causal LM — verbalizes an injected activation
        self.ar = ar_model  # truncated transformer + affine map — reconstructs it

    @property
    def d_model(self) -> int:
        return int(self.av.config.hidden_size)

    @torch.no_grad()
    def verbalize(
        self,
        activation: torch.Tensor,
        *,
        max_new_tokens: int = 256,
        enable_thinking: bool = False,
        do_sample: bool = False,
        return_explanation: bool = False,
    ) -> str:
        """Run the AV mechanism: explain an activation vector in natural language.

        Builds the AV prompt with the placeholder token, replaces that token's
        embedding with the (scaled) `activation`, and generates the explanation.

        activation: [d_model] (single site). Scaled per `config.injection_scale`.
        return_explanation: if True, return just the text inside <explanation>
            tags (None becomes ""); otherwise return the full decoded output.
        """
        device = self.av.device
        cfg = self.config

        # 1. AV prompt with the real placeholder token in the <concept> slot.
        messages = build_av_messages(cfg.placeholder_token)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        enc = self.tokenizer(prompt, return_tensors="pt").to(device)
        input_ids = enc["input_ids"]

        # 2. Scale the activation to the configured injection norm.
        scale = cfg.resolve_injection_scale(self.d_model)
        vec = normalize_activation(activation.reshape(1, -1).to(device), scale)

        # 3. Embed the prompt, then overwrite the placeholder row with the vector.
        embeddings = self.av.get_input_embeddings()(input_ids)
        injected = inject_at_placeholder(input_ids, embeddings, vec, cfg.placeholder_token_id)

        # 4. Generate from the injected embeddings (decoder-only generate with
        #    inputs_embeds returns only the new continuation tokens).
        generated = self.av.generate(
            inputs_embeds=injected,
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )
        text = self.tokenizer.decode(generated[0], skip_special_tokens=True)
        if return_explanation:
            return extract_explanation(text) or ""
        return text

    @classmethod
    def from_pretrained(
        cls,
        config: NLAConfig | None = None,
        *,
        dtype: torch.dtype = torch.bfloat16,
        device_map: str | dict[str, int] | None = "auto",
        trust_remote_code: bool = True,
    ) -> NLA:
        config = config or NLAConfig()
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)

        # --- AV: the full target model ---
        av_model = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )

        # --- AR: truncated to the first (l+1) blocks ---
        ar_backbone, d_model = cls._load_truncated_backbone(
            config, dtype, device_map, trust_remote_code
        )
        ar_model = ARModel(ar_backbone, d_model=d_model, bias=config.ar_affine_bias)

        return cls(config, tokenizer, av_model, ar_model)

    @staticmethod
    def _load_truncated_backbone(
        config: NLAConfig,
        dtype: torch.dtype,
        device_map: str | dict[str, int] | None,
        trust_remote_code: bool,
    ) -> tuple[Any, int]:
        """Load Qwen3 with only its first `ar_num_layers` (= l+1) blocks.

        Truncates the config *before* `from_pretrained` so the loader reads only
        the kept blocks (transformers will warn about the unused tail weights —
        expected). Then strips lm_head and the final norm.
        """
        hf_config = AutoConfig.from_pretrained(config.model_id, trust_remote_code=trust_remote_code)
        assert config.ar_num_layers <= hf_config.num_hidden_layers, (
            f"ar_num_layers={config.ar_num_layers} exceeds base depth {hf_config.num_hidden_layers}"
        )
        truncate_config_layers(hf_config, config.ar_num_layers)

        backbone = AutoModelForCausalLM.from_pretrained(
            config.model_id,
            config=hf_config,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        strip_lm_head(backbone)
        strip_final_norm(inner_transformer(backbone))
        return backbone, hf_config.hidden_size
