"""Emit the reference repo's `nla_meta.yaml` for one of our checkpoints.

Stage 2 runs inside the reference's Miles integration, which reads its NLA
settings from a sidecar beside the checkpoint (`nla.config.load_nla_config` ->
`{ckpt_dir}/nla_meta.yaml`). Our `training.sft.save_checkpoint` writes weights and
a tokenizer only, so this module produces that file.

Two fields carry values this project had to derive, and either one wrong is the
difference between the AV reading the vector and free-associating off the
placeholder token:

  injection_scale  1000 - the reference has NO default; an absent key resolves to
                          None and `train_actor` asserts. Ours is 1000, not
                          sqrt(d_model), because the rule is "a round number a bit
                          above the dataset's mean activation norm" and ours
                          average ~900 (D29).
  mse_scale        sqrt_d_model (45.25 at d_model 2048) - a SEPARATE knob (D30);
                          reusing injection_scale here inflates the loss ~488x.

`injection_char` is their name for our placeholder token. No vocabulary entry is
added: `<|image_pad|>` (151655) is an existing token the text-only model never
emits, and injection is at the embedding level so lexical identity is irrelevant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from reasoning_attention.config import D_MODEL, NLAConfig
from reasoning_attention.nla.prompts import AR_TEMPLATE, AV_TEMPLATE, build_av_content

AR_SUFFIX = "</text> <summary>"


def canonical_neighbors(tokenizer: Any, cfg: NLAConfig) -> tuple[int, int]:
    """Token ids either side of the placeholder in the rendered AV prompt.

    The reference verifies these at load time to catch tokenizer drift. Asserts a
    single injection site: more than one and injection is ambiguous, so the run
    would write the activation into the wrong position.
    """
    ids = tokenizer(build_av_content(cfg.placeholder_token), add_special_tokens=False)["input_ids"]
    sites = [i for i, t in enumerate(ids) if t == cfg.placeholder_token_id]
    assert len(sites) == 1, (
        f"expected exactly 1 placeholder site in the AV prompt, found {len(sites)} — "
        f"injection would be ambiguous"
    )
    pos = sites[0]
    assert 0 < pos < len(ids) - 1, "placeholder must not be the first or last token"
    return ids[pos - 1], ids[pos + 1]


def build_model_sidecar(
    tokenizer: Any,
    cfg: NLAConfig | None = None,
    injection_scale: float | str | None = None,
    mse_scale: float | str | None = None,
) -> dict[str, Any]:
    """The sidecar dict for a *model* checkpoint (`kind: nla_model`)."""
    cfg = cfg or NLAConfig()
    left, right = canonical_neighbors(tokenizer, cfg)
    # tokenize(suffix)[1:] — the first token merges with the preceding character
    # (e.g. "detail." + "</" -> ".</"), so it is unstable and excluded.
    suffix_ids = tokenizer(AR_SUFFIX, add_special_tokens=False)["input_ids"][1:]
    return {
        "kind": "nla_model",
        "d_model": D_MODEL,
        "extraction": {
            "layer_index": cfg.extraction_layer,
            "injection_scale": injection_scale
            if injection_scale is not None
            else cfg.injection_scale,
            "mse_scale": mse_scale if mse_scale is not None else cfg.mse_scale,
        },
        "tokens": {
            "injection_char": cfg.placeholder_token,
            "injection_token_id": cfg.placeholder_token_id,
            "injection_left_neighbor_id": left,
            "injection_right_neighbor_id": right,
            "critic_suffix_ids": suffix_ids,
        },
        # Their loader formats the actor template as
        # `actor_template.format(injection_char=...)` (`nla/schema.py`
        # compute_canonical_neighbors), so the exported template must use THEIR
        # field name. Ours says {placeholder} — the one documented deviation from
        # their verbatim template — so translate it on the way out and leave the
        # in-repo constant alone. The AR template needs no translation: both call
        # the field {explanation}.
        "prompt_templates": {
            "av": AV_TEMPLATE.replace("{placeholder}", "{injection_char}"),
            "ar": AR_TEMPLATE,
        },
        # K, the extraction LAYER INDEX (20) — not the layer COUNT (21). Their
        # loader reads this into a field called `critic_num_layers`, but
        # rl_preflight asserts `num_hidden_layers == k + 1`, so the value is K.
        # Writing ar_num_layers here made the critic claim extraction at layer 21
        # while holding blocks 0..20. Their notes price this exact off-by-one at a
        # ~0.32 FVE ceiling, because the head must then partly undo a block the
        # gold activation never passed through.
        "critic": {"extraction_layer_index": cfg.extraction_layer},
    }


def write_model_sidecar(checkpoint_dir: str | Path, meta: dict[str, Any]) -> Path:
    """Write `meta` to `{checkpoint_dir}/nla_meta.yaml` and return the path."""
    out = Path(checkpoint_dir) / "nla_meta.yaml"
    out.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    return out
