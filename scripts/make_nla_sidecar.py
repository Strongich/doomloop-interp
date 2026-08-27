"""Write a reference-format `nla_meta.yaml` into an SFT checkpoint directory.

Stage 2 runs inside the reference's Miles integration, which loads its NLA
settings from a sidecar next to the checkpoint (`nla.config.load_nla_config` ->
`{ckpt_dir}/nla_meta.yaml`). Our `training.sft.save_checkpoint` writes only
weights and a tokenizer, so that file has to be produced here.

Two fields carry the values this project had to work out, and getting either
wrong is the difference between the AV reading the vector and free-associating
off the placeholder:

  injection_scale  1000  - the reference has NO default for this; an absent key
                          resolves to None and train_actor asserts. It is 1000
                          for us, not sqrt(d_model), because the rule is "a round
                          number just above the dataset's mean activation norm"
                          and ours average ~900 (D29).
  mse_scale        sqrt_d_model (45.25 at d_model 2048) - a SEPARATE knob (D30).
                          Reusing injection_scale here inflates the loss ~488x
                          and pins grad_norm at the clip.

`injection_char` is the reference's name for what we call the placeholder token.
We do not add a vocabulary entry: `<|image_pad|>` (151655) is an existing token
the text-only model never emits, and injection happens at the embedding level so
its lexical identity is irrelevant.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from reasoning_attention.config import NLAConfig
from reasoning_attention.nla.prompts import AR_TEMPLATE, AV_TEMPLATE, build_av_content

AR_SUFFIX = "</text> <summary>"


def canonical_neighbors(tokenizer, cfg: NLAConfig) -> tuple[int, int, int]:
    """Token ids either side of the placeholder in the rendered AV prompt.

    The reference verifies these at load time to catch tokenizer drift. Returns
    (left_id, right_id, n_sites) — n_sites must be exactly 1 or injection is
    ambiguous and the run would inject into the wrong position.
    """
    ids = tokenizer(build_av_content(cfg.placeholder_token), add_special_tokens=False)["input_ids"]
    sites = [i for i, t in enumerate(ids) if t == cfg.placeholder_token_id]
    assert len(sites) == 1, (
        f"expected exactly 1 placeholder site in the AV prompt, found {len(sites)}. "
        f"Injection would be ambiguous."
    )
    pos = sites[0]
    assert pos > 0 and pos < len(ids) - 1, "placeholder must not be the first or last token"
    return ids[pos - 1], ids[pos + 1], len(sites)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="AV or AR checkpoint dir")
    ap.add_argument("--kind", default="nla_model", choices=["nla_model"])
    ap.add_argument("--injection-scale", default=None, help="default: NLAConfig.injection_scale")
    ap.add_argument("--mse-scale", default=None, help="default: NLAConfig.mse_scale")
    args = ap.parse_args()

    cfg = NLAConfig()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    left, right, _ = canonical_neighbors(tok, cfg)
    # tokenize(suffix)[1:] — the first token merges with the preceding character
    # (e.g. "detail." + "</" -> ".</"), so it is unstable and excluded.
    suffix_ids = tok(AR_SUFFIX, add_special_tokens=False)["input_ids"][1:]

    inj = args.injection_scale if args.injection_scale is not None else cfg.injection_scale
    mse = args.mse_scale if args.mse_scale is not None else cfg.mse_scale

    meta = {
        "kind": args.kind,
        "d_model": cfg.d_model,
        "extraction": {
            "layer_index": cfg.extraction_layer,
            "injection_scale": inj,
            "mse_scale": mse,
        },
        "tokens": {
            "injection_char": cfg.placeholder_token,
            "injection_token_id": cfg.placeholder_token_id,
            "injection_left_neighbor_id": left,
            "injection_right_neighbor_id": right,
            "critic_suffix_ids": suffix_ids,
        },
        "prompt_templates": {"av": AV_TEMPLATE, "ar": AR_TEMPLATE},
        "critic": {"extraction_layer_index": cfg.ar_num_layers},
    }

    out = Path(args.checkpoint) / "nla_meta.yaml"
    out.write_text(yaml.safe_dump(meta, sort_keys=False), encoding="utf-8")
    print(f"wrote {out}")
    print(f"  injection_scale={inj}  mse_scale={mse}  d_model={cfg.d_model}")
    print(f"  placeholder {cfg.placeholder_token!r} id={cfg.placeholder_token_id} "
          f"neighbors=({left}, {right})")
    print(f"  critic_suffix_ids={suffix_ids} (from {AR_SUFFIX!r})")
    print(f"  critic layers={cfg.ar_num_layers} | extraction layer={cfg.extraction_layer}")


if __name__ == "__main__":
    main()
