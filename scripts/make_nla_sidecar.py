"""Write a reference-format `nla_meta.yaml` into a checkpoint directory.

Thin CLI over `reasoning_attention.nla.sidecar_export`, which carries the reasoning
for the two values that matter (injection_scale 1000, mse_scale sqrt_d_model).
"""

import argparse

from transformers import AutoTokenizer

from reasoning_attention.config import NLAConfig
from reasoning_attention.nla.sidecar_export import build_model_sidecar, write_model_sidecar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="AV or AR checkpoint dir")
    ap.add_argument("--injection-scale", default=None)
    ap.add_argument("--mse-scale", default=None)
    args = ap.parse_args()

    cfg = NLAConfig()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    meta = build_model_sidecar(tok, cfg, args.injection_scale, args.mse_scale)
    out = write_model_sidecar(args.checkpoint, meta)
    e, t = meta["extraction"], meta["tokens"]
    print(f"wrote {out}")
    print(f"  injection_scale={e['injection_scale']}  mse_scale={e['mse_scale']}")
    print(f"  placeholder {t['injection_char']!r} id={t['injection_token_id']} "
          f"neighbors=({t['injection_left_neighbor_id']}, {t['injection_right_neighbor_id']})")
    print(f"  critic_suffix_ids={t['critic_suffix_ids']} | critic layers="
          f"{meta['critic']['extraction_layer_index']}")


if __name__ == "__main__":
    main()
