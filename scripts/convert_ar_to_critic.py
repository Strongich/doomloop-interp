"""Convert our AR checkpoint into the layout the reference's critic loader expects.

Our `training.sft.save_checkpoint` writes the AR as two pieces — `backbone/` (a
truncated HF model) and `affine.pt` — because `ARModel` is not a
`PreTrainedModel`. `NLACriticModel.from_pretrained` instead wants ONE flat
directory: the truncated HF model at the top level plus `value_head.safetensors`
beside it (`nla/models.py:162`).

The weights need no surgery. Their `value_head` is `nn.Linear(d, d, bias=False)`
saved with the `value_head.` prefix stripped, so its state dict is `{"weight":
[d, d]}` — byte-for-byte the shape and key our `affine.pt` already holds, because
`ar_affine_bias=False` matches their bias-free head. This is a re-layout, not a
conversion.

The affine is kept in fp32 (it is trained in fp32); their loader casts it to the
backbone dtype after loading, so no precision decision is made here.
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoTokenizer

from reasoning_attention.config import D_MODEL, NLAConfig
from reasoning_attention.nla.sidecar_export import build_model_sidecar, write_model_sidecar


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ar-checkpoint", required=True, help="our AR dir (has backbone/ + affine.pt)")
    ap.add_argument("--output", required=True, help="flat critic dir for --critic-load")
    ap.add_argument(
        "--tokenizer-from",
        default=None,
        help="dir to load the tokenizer from before re-saving it into --output. "
        "Defaults to the AR checkpoint root. Point it at the base model when the "
        "AR's own tokenizer_config.json was written by an incompatible "
        "transformers major version.",
    )
    ap.add_argument(
        "--verify-with",
        default=None,
        help="path to the interpreter that will CONSUME this critic (normally "
        ".venv-rl/bin/python). Loads the written tokenizer with it and asserts the "
        "placeholder round-trips, instead of guessing from version numbers.",
    )
    ap.add_argument(
        "--megatron-compat",
        action="store_true",
        help="also write model-megatron-compat.safetensors (norm.weight=ones, "
        "lm_head.weight=eye). Only needed for the megatron backend; our GRPO runs "
        "--train-backend fsdp, where mbridge is not involved.",
    )
    args = ap.parse_args()
    _cfg = NLAConfig()
    PLACEHOLDER_TOKEN = _cfg.placeholder_token
    PLACEHOLDER_TOKEN_ID = _cfg.placeholder_token_id

    src = Path(args.ar_checkpoint)
    backbone, affine = src / "backbone", src / "affine.pt"
    assert backbone.is_dir(), f"{backbone} not found — is this an AR checkpoint?"
    assert affine.is_file(), f"{affine} not found — is this an AR checkpoint?"

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for item in backbone.iterdir():
        if item.is_dir():
            shutil.copytree(item, out / item.name, dirs_exist_ok=True)
        else:
            shutil.copy2(item, out / item.name)
    print(f"copied backbone -> {out}")

    # The tokenizer lives in the AR checkpoint ROOT, not in backbone/: our
    # save_checkpoint writes `backbone.save_pretrained(dir/"backbone")` but
    # `tokenizer.save_pretrained(dir)`. Leaving the critic dir with no tokenizer
    # makes the reference fail its drift check with "'<|image_pad|>' -> []".
    #
    # RE-SAVE rather than copy. The AR checkpoint was written on the training side
    # under transformers 5.x, whose tokenizer_config.json is not backward-readable:
    # `extra_special_tokens` is serialized as a LIST there and as a DICT in 4.x, so
    # 4.57 dies with "'list' object has no attribute 'keys'". Round-tripping through
    # the *running* AutoTokenizer emits this env's format, and the assert below
    # refuses to write a file the RL venv cannot read.
    tok_src = Path(args.tokenizer_from) if args.tokenizer_from else src
    tokenizer = AutoTokenizer.from_pretrained(str(tok_src), trust_remote_code=True)
    # Megatron/FSDP critic_fwd passes attention_mask=None (causal-only), so a
    # left-pad would be attended by the last real token. Match the reference.
    tokenizer.padding_side = "right"
    tokenizer.save_pretrained(str(out))
    print(f"re-saved tokenizer from {tok_src} (transformers {transformers.__version__})")

    cfg_path = out / "tokenizer_config.json"
    tok_cfg = json.loads(cfg_path.read_text())
    # D35 was a FORMAT mismatch: the AR checkpoint was written under transformers
    # 5.x, whose tokenizer_config.json 4.57 cannot read (`extra_special_tokens` is
    # a list there, a dict in 4.x). The original guard hardcoded "the RL venv pins
    # 4.57" — stale since D44 moved it to 5.12.1, where 5.x format is the CORRECT
    # output and that guard aborts a valid conversion.
    #
    # Version heuristics keep going stale, so test the thing we actually care
    # about: can the venv that will READ this file load it, and does the
    # placeholder still round-trip? (The reference's drift check fails with
    # "'<|image_pad|>' -> []" when it does not.)
    if args.verify_with:
        probe = subprocess.run(
            [
                args.verify_with,
                "-c",
                "import sys;from transformers import AutoTokenizer;"
                "t=AutoTokenizer.from_pretrained(sys.argv[1],trust_remote_code=True);"
                "ids=t.encode(sys.argv[2],add_special_tokens=False);"
                "print(t.__class__.__module__.split('.')[0],ids)",
                str(out),
                PLACEHOLDER_TOKEN,
            ],
            capture_output=True,
            text=True,
        )
        if probe.returncode != 0:
            raise SystemExit(
                f"{args.verify_with} cannot read {cfg_path}:\n{probe.stderr.strip()[-800:]}\n"
                f"Re-run this script with that interpreter so the tokenizer is "
                f"written in the format the RL venv reads."
            )
        if f"[{PLACEHOLDER_TOKEN_ID}]" not in probe.stdout:
            raise SystemExit(
                f"placeholder {PLACEHOLDER_TOKEN!r} does not round-trip to "
                f"[{PLACEHOLDER_TOKEN_ID}] under {args.verify_with}: {probe.stdout.strip()}"
            )
        print(
            f"verified: {args.verify_with} reads the tokenizer, "
            f"{PLACEHOLDER_TOKEN} -> [{PLACEHOLDER_TOKEN_ID}]"
        )
    else:
        print(
            "WARNING: no --verify-with; the RL venv's ability to read this "
            "tokenizer is UNVERIFIED (see D35)"
        )

    sd = torch.load(affine, map_location="cpu")
    assert set(sd) == {"weight"}, f"expected only 'weight' in affine.pt, got {sorted(sd)}"
    w = sd["weight"]
    assert w.shape == (D_MODEL, D_MODEL), (
        f"affine weight is {tuple(w.shape)}, want {(D_MODEL,) * 2}"
    )
    save_file({"weight": w.contiguous()}, str(out / "value_head.safetensors"))
    print(f"affine.pt -> value_head.safetensors {tuple(w.shape)} {w.dtype}")

    if args.megatron_compat:
        dtype = w.dtype
        save_file(
            {
                "model.norm.weight": torch.ones(D_MODEL, dtype=dtype),
                "lm_head.weight": torch.eye(D_MODEL, dtype=dtype),
            },
            str(out / "model-megatron-compat.safetensors"),
        )
        print("wrote model-megatron-compat.safetensors")

    cfg = NLAConfig()
    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    print(f"sidecar -> {write_model_sidecar(out, build_model_sidecar(tok, cfg))}")

    n_layers = json.loads((out / "config.json").read_text())["num_hidden_layers"]
    print(f"config.json num_hidden_layers={n_layers} (want {cfg.ar_num_layers})")
    assert n_layers == cfg.ar_num_layers, (
        f"backbone has {n_layers} layers but the AR should have {cfg.ar_num_layers} "
        f"(layer {cfg.extraction_layer} + 1) — the truncation did not survive the save"
    )
    # The check the reference performs at load time. Doing it here means a broken
    # copy fails in this script instead of deep inside Miles at launch.
    tok_out = AutoTokenizer.from_pretrained(str(out))
    ids = tok_out.encode(cfg.placeholder_token, add_special_tokens=False)
    assert ids == [cfg.placeholder_token_id], (
        f"placeholder {cfg.placeholder_token!r} encodes to {ids} with the copied "
        f"tokenizer, expected [{cfg.placeholder_token_id}] — the tokenizer did not "
        f"survive the copy"
    )
    print(f"placeholder round-trip OK: {cfg.placeholder_token!r} -> {ids}")
    print("OK")


if __name__ == "__main__":
    main()
