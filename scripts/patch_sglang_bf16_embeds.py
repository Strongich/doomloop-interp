"""Teach sglang's /generate to decode bf16-base64 input_embeds.

Every rollout request ships the prompt's whole embedding matrix. At our
injection_scale the client takes the fp32-JSON branch: ~12 MB per request
against ~2.8 MB for bf16-base64, and at 512 requests per rollout that is ~6 GB
serialized per step. The ep1 run measured the cost directly — of 85.5s per step,
66.8s (78%) was `perf/train_wait_time`, the actor idle while sglang generated,
against ~10-20s of real generation for 512 sequences of 150 tokens.

This is the SERVER half of a two-sided protocol. `nla/rollout/nla_generate.py`
already sends `input_embeds_b64_bf16` (bf16 bytes viewed as int16, since numpy
has no bf16) plus `input_embeds_shape`; without a decoder here FastAPI validates
a base64 STRING against a numeric-array schema and answers 400 on every request.
The run then reaches step 0, generates nothing, and dies.

WHY THIS IS A REBASE, not a fresh patch: the reference ships
`patches/nla_input_embeds_b64.patch`, whose context assumes the identifiers
`data`, `orjson` and `_GEN_REQ_FIELDS` from its sibling `nla_input_embeds.patch`.
On sglang v0.5.15 that sibling applies with NLA-prefixed names instead —
`_nla_data`, `_orjson`, `_NLA_GEN_REQ_FIELDS` — so the b64 hunk's context no
longer matches and `setup_rl_stack.sh` SKIPs it without failing. That is the
2026-08-28 failure recorded in `scripts/train_grpo.sh`, and the reason that
script checks for the decoder rather than trusting the flag.

NUMERICS: bf16 keeps 8 mantissa bits, so transport resolution is ~0.4% relative.
The reference's own gate calls `scale < 1000` safe and cites gemma27b at
injection_scale 60000 (~256 resolution, 4% KL spikes) as the failure case. Ours
is exactly 1000 — the boundary — so this is a real numerics change, not a pure
encoding change. The check is `tis_k3` in the training log, the train<->rollout
logprob mismatch: ~0.001 when transport is right, ~0.20 when it is wrong. Run
with TIS_METRICS=1 (the default) and read it at step 0.

`.rl-src/sglang` is cloned per machine and not carried by our history.
Idempotent; re-run after any re-clone or sglang upgrade.

Usage:
    python scripts/patch_sglang_bf16_embeds.py
    python scripts/patch_sglang_bf16_embeds.py --http-server /path/to/http_server.py
"""

import argparse
import subprocess
import sys
from pathlib import Path

MARKER = "input_embeds_b64_bf16"

# The line the sibling NLA patch installs. We anchor on it rather than on stock
# sglang because the bf16 decode must land AFTER the body is parsed and BEFORE
# GenerateReqInput is constructed — the dataclass is what would reject a string.
ANCHOR = "    _nla_data = _orjson.loads(await request.body())"

INSERT = '''
    # === NLA: bf16-base64 input_embeds (12MB -> 2.8MB on the wire) ===
    # The client sends bf16 bytes viewed as int16 (numpy has no bf16) plus the
    # shape; reinterpret here. io_struct's isinstance(..., float) check needs
    # real Python floats, hence .tolist(). schedule_batch casts to bf16 anyway,
    # so this is bit-exact end-to-end -- fp16 would lose dynamic range.
    # See scripts/patch_sglang_bf16_embeds.py.
    if "input_embeds_b64_bf16" in _nla_data:
        import base64 as _nla_base64

        import numpy as _nla_np
        import torch as _nla_torch

        _nla_raw = _nla_base64.b64decode(_nla_data.pop("input_embeds_b64_bf16"))
        _nla_shape = _nla_data.pop("input_embeds_shape")
        _nla_i16 = _nla_np.frombuffer(_nla_raw, dtype=_nla_np.int16).reshape(_nla_shape)
        _nla_data["input_embeds"] = (
            _nla_torch.from_numpy(_nla_i16.copy())
            .view(_nla_torch.bfloat16)
            .float()
            .tolist()
        )
    # === end NLA ==='''


def default_http_server() -> Path:
    """Ask the RL interpreter where sglang actually lives.

    The path differs per machine (.rl-src checkout vs site-packages), and
    guessing it is how the original patch went to the wrong file.
    """
    rl_python = Path(__file__).resolve().parents[1] / ".venv-rl" / "bin" / "python"
    interpreter = str(rl_python) if rl_python.is_file() else sys.executable
    out = subprocess.run(
        [interpreter, "-c",
         "import sglang.srt.entrypoints.http_server as m; print(m.__file__)"],
        capture_output=True, text=True, check=True,
    )
    return Path(out.stdout.strip())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--http-server", type=Path, default=None)
    args = ap.parse_args()

    path = args.http_server or default_http_server()
    assert path.is_file(), f"{path} not found"
    text = path.read_text(encoding="utf-8")

    if MARKER in text:
        print(f"{path}: already decodes bf16-base64 input_embeds")
        return

    assert ANCHOR in text, (
        f"{path} has no line `{ANCHOR.strip()}`.\n"
        "That line comes from the reference's nla_input_embeds.patch, which must be\n"
        "applied first — this patch only adds the bf16 decode on top of it. If the\n"
        "sibling patch applied under different names, re-derive the anchor rather\n"
        "than forcing this one."
    )
    assert text.count(ANCHOR) == 1, (
        f"expected exactly 1 anchor in {path}, found {text.count(ANCHOR)} — "
        "upstream changed it; re-derive this patch rather than forcing it"
    )

    path.write_text(text.replace(ANCHOR, ANCHOR + INSERT, 1), encoding="utf-8")
    print(f"{path}: /generate now decodes bf16-base64 input_embeds")
    print("Run with BF16_EMBEDS=1 and check tis_k3 at step 0 (~0.001 good, ~0.20 bad).")


if __name__ == "__main__":
    main()
