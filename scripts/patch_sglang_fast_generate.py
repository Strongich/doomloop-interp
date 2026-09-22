#!/usr/bin/env python3
"""Skip FastAPI/Pydantic auto-parse on sglang's /generate (D49).

v0.5.15 declares `async def generate_request(obj: GenerateReqInput, request: Request)`,
so FastAPI validates the whole JSON body into the dataclass before our handler
runs. For an NLA rollout that body is the prompt's entire embedding matrix as an
fp32 JSON list — ~12 MB, ~300k floats. The reference measured that auto-parse at
**155 ms/req for 448K floats**; at 512 requests per rollout that is ~79 s, against
a measured `perf/train_wait_time` of 118 s. It is single-threaded in the FastAPI
worker, which is why `#running-req` sits at 2.3 while the scheduler idles and why
moving the POSTs onto Ray actors (--use-distributed-post) bought only ~8%.

This is hunk 1 of their `nla_input_embeds.patch`, rewritten against v0.5.15 —
their version does not apply (upstream moved the file, and setup_rl_stack.sh's
tolerant loop SKIPped it silently during the v0.5.15 migration). It changes NO
numerics: same fp32 values, just parsed by orjson instead of Pydantic. The bf16
transport (hunk 2) is a separate, numerics-affecting change.

Idempotent. Run with the RL venv's python.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

MARKER = "_NLA_GEN_REQ_FIELDS"

OLD_SIG = (
    "async def generate_request(obj: GenerateReqInput, request: Request):\n"
    '    """Handle a generate request."""\n'
)

NEW_SIG = f'''async def generate_request(request: Request):
    """Handle a generate request."""
    # === NLA: skip FastAPI auto-parse (155ms/req for ~450K floats) ===
    # The body is the prompt's whole embedding matrix; Pydantic validating it
    # field-by-field dominates the rollout wait. orjson parses the same bytes and
    # we build the dataclass ourselves. Unknown keys are dropped rather than
    # raising, matching FastAPI's behaviour for extra fields here.
    _nla_data = _orjson.loads(await request.body())
    obj = GenerateReqInput(
        **{{k: v for k, v in _nla_data.items() if k in {MARKER}}}
    )
'''


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        print("usage: patch_sglang_fast_generate.py <path to http_server.py>")
        return 2
    path = Path(sys.argv[1])
    src = path.read_text()

    if MARKER in src:
        print(f"already patched: {path}")
        return 0
    if OLD_SIG not in src:
        print(
            f"ERROR: {path} does not contain the expected v0.5.15 signature.\n"
            f"Upstream changed /generate; re-derive this patch rather than forcing it.",
            file=sys.stderr,
        )
        return 1

    # Field whitelist + orjson import, placed after the last top-level import.
    preamble = f"""
# === NLA ===
import dataclasses as _nla_dataclasses

import orjson as _orjson

{MARKER} = {{f.name for f in _nla_dataclasses.fields(GenerateReqInput)}}
# === end NLA ===
"""
    anchor = re.search(r"^app = FastAPI\(", src, re.M)
    if anchor is None:
        print("ERROR: could not find `app = FastAPI(` to anchor the preamble", file=sys.stderr)
        return 1
    src = src[: anchor.start()] + preamble.lstrip("\n") + "\n" + src[anchor.start() :]
    src = src.replace(OLD_SIG, NEW_SIG, 1)
    path.write_text(src)
    print(f"patched {path}")
    print("  /generate now parses with orjson and builds GenerateReqInput directly")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
