"""Make miles' `ring_flash_attn` import lazy so transformers 5.x can load it.

`ring-flash-attn` 0.1.8 (its latest — there is no newer release) does:

    from transformers.modeling_flash_attention_utils import is_flash_attn_greater_or_equal_2_10

and transformers 5.x removed that symbol. miles imports the package at MODULE
level in `backends/fsdp_utils/actor.py`, so merely importing
`miles.backends.fsdp_utils` raises ImportError under transformers 5.12.1 — which
is the version sglang v0.5.15 pins, and the version we need because 4.57.1
mis-computes Qwen3's injected forward (D44).

The import is genuinely optional for us: the only call site is guarded by
`if self.parallel_state.cp_size > 1`, i.e. context parallelism, which we do not
use (cp_size defaults to 1 and we never set it). Moving the import to that call
site keeps ring attention working for anyone who does use CP, while letting the
module import cleanly for everyone who does not.

`.rl-src/miles` is a disposable clone that `setup_rl_stack.sh` hard-resets before
re-applying patches, so this must run after that — see the setup script.

Usage:
    python scripts/patch_miles_ring_attn.py --miles-src .rl-src/miles
"""

import argparse
from pathlib import Path

TOP_IMPORT = "from ring_flash_attn import update_ring_flash_attn_params\n"

CALL_SITE = """                update_ring_flash_attn_params(cu_seqlens, self.cp_group)"""

LAZY_CALL = """                # Imported here, not at module scope: ring-flash-attn 0.1.8 pulls
                # `is_flash_attn_greater_or_equal_2_10`, which transformers 5.x
                # removed. Only context parallelism needs it, and cp_size > 1 is
                # the branch we are already inside. See D45.
                from ring_flash_attn import update_ring_flash_attn_params

                update_ring_flash_attn_params(cu_seqlens, self.cp_group)"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--miles-src", default=".rl-src/miles")
    args = ap.parse_args()

    path = Path(args.miles_src) / "miles" / "backends" / "fsdp_utils" / "actor.py"
    assert path.is_file(), f"{path} not found — is --miles-src correct?"
    text = path.read_text(encoding="utf-8")

    if TOP_IMPORT not in text:
        print(f"{path}: module-level ring_flash_attn import already removed")
        return

    assert text.count(TOP_IMPORT) == 1, (
        f"expected 1 module-level import in {path}, found {text.count(TOP_IMPORT)}"
    )
    assert text.count(CALL_SITE) == 1, (
        f"expected exactly 1 call site in {path}, found {text.count(CALL_SITE)} — "
        f"upstream moved it; re-derive this patch rather than forcing it"
    )
    text = text.replace(TOP_IMPORT, "", 1).replace(CALL_SITE, LAZY_CALL, 1)
    path.write_text(text, encoding="utf-8")
    print(f"{path}: ring_flash_attn import is now lazy (context-parallel only)")


if __name__ == "__main__":
    main()
