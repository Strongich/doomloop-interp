"""Fix ring-flash-attn's broken transformers fallback.

`ring_flash_attn/adapters/hf_adapter.py` (0.1.8, its latest release) guards the
import of `is_flash_attn_greater_or_equal_2_10` — but both branches read from the
same module, so the fallback re-raises the very error it was meant to catch:

    try:
        from transformers.modeling_flash_attention_utils import (
            is_flash_attn_greater_or_equal_2_10,
        )
    except ImportError:
        # transformers <= 4.53.x
        from transformers.modeling_flash_attention_utils import (   # <- same module
            is_flash_attn_greater_or_equal_2_10,
        )

transformers 5.x did not delete the symbol, it MOVED it to `transformers.utils`
(verified: `transformers.utils.is_flash_attn_greater_or_equal_2_10` exists at
5.12.1). So the fallback only needs to point at the right module.

This matters because miles imports ring_flash_attn at module scope in at least
two places (`fsdp_utils/actor.py`, `fsdp_utils/parallel.py`), so the ImportError
takes down `import miles.backends.fsdp_utils` entirely — and transformers 5.12.1
is what sglang v0.5.15 pins, which we need because 4.57.1 mis-computes Qwen3's
injected forward (D44).

Patching the installed package rather than miles fixes every call site at once.
Re-run after any `.venv-rl` rebuild. Idempotent.

Usage:
    python scripts/patch_ring_flash_attn.py --venv .venv-rl
"""

import argparse
import sys
from pathlib import Path

STOCK = """except ImportError:
    # transformers <= 4.53.x
    from transformers.modeling_flash_attention_utils import (
        is_flash_attn_greater_or_equal_2_10,
    )"""

PATCHED = """except ImportError:
    # transformers 5.x moved it to transformers.utils; the original fallback here
    # re-read the same module and so re-raised. See scripts/patch_ring_flash_attn.py.
    from transformers.utils import (
        is_flash_attn_greater_or_equal_2_10,
    )"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--venv", default=".venv-rl")
    ap.add_argument("--python-version", default="python3.11")
    args = ap.parse_args()

    path = (
        Path(args.venv)
        / "lib"
        / args.python_version
        / "site-packages"
        / "ring_flash_attn"
        / "adapters"
        / "hf_adapter.py"
    )
    if not path.is_file():
        print(f"{path} not found — ring-flash-attn not installed; nothing to do")
        return

    text = path.read_text(encoding="utf-8")
    if "from transformers.utils import (" in text:
        print(f"{path}: fallback already points at transformers.utils")
        return
    if STOCK not in text:
        print(
            f"WARNING: {path} does not contain the known-broken fallback verbatim. "
            f"Upstream may have fixed it; verify the import works before training.",
            file=sys.stderr,
        )
        return
    path.write_text(text.replace(STOCK, PATCHED, 1), encoding="utf-8")
    print(f"{path}: fallback now imports from transformers.utils")


if __name__ == "__main__":
    main()
