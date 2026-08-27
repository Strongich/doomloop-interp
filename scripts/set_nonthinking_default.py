"""Make a Qwen3 checkpoint's chat template default to NON-thinking mode.

Why this is a checkpoint fix and not a caller fix
-------------------------------------------------
Our AV was SFT'd exclusively with `enable_thinking=False`
(`training/data.py`, `nla/model.py`), so every training prompt ended:

    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n<explanation>...

Qwen3's stock template only emits that closed `<think>` pair when
`enable_thinking` is explicitly false:

    {%- if enable_thinking is defined and enable_thinking is false %}

The reference's rollout calls `apply_chat_template(messages, tokenize=False,
add_generation_prompt=True)` (`nla/rollout/nla_generate.py:165`) and passes no
`enable_thinking`, so at RL time the block vanished, Qwen3 opened its own
`<think>`, and the AV went far out of distribution — degenerating into
multilingual noise, never emitting `</explanation>`, so every sample was
TRUNCATED->FAILED at a flat -2 reward. Their case-study models (Qwen2.5,
Gemma-3, Llama-3) have no thinking mode, so the omission is harmless for them.

Rather than patch their tree, invert the template's default so the checkpoint
declares how it must be prompted. Explicit callers are unaffected:

    enable_thinking=False    -> block emitted (unchanged)
    enable_thinking=True     -> block omitted (unchanged, opt back in)
    omitted                  -> block emitted (was omitted; this is the fix)

Usage:
    python scripts/set_nonthinking_default.py --checkpoint /path/to/av_sft
"""

import argparse
from pathlib import Path

STOCK = """{%- if enable_thinking is defined and enable_thinking is false %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}"""

PATCHED = """{%- if not (enable_thinking is defined and enable_thinking) %}
        {{- '<think>\\n\\n</think>\\n\\n' }}
    {%- endif %}"""

MARKER = "<think>\\n\\n</think>\\n\\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True, help="dir holding chat_template.jinja")
    args = ap.parse_args()

    path = Path(args.checkpoint) / "chat_template.jinja"
    assert path.is_file(), f"{path} not found — is this an HF checkpoint dir?"
    text = path.read_text(encoding="utf-8")

    if PATCHED.split("\n")[0] in text:
        print(f"{path}: already defaults to non-thinking")
        return

    assert text.count(MARKER) == 1, (
        f"expected exactly one {MARKER!r} site in {path}, found {text.count(MARKER)} — "
        f"the template is not the stock Qwen3 one; patch it by hand"
    )
    assert STOCK in text, (
        f"the stock guard was not found verbatim in {path}. Qwen3's template may have "
        f"changed; re-derive the replacement rather than forcing it."
    )
    path.write_text(text.replace(STOCK, PATCHED, 1), encoding="utf-8")
    print(f"{path}: now defaults to non-thinking")


if __name__ == "__main__":
    main()
