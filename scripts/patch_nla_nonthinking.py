"""Pass `enable_thinking=False` at the reference rollout's chat-template call.

Qwen3's template defaults `enable_thinking` to True, which leaves the assistant
turn open so the model writes its own `<think>` block. Our AV was SFT'd
exclusively with `enable_thinking=False`, so every training prompt ended:

    <|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n<explanation>...

`nla/rollout/nla_generate.py` omits the kwarg, so at RL time the AV opened a
`<think>` block, drifted out of distribution into multilingual noise, never
emitted `</explanation>`, and every sample went TRUNCATED->FAILED at the flat
-2.0 reward — zero advantage variance, pg_loss exactly 0 (D43).

Their case-study models (Qwen2.5, Gemma-3, Llama-3.3) have no thinking mode, so
the omission is correct upstream. This is a Qwen3-specific adaptation.

Safe as a single-site change: the RL loss mask comes from `response_length`
(`miles/rollout/sglang_rollout.py`), not from a second chat-template render.
`MultiTurnLossMaskGenerator` (which renders its own prompts) is used only by
`sft_rollout.py`, and `schema.compute_canonical_neighbors` is unaffected because
the injection token sits inside the user content, ahead of any thinking
scaffolding.

`natural_language_autoencoders/` is gitignored and cloned per machine, so this
edit does not live in our history — hence this script. Idempotent; re-run after
any re-clone or `git pull` of that tree.

Usage:
    python scripts/patch_nla_nonthinking.py --nla-repo natural_language_autoencoders
"""

import argparse
from pathlib import Path

STOCK = """    prompt_str = _TOKENIZER.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )"""

PATCHED = """    prompt_str = _TOKENIZER.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        # Qwen3 defaults this to True and would leave the turn open for the model
        # to write its own <think> block. The AV was SFT'd with it False
        # throughout, so this must match or the policy is off-distribution. See
        # implementation-notes.md D43.
        enable_thinking=False,
    )"""

SENTINEL = "enable_thinking=False,"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nla-repo", default="natural_language_autoencoders")
    args = ap.parse_args()

    path = Path(args.nla_repo) / "nla" / "rollout" / "nla_generate.py"
    assert path.is_file(), f"{path} not found — is --nla-repo correct?"
    text = path.read_text(encoding="utf-8")

    if SENTINEL in text:
        print(f"{path}: already passes enable_thinking=False")
        return

    assert STOCK in text, (
        f"the stock apply_chat_template call was not found verbatim in {path}. "
        f"Upstream may have changed it — re-derive the replacement by hand rather "
        f"than forcing this patch."
    )
    path.write_text(text.replace(STOCK, PATCHED, 1), encoding="utf-8")
    print(f"{path}: now passes enable_thinking=False")


if __name__ == "__main__":
    main()
