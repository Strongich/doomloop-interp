"""Unwrap the BatchEncoding that transformers 5.x returns from apply_chat_template.

`nla/schema.py:compute_canonical_neighbors` does:

    ids = tokenizer.apply_chat_template(..., tokenize=True, add_generation_prompt=True)
    matches = [i for i, tid in enumerate(ids) if tid == injection_token_id]

Under transformers 4.x that call returns `list[int]`. Under 5.x it returns a
`BatchEncoding`, and iterating a BatchEncoding yields its KEY NAMES
(`['input_ids', 'attention_mask']`) — so the comparison against an int token id
never matches and the function dies with:

    AssertionError: injection token id 151655 ('<|image_pad|>') appears 0x in
    canonical actor prompt (expected 1)

which reads like a broken tokenizer or a wrong placeholder, and is neither.

This function is load-bearing: datagen uses it to POPULATE the sidecar's neighbor
ids and `load_nla_config` uses it to VERIFY them against the live tokenizer, so
every config load goes through it. We need transformers 5.12.1 for sglang v0.5.15
(D44), so the reference's 4.x assumption has to be relaxed.

`natural_language_autoencoders/` is gitignored and cloned per machine, so this
edit is not carried by our history. Idempotent; re-run after any re-clone.

Usage:
    python scripts/patch_nla_batchencoding.py --nla-repo natural_language_autoencoders
"""

import argparse
from pathlib import Path

STOCK = """    matches = [i for i, tid in enumerate(ids) if tid == injection_token_id]"""

PATCHED = """    # transformers 5.x returns a BatchEncoding here, not list[int]; iterating one
    # yields its key names, so the search below would find 0 matches and assert.
    # See scripts/patch_nla_batchencoding.py.
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    if len(ids) and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    matches = [i for i, tid in enumerate(ids) if tid == injection_token_id]"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nla-repo", default="natural_language_autoencoders")
    args = ap.parse_args()

    path = Path(args.nla_repo) / "nla" / "schema.py"
    assert path.is_file(), f"{path} not found — is --nla-repo correct?"
    text = path.read_text(encoding="utf-8")

    if 'if hasattr(ids, "keys"):' in text:
        print(f"{path}: already unwraps BatchEncoding")
        return
    assert text.count(STOCK) == 1, (
        f"expected exactly 1 match line in {path}, found {text.count(STOCK)} — "
        f"upstream changed it; re-derive this patch rather than forcing it"
    )
    path.write_text(text.replace(STOCK, PATCHED, 1), encoding="utf-8")
    print(f"{path}: compute_canonical_neighbors now unwraps BatchEncoding")


if __name__ == "__main__":
    main()
