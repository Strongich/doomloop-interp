"""Memoize the constant per-request prompt work in nla_generate.

Every RL prompt is byte-identical — the AV template with one injected slot — but
`_prep_payload_sync` recomputes `apply_chat_template`, `encode`, and the
`[1, T, d]` embedding lookup for all 512 samples of every rollout. Measured with
NLA_DEBUG_TIMING=1 on transformers 5.12.1:

    prep=75..951ms (typ ~100ms)   post=2101..11757ms

against the ~30ms their docstring assumes. That work holds the GIL, so dispatch
serializes: ~50s of CPU to hand out one rollout, the event loop cannot service
responses (hence the inflated `post`), and sglang never gets above
`#running-req: 4` — versus 333 concurrent on the old stack. Net effect was
~180s/step instead of ~78s.

Caching keyed on the message content makes prep a dict hit plus an 885 KB clone.
Correctness is unchanged: a different prompt yields a different key and is
computed normally.

The block is located by its first and last lines rather than matched verbatim,
because we add comments to this function elsewhere (D43's enable_thinking).
Idempotent. Re-run after any re-clone of natural_language_autoencoders/.

Usage:
    python scripts/patch_nla_prompt_cache.py --nla-repo natural_language_autoencoders
"""

import argparse
import sys
from pathlib import Path

START = "    prompt_str = _TOKENIZER.apply_chat_template("
END = "        embeds = (_EMBED(ids_tensor) * _EMBED_SCALE).float()  # [1, T, d]"
MARKER = "_PROMPT_CACHE"

REPLACEMENT = '''    # --- memoized: every RL prompt is the same template (see patch script) ---
    _key = tuple((m.get("role"), m.get("content")) for m in messages)
    _hit = _PROMPT_CACHE.get(_key)
    if _hit is None:
        _prompt_str = _TOKENIZER.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            # Qwen3 defaults this to True and would leave the turn open for the
            # model to write its own <think> block. The AV was SFT'd with it
            # False throughout, so this must match (D43).
            enable_thinking=False,
        )
        # add_special_tokens=False is LOAD-BEARING for Gemma/Llama: the template
        # string already has <bos>; encoding with specials would prepend a second
        # one and shift every position, landing the injection wrong.
        _ids = _TOKENIZER.encode(_prompt_str, add_special_tokens=False)
        _ids_tensor = torch.tensor(_ids, dtype=torch.long).unsqueeze(0)  # [1, T]
        with torch.no_grad():
            _base = (_EMBED(_ids_tensor) * _EMBED_SCALE).float()  # [1, T, d]
        _hit = (_ids, _ids_tensor, _base)
        _PROMPT_CACHE[_key] = _hit
    input_ids, ids_tensor, _base_embeds = _hit
    # Clone: the caller injects into this buffer, and the cached copy must stay
    # pristine for the next sample. ~885 KB memcpy at T=108, d=2048.
    embeds = _base_embeds.clone()'''


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nla-repo", default="natural_language_autoencoders")
    args = ap.parse_args()

    path = Path(args.nla_repo) / "nla" / "rollout" / "nla_generate.py"
    assert path.is_file(), f"{path} not found — is --nla-repo correct?"
    text = path.read_text(encoding="utf-8")

    if MARKER in text:
        print(f"{path}: prompt cache already installed")
        return

    lines = text.split("\n")
    try:
        i = lines.index(START)
        j = lines.index(END)
    except ValueError:
        print(
            f"WARNING: could not locate the prep block in {path} "
            f"(START found={START in lines}, END found={END in lines}). "
            f"Upstream changed it — re-derive rather than forcing.",
            file=sys.stderr,
        )
        return
    assert i < j, f"block markers out of order in {path} ({i} >= {j})"

    out = lines[:i] + REPLACEMENT.split("\n") + lines[j + 1 :]
    # Module-level cache, declared next to the other module globals.
    anchor = "_ENGINE_URLS: list[str] | None = None"
    body = "\n".join(out)
    assert anchor in body, f"could not find the globals anchor in {path}"
    body = body.replace(
        anchor,
        "# Constant-prompt memo: key -> (input_ids, ids_tensor, base_embeds).\n"
        "# One entry in practice; bounded by the number of distinct prompts.\n"
        "_PROMPT_CACHE: dict = {}\n" + anchor,
        1,
    )
    path.write_text(body, encoding="utf-8")
    print(f"{path}: prompt work is now memoized")


if __name__ == "__main__":
    main()
