#!/usr/bin/env python3
r"""V_ban for the decode-time doubt-word ban (EXPERIMENT-reviewer-baselines.md §B2).

Every vocabulary id whose decoded string, leading whitespace stripped, equals one of the
single-word entries of `loops.DOUBT_MARKERS` (plus "Hold" from "Hold on") in any letter
case, optionally followed only by ASCII punctuation (`Wait,`). Built from the Qwen3-1.7B
tokenizer and checked identical on every target tokenizer present in the HF cache.

The file also carries what the worker-side logits processor needs to place the matched
ban: the paragraph-boundary ids (tokens with >= 2 newlines, the same set the steering
hook uses) and `</think>`.

    uv run python scripts/baselines_ban_tokens.py
"""

from __future__ import annotations

import json
import re
import string
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from transformers import AutoTokenizer  # noqa: E402

from reasoning_attention.loops import DOUBT_MARKERS  # noqa: E402

OUT = Path("data/baselines/ban_tokens.json")
TARGETS = ("Qwen/Qwen3-8B", "Qwen/Qwen3-30B-A3B", "Qwen/Qwen3-235B-A22B-FP8")


def ban_words() -> list[str]:
    words = [m for m in DOUBT_MARKERS if " " not in m]
    words.append("Hold")  # "Hold on" opens with it
    return words


def build(tok) -> tuple[list[dict], list[int]]:
    from suppress_answer import NEWLINE_CHAR

    words = {w.lower() for w in ban_words()}
    punct = re.escape(string.punctuation)
    pat = re.compile(rf"^(?P<w>[A-Za-z]+)[{punct}]*$")
    ban = []
    boundaries = []
    for i in range(len(tok)):
        if tok.convert_ids_to_tokens(i).count(NEWLINE_CHAR) >= 2:
            boundaries.append(i)
        s = tok.decode([i])
        m = pat.match(s.lstrip())
        if m and m.group("w").lower() in words:
            ban.append({"id": i, "decoded": s, "token": tok.convert_ids_to_tokens(i)})
    return ban, boundaries


def main() -> None:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    ban, boundaries = build(tok)
    close_id = int(tok.convert_tokens_to_ids("</think>"))
    checked = []
    for name in TARGETS:
        try:
            other = AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception as e:  # not cached: nothing to compare
            print(f"skip {name}: {type(e).__name__}")
            continue
        b2, bd2 = build(other)
        if [x["id"] for x in b2] != [x["id"] for x in ban] or bd2 != boundaries or int(
            other.convert_tokens_to_ids("</think>")
        ) != close_id:
            raise SystemExit(f"{name}: ban/boundary/close ids differ from Qwen3-1.7B")
        checked.append(name)
    by_word: dict[str, int] = {}
    pat = re.compile(r"[A-Za-z]+")
    for x in ban:
        w = pat.search(x["decoded"]).group(0).lower()
        by_word[w] = by_word.get(w, 0) + 1
    out = {
        "words": ban_words(),
        "rule": "decoded.lstrip() == word (any case), optionally followed only by "
                "ASCII punctuation",
        "tokenizer": "Qwen/Qwen3-1.7B",
        "identical_on": checked,
        "size": len(ban),
        "by_word": by_word,
        "ban": ban,
        "close_id": close_id,
        "boundary_ids": boundaries,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"|V_ban| = {len(ban)}  by word {by_word}; boundaries {len(boundaries)}; "
          f"identical on {checked}")
    for x in ban:
        print(f"  {x['id']:>7} {x['decoded']!r}")


if __name__ == "__main__":
    main()
