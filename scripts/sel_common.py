"""Shared definitions for EXPERIMENT-selective-doubt.md (§S3): paragraphs, doubt
paragraphs and the forced-answer probe prompt, all on the STORED token ids.

  boundary          a generated token inside <think> carrying >= 2 newlines -- the site N is
                    built and injected at (reasoning_policy_vllm boundary_set)
  paragraph k+1     the tokens after boundary b_k up to and including b_{k+1} (the last one
                    ends at </think>, or at the end of a capped trace)
  doubt paragraph   suppress_answer.doubt_stats' criterion -- the paragraph's first sentence
                    (split at [.!?] + whitespace) contains a DOUBT_MARKERS word,
                    case-insensitive -- so every count matches the `doubt_blocks` column.
                    `opens` additionally records whether the stripped paragraph STARTS with one.
  probe at b        prompt_ids + ids[:b] + tok(decode(ids[b]).rstrip() + STEM): the
                    probe_candidates.py stem, applied on token ids so the frozen prefix is
                    exactly what was generated (and shares its KV blocks).
"""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reasoning_attention.loops import DOUBT_MARKERS, marker_matches  # noqa: E402

FORMAT = "Give your final answer in \\boxed{}."
STEM = "\n</think>\n\n**Final Answer:** $\\boxed{"
NEWLINE_CHAR = "Ċ"
_SENT = re.compile(r"(?<=[.!?])\s")


class Paragraphs:
    def __init__(self, tok: Any) -> None:
        self.tok = tok
        pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
        self.boundary = {i for i, t in enumerate(pieces) if t and t.count(NEWLINE_CHAR) >= 2}
        self.close = int(tok.convert_tokens_to_ids("</think>"))

    def prompt_ids(self, question: str) -> list[int]:
        from reasoning_attention.data.math_datasets import build_messages

        m = [{"role": "system", "content": FORMAT}, *build_messages(question)]
        t = self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                         enable_thinking=True)
        return list(self.tok(t, add_special_tokens=False)["input_ids"])

    def split(self, ids: list[int]) -> dict:
        """boundaries b_0..b_{n-1}, think_end, and per paragraph k+1 (k = 0..n-1) whether it is
        a doubt paragraph. doubt[k] refers to the paragraph AFTER boundary b_k."""
        think_end = ids.index(self.close) if self.close in ids else len(ids)
        bs = [i for i in range(think_end) if ids[i] in self.boundary]
        doubt, opens = [], []
        for k, b in enumerate(bs):
            end = bs[k + 1] + 1 if k + 1 < len(bs) else think_end
            text = self.tok.decode(ids[b + 1 : end], skip_special_tokens=False).strip()
            first = _SENT.split(text, maxsplit=1)[0]
            doubt.append(any(marker_matches(first, m, case_sensitive=False) for m in DOUBT_MARKERS))
            opens.append(any(text.lower().startswith(m.lower()) for m in DOUBT_MARKERS))
        return {"boundaries": bs, "think_end": think_end, "doubt": doubt, "opens": opens}

    def probe_ids(self, prompt: list[int], ids: list[int], cut: int) -> list[int]:
        """Freeze the trace through token `cut` (inclusive) and force the answer box."""
        tail = self.tok.decode([ids[cut]], skip_special_tokens=False).rstrip() + STEM
        return prompt + ids[:cut] + list(self.tok(tail, add_special_tokens=False)["input_ids"])
