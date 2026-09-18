#!/usr/bin/env python3
"""How much of a reasoning trace is loop?

`doubt_blocks` counts paragraphs opening with a marker word. Both the NLA
direction and difference-of-means drive it to zero, so it saturates and stops
discriminating -- while the traces plainly still loop. These four measures look
at the behaviour instead of the vocabulary, and none of them depends on a marker
list:

  rep20   fraction of 20-token windows that have appeared earlier in the trace.
          The standard repetition measure. Verbatim-ish; catches tight loops.

  dupblk  fraction of paragraphs that are near-duplicates of an earlier one
          (word-set Jaccard >= 0.6). Catches loops that reword rather than repeat.

  post    tokens generated AFTER the gold answer first appears, as a share of the
          trace. This is the most direct measure of waste: the model already had
          the answer and kept going. Only computed for traces that state it.

  gzip    compressed size / raw size. No thresholds, no tuning -- text that
          repeats itself compresses better. A sanity check on the other three.

    uv run python scripts/loop_metrics.py
"""

from __future__ import annotations

import argparse
import gzip
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_attention.loops import gold_span  # noqa: E402

SRCS = {
    "baseline": ("data/tier3/step1_A_texts.jsonl", "none"),
    "N (NLA)": ("data/dom/A_ref_texts.jsonl", "suppress"),
    "D (diff-means)": ("data/dom/D_diffmeans_texts.jsonl", "suppress"),
    "P (probe)": ("data/dom/P_probe_texts.jsonl", "suppress"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/dom_testset150.jsonl"))
    p.add_argument("--n", type=int, default=20, help="window size for rep-n")
    return p.parse_args()


def think(t: str) -> str:
    return t.split("</think>")[0]


def rep_n(words: list[str], n: int) -> float:
    if len(words) <= n:
        return 0.0
    seen: set[tuple[str, ...]] = set()
    dup = 0
    total = 0
    for i in range(len(words) - n + 1):
        w = tuple(words[i : i + n])
        total += 1
        if w in seen:
            dup += 1
        else:
            seen.add(w)
    return dup / total


def dup_blocks(body: str) -> float:
    bs = [set(re.findall(r"[a-z0-9]+", b.lower()))
          for b in body.split("\n\n") if len(b.strip()) > 40]
    if len(bs) < 2:
        return 0.0
    dup = 0
    for i, x in enumerate(bs):
        for y in bs[:i]:
            u = x | y
            if u and len(x & y) / len(u) >= 0.6:
                dup += 1
                break
    return dup / len(bs)


def post_answer(body: str, gold: str) -> float | None:
    """Share of the reasoning that happens after the gold value first appears.

    Uses the project's own `gold_span`, not a local regex: it is word-bounded so
    gold "7" does not match inside "70", and it rejects "12.5" for gold 12 while
    still accepting a sentence-final "12.".
    """
    span = gold_span(body, str(gold).strip())
    if not span:
        return None
    return 1.0 - span[0] / max(len(body), 1)


def main() -> None:
    args = parse_args()
    meta = {json.loads(x)["question_id"]: json.loads(x) for x in args.traces.open()}
    print(f"{'arm':<17}{'rep20':>9}{'dupblk':>9}{'post-ans':>10}{'gzip':>8}{'blocks':>9}")
    print("-" * 62)
    for lab, (path, key) in SRCS.items():
        r20: list[float] = []
        dbk: list[float] = []
        pa: list[float] = []
        gz: list[float] = []
        nb: list[int] = []
        for line in Path(path).open():
            r = json.loads(line)
            q = r["question_id"]
            if q not in meta:
                continue
            t = r.get(key)
            if not t:
                continue
            body = think(t)
            words = body.split()
            if len(words) < 50:
                continue
            r20.append(rep_n(words, args.n))
            dbk.append(dup_blocks(body))
            v = post_answer(body, meta[q]["gold"])
            if v is not None:
                pa.append(v)
            raw = body.encode()
            gz.append(len(gzip.compress(raw)) / max(len(raw), 1))
            nb.append(len([b for b in body.split("\n\n") if b.strip()]))
        print(f"{lab:<17}{100*st.mean(r20):>8.1f}%{100*st.mean(dbk):>8.0f}%"
              f"{100*st.mean(pa):>9.0f}%{st.mean(gz):>8.3f}{st.mean(nb):>9.1f}")
              
    print("\nlower is better on all four. `post-ans` is the share of the reasoning")
    print("that happens after the model has already written the correct value.")


if __name__ == "__main__":
    main()
