#!/usr/bin/env python3
"""Does suppression degrade as injections accumulate within a trace?

arXiv:2605.10664 names a failure mode for residual-stream steering in
autoregressive generation: a steered state is written into the KV cache and
re-attended by every later token, so a local perturbation **compounds**. Their
baseline collapses to near-zero coherence by turn 5.

Finding 5 injects ~36 times in one generation and has never been checked on that
axis. Per-injection harm is not the right unit; accumulation is.

The test needs no GPU — the full generated text of all three conditions is already
on disk. Bin each trace's blocks by their index k, and compare how the three
conditions evolve with k:

  * block length      — contamination predicts the suppressed arm degrading
  * repetition        — its symptom is incoherence and looping, not clean brevity
  * doubt-opening rate — does suppression stop working deeper into a trace?

The `random` arm is the control that makes this decisive: it perturbs and
contaminates the cache identically, at the same sites with matched norm. If
degradation is contamination-as-such, `random` degrades too. If it is the
direction, only `suppress` moves.

    uv run python scripts/contamination_curve.py
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_attention.loops import DOUBT_MARKERS  # noqa: E402

CONDITIONS = ("none", "suppress", "random")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--texts", type=Path, default=Path("data/suppress_answer_texts_a1.0.jsonl"))
    p.add_argument("--bins", default="1-5,6-10,11-15,16-20,21-30,31-50")
    p.add_argument("--out", type=Path, default=Path("data/contamination_curve.csv"))
    return p.parse_args()


def think_blocks(text: str) -> list[str]:
    body = text.split("</think>")[0].split("<think>")[-1]
    return [b.strip() for b in body.split("\n\n") if b.strip()]


def repetition(text: str, n: int = 3) -> float:
    """Fraction of character n-grams that are repeats. 0 = no repetition."""
    grams = [text[i : i + n] for i in range(max(0, len(text) - n + 1))]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def opens_with_doubt(block: str) -> bool:
    head = block.lstrip()
    return any(head.lower().startswith(m.lower()) for m in DOUBT_MARKERS)


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.texts.open()]
    print(f"{len(rows)} traces x {len(CONDITIONS)} conditions\n")

    bins: list[tuple[int, int]] = []
    for part in args.bins.split(","):
        lo, hi = part.split("-")
        bins.append((int(lo), int(hi)))

    # per (condition, bin): block lengths, repetition scores, doubt flags
    acc: dict[tuple[str, int], dict[str, list[float]]] = defaultdict(
        lambda: {"len": [], "rep": [], "doubt": []}
    )
    for r in rows:
        for cond in CONDITIONS:
            blocks = think_blocks(r[cond])
            for k, b in enumerate(blocks, 1):
                bi = next((i for i, (lo, hi) in enumerate(bins) if lo <= k <= hi), None)
                if bi is None:
                    continue
                cell = acc[(cond, bi)]
                cell["len"].append(len(b))
                cell["rep"].append(repetition(b))
                cell["doubt"].append(1.0 if opens_with_doubt(b) else 0.0)

    hdr = f"{'block index':<14}" + "".join(f"{c:>26}" for c in CONDITIONS)
    print("MEAN BLOCK LENGTH (chars) | REPETITION | DOUBT-OPENING RATE")
    print(hdr)
    print("-" * len(hdr))
    out_rows = []
    for bi, (lo, hi) in enumerate(bins):
        cells = []
        for cond in CONDITIONS:
            c = acc[(cond, bi)]
            if not c["len"]:
                cells.append(f"{'--':>26}")
                continue
            cells.append(
                f"{st.mean(c['len']):>8.0f} {st.mean(c['rep']):>7.3f} {st.mean(c['doubt']):>8.2f}"
            )
            out_rows.append(
                {
                    "bin": f"{lo}-{hi}",
                    "condition": cond,
                    "n_blocks": len(c["len"]),
                    "mean_len": round(st.mean(c["len"]), 1),
                    "mean_repetition": round(st.mean(c["rep"]), 4),
                    "doubt_rate": round(st.mean(c["doubt"]), 4),
                }
            )
        print(f"{f'{lo}-{hi}':<14}" + "".join(cells))
    print("\n(each cell: length  repetition  doubt-rate)")

    import csv

    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0]))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {args.out}")

    # The decisive read: does the suppress-vs-none gap widen or close with depth?
    print("\nSUPPRESSION EFFICACY BY DEPTH (doubt rate, suppress vs none)")
    for bi, (lo, hi) in enumerate(bins):
        n_, s_, r_ = (acc[(c, bi)]["doubt"] for c in CONDITIONS)
        if not n_ or not s_:
            continue
        print(
            f"  blocks {f'{lo}-{hi}':<8} none {st.mean(n_):.3f}  "
            f"suppress {st.mean(s_):.3f}  random {st.mean(r_):.3f}  "
            f"=> removed {100 * (1 - st.mean(s_) / max(st.mean(n_), 1e-9)):.0f}%"
        )


if __name__ == "__main__":
    main()
