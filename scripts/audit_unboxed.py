r"""Arm-blind sensitivity analysis for outputs that never emitted a `\boxed{}`.

The locked metric counts only a boxed answer, and it stays the primary result.
But `brevityA` produced 74 unboxed outputs against 6-8 for every other arm, and
under boxed-only grading each of those scores wrong however good the reasoning
was. Its entire 5.6 pp deficit is inside that gap, so the headline "prompting for
brevity costs accuracy" is not separable from "prompting for brevity breaks the
answer format" without looking.

This applies ONE rule to EVERY arm's unboxed outputs, never only to the arm that
has a problem -- an extractor tuned on the deficient arm would manufacture the
result it was built to test. It reports the primary metric unchanged alongside
the sensitivity metric, and the bracket between them.

The rule is deliberately strict. It fires only when the tail of the output states
a final answer with an explicit cue, and it takes the LAST such statement. A
looser "find any trailing number" rule would rescue outputs that merely stopped
mid-calculation, inflating every arm and flattering whichever one rambles most.
Outputs that state nothing decidable stay wrong, which is the conservative
direction.

    uv run python scripts/audit_unboxed.py --run data/reasoning_policy_v1/stage2
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import re
import statistics as st
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# An explicit final-answer cue, then the value. Anchored to the output's tail.
CUES = re.compile(
    r"(?:final answer|the answer is|answer:|answer is|therefore,?|thus,?|so,?)\s*"
    r"[:\-=]?\s*\**\s*\$?(-?[\d,]+(?:\.\d+)?)\b",
    re.IGNORECASE,
)
# A bare "**42**" or "= 42" ending the text also counts as a stated answer.
TERMINAL = re.compile(r"(?:\*\*|=)\s*\$?(-?[\d,]+(?:\.\d+)?)\s*\**\s*\.?\s*$")


def extract(text: str, tail_chars: int = 400) -> str | None:
    """The stated final answer in the output's tail, or None if undecidable."""
    tail = text[-tail_chars:]
    hits = CUES.findall(tail)
    if hits:
        return hits[-1].replace(",", "")
    m = TERMINAL.search(tail.strip())
    return m.group(1).replace(",", "") if m else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from reasoning_attention.grading import grade

    journals = sorted(args.run.parent.glob(f"{args.run.name}_shard*/rollouts.jsonl"))
    if not journals:
        journals = [args.run / "rollouts.jsonl"]
    rows = [json.loads(line) for p in journals for line in p.open()]
    # The merged analysis set defines which rows count; shard journals hold text.
    keep = {
        (r["question_id"], r["policy"], int(r["seed"]))
        for r in csv.DictReader((args.run / "rollouts.csv").open())
    }
    rows = [r for r in rows if (r["question_id"], r["policy"], int(r["seed"])) in keep]
    seen: set[tuple] = set()
    uniq = []
    for r in rows:
        k = (r["question_id"], r["policy"], int(r["seed"]))
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    rows = uniq

    per = collections.defaultdict(lambda: {"n": 0, "boxed_ok": 0, "unboxed": 0,
                                           "rescued": 0, "undecidable": 0})
    rescued_rows = []
    for r in rows:
        s = per[r["policy"]]
        s["n"] += 1
        s["boxed_ok"] += int(r["correct"])
        if r["has_answer"]:
            continue
        s["unboxed"] += 1
        stated = extract(r["text"])
        if stated is None:
            s["undecidable"] += 1
            continue
        g = grade("\\boxed{" + stated + "}", r["gold"])
        if g.is_correct:
            s["rescued"] += 1
            rescued_rows.append({"question_id": r["question_id"], "policy": r["policy"],
                                 "seed": r["seed"], "gold": r["gold"], "stated": stated})

    base = per["base"]
    b_primary = 100 * base["boxed_ok"] / base["n"]
    b_sens = 100 * (base["boxed_ok"] + base["rescued"]) / base["n"]
    print(f"{'policy':14s}{'n':>5s}{'unboxed':>9s}{'rescued':>9s}{'undecid':>9s}"
          f"{'primary':>9s}{'sens':>8s}{'d_prim':>8s}{'d_sens':>8s}")
    for p in sorted(per):
        s = per[p]
        prim = 100 * s["boxed_ok"] / s["n"]
        sens = 100 * (s["boxed_ok"] + s["rescued"]) / s["n"]
        print(f"{p:14s}{s['n']:5d}{s['unboxed']:9d}{s['rescued']:9d}{s['undecidable']:9d}"
              f"{prim:9.1f}{sens:8.1f}{prim - b_primary:+8.1f}{sens - b_sens:+8.1f}")
    print("\nprimary = the locked boxed-only metric, unchanged and authoritative.")
    print("sens    = same rule applied to every arm's unboxed outputs.")
    print("The bracket between the two columns is the honest range for any arm")
    print("whose unboxed rate differs materially from the baseline's.")

    if args.out:
        args.out.write_text(json.dumps(rescued_rows, indent=1))
        print(f"\nwrote {len(rescued_rows)} rescued outputs to {args.out}")


if __name__ == "__main__":
    main()
