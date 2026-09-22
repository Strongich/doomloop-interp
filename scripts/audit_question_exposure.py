#!/usr/bin/env python3
"""Which GSM8K questions has this project already touched, and in which split?

The fresh-evaluation cohort for EXPERIMENT-followup-small-model.md must be drawn
from questions no earlier stage of this work has seen, AND must be reported
against a named split rather than a pooled index range.

Two facts make that non-trivial here:

1. `question_id` is `gsm8k:<i>` where `i` indexes the CONCATENATION built by
   `math_datasets.prepare_gsm8k()` -- train first, then test. So
   `0 <= i < 7473` is the official train split and `7473 <= i < 8792` is the
   official test split. An id alone does not say which.
2. Exposure is spread over ~90 artifacts written by a dozen scripts, including
   `data/traces_all.jsonl` (the whole rollout corpus). Auditing only the obvious
   cohort files understates it.

Exposure is also not one kind of thing, so this separates two tiers:

  CENSUS     `data/traces/gsm8k.jsonl` (and its `traces_all.jsonl` copy) holds
             exactly 4 rollouts for every one of the 8,792 questions. It is an
             exhaustive unselected sweep -- a census. A question appearing ONLY
             there has informed no direction, no cohort and no hyperparameter,
             so it is not contaminated in any sense that matters for a held-out
             claim. It does mean per-question difficulty was knowable in advance,
             which is how the balanced 400 cohort was drawn; a fresh sample must
             therefore be drawn WITHOUT consulting those bands.
  SELECTION  every other artifact: cohorts, direction-fitting probe sets,
             experiment results. Appearing here means the question participated
             in a choice this project made, and disqualifies it from a fresh set.

This scans every data artifact for `gsm8k:<digits>`, maps each id to its split
and tier, and reports what is left. Writes the eligible test-split ids so the
sampler can be a deterministic draw from a recorded pool rather than an ad-hoc
filter.

    uv run python scripts/audit_question_exposure.py
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# openai/gsm8k, config "main". Verified against the installed datasets copy.
N_TRAIN = 7473
N_TEST = 1319
N_TOTAL = N_TRAIN + N_TEST

ID_RE = re.compile(rb"gsm8k:(\d+)")

# Exhaustive unselected rollout corpora -- see the CENSUS tier above.
CENSUS = {"data/traces/gsm8k.jsonl", "data/traces_all.jsonl"}


def split_of(i: int) -> str:
    if i < N_TRAIN:
        return "train"
    if i < N_TOTAL:
        return "test"
    return "out_of_range"


def scan(root: Path) -> dict[int, set[str]]:
    """question index -> set of artifact paths that mention it.

    Also used for its own output file, so a re-run does not count the previous
    audit's recorded pool as fresh exposure.
    """
    seen: dict[int, set[str]] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".jsonl", ".csv", ".json", ".md", ".txt"}:
            continue
        if path.name == "exposure_audit.json":
            continue
        try:
            blob = path.read_bytes()
        except OSError:
            continue
        for m in ID_RE.finditer(blob):
            seen.setdefault(int(m.group(1)), set()).add(str(path))
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("data/exposure_audit.json"))
    args = ap.parse_args()

    seen = scan(args.data)
    selected = {i: sorted(p for p in paths if p not in CENSUS) for i, paths in seen.items()}
    selected = {i: p for i, p in selected.items() if p}

    print(f"gsm8k index space: train [0,{N_TRAIN}) test [{N_TRAIN},{N_TOTAL})")
    print(f"artifacts scanned under {args.data}/ ; census corpora: {sorted(CENSUS)}")
    print()
    print(f"{'':14s}{'census':>10s}{'selection':>12s}{'eligible':>10s}")
    pools: dict[str, list[int]] = {}
    for name, lo, hi in (("train", 0, N_TRAIN), ("test", N_TRAIN, N_TOTAL)):
        rng = set(range(lo, hi))
        cen = len(rng & set(seen))
        sel = rng & set(selected)
        free = sorted(rng - sel)
        pools[name] = free
        print(f"  {name:12s}{cen:10d}{len(sel):12d}{len(free):10d}")

    print()
    print(f"ELIGIBLE official test ids : {len(pools['test'])} of {N_TEST}")
    print(f"ELIGIBLE official train ids: {len(pools['train'])} of {N_TRAIN}")

    counts: dict[str, int] = {}
    for paths in selected.values():
        for pth in paths:
            counts[pth] = counts.get(pth, 0) + 1
    print("\ntop SELECTION artifacts by distinct ids touched:")
    for pth, n in sorted(counts.items(), key=lambda kv: -kv[1])[:12]:
        print(f"  {n:5d}  {pth}")

    args.out.write_text(
        json.dumps(
            {
                "n_train": N_TRAIN,
                "n_test": N_TEST,
                "census_corpora": sorted(CENSUS),
                "selection_touched_test": sorted(i for i in selected if split_of(i) == "test"),
                "selection_touched_train_count": sum(
                    1 for i in selected if split_of(i) == "train"
                ),
                "eligible_test": pools["test"],
                "eligible_train_count": len(pools["train"]),
            },
            indent=1,
        )
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
