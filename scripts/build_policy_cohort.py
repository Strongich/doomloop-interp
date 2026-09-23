#!/usr/bin/env python3
"""Cohort files for the reasoning-policy sweep: development now, fresh later.

Development reuses the existing 400-question Finding 9 cohort unchanged, so the
delay/strength sweep is measured on the same questions as Finding 10's
diagnostics. That cohort is difficulty-band balanced (100 each of all_right,
mostly_wrong, mixed, all_wrong) and is therefore NOT the natural GSM8K
distribution -- its base accuracy near 53% is a property of how it was built.
Every accuracy quoted from development inherits that and must say so.

Stage 1 of the two-stage sweep screens every configuration on a 200-question
subset. The subset is drawn round-robin within band after a seeded shuffle, so
it keeps the 50/50/50/50 balance rather than sampling it and hoping. Screening
on an unbalanced subset would confound configuration with difficulty mixture.

`--fresh` draws the confirmatory cohort instead: a seeded sample from the
selection-free official GSM8K **test** ids recorded by
`scripts/audit_question_exposure.py`. Two rules it enforces that a one-line
`random.sample` would not:

  * test split only -- pooling the 5,971 eligible train ids would forfeit the
    right to call the result GSM8K test accuracy;
  * no consultation of difficulty. The census corpus means every question's band
    is knowable in advance, so drawing a "balanced" or "hard" fresh set would
    quietly reintroduce the selection the fresh set exists to avoid.

`--math500` writes the generalization set: all 500 HuggingFaceH4/MATH-500
problems, loaded directly rather than through `math_datasets.load_one`, because
the common schema drops `subject` and `level` and the report needs both. It also
writes a budget-check subset, 10 per level, seeded. That subset runs the
UNTREATED baseline only, to see whether the 16,384-token budget caps harder
problems. The budget is arm-blind -- every arm gets the same one -- so choosing it
from untreated lengths does not favour any policy; the subset stays in the
evaluation set and its use is disclosed.

    uv run python scripts/build_policy_cohort.py
    uv run python scripts/build_policy_cohort.py --fresh 300 --seed 20260922
    uv run python scripts/build_policy_cohort.py --math500
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path


def write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    bands = collections.Counter(r.get("band", "-") for r in rows)
    print(f"wrote {path}  n={len(rows)}  bands={dict(bands)}")


def development(out: Path, subset: int, seed: int) -> None:
    src = Path("data/after_answer_400.jsonl")
    rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
    keep = [
        {
            "question_id": r["question_id"],
            "dataset": r["dataset"],
            "question": r["question"],
            "gold": str(r["gold"]),
            "band": r["band"],
            "hist_acc": r.get("hist_acc"),
        }
        for r in rows
    ]
    # The cohort is kept exactly as Finding 9 built it, including its single
    # AIME question. Dropping that would be a silent deviation from "the
    # existing 400-question cohort" for a 1-in-400 change.
    write(out / "dev400.jsonl", keep)

    by_band: dict[str, list[dict]] = collections.defaultdict(list)
    for r in keep:
        by_band[r["band"]].append(r)
    rng = random.Random(seed)
    for band in by_band.values():
        rng.shuffle(band)
    picked: list[dict] = []
    order = sorted(by_band)
    i = 0
    while len(picked) < subset:
        band = by_band[order[i % len(order)]]
        if band:
            picked.append(band.pop())
        elif all(not by_band[b] for b in order):
            break
        i += 1
    picked.sort(key=lambda r: r["question_id"])
    write(out / f"dev{subset}.jsonl", picked)


def fresh(out: Path, n: int, seed: int) -> None:
    import sys

    sys.path.insert(0, "src")
    from reasoning_attention.data.math_datasets import load_one

    audit = json.loads(Path("data/exposure_audit.json").read_text())
    eligible = audit["eligible_test"]
    if n > len(eligible):
        raise ValueError(f"Asked for {n} but only {len(eligible)} selection-free test ids exist")
    rng = random.Random(seed)
    picked = sorted(rng.sample(eligible, n))
    ds = load_one("gsm8k")
    rows = [
        {
            "question_id": f"gsm8k:{i}",
            "dataset": "gsm8k",
            "question": ds[i]["question"],
            "gold": str(ds[i]["answer"]),
            "split": ds[i]["split"],
        }
        for i in picked
    ]
    if any(r["split"] != "test" for r in rows):
        raise ValueError("Fresh cohort must be official test split only")
    write(out / f"fresh_gsm8k_test_{n}.jsonl", rows)
    print(f"  drawn from {len(eligible)} selection-free test ids with seed {seed}")


def math500(out: Path, seed: int, per_level: int = 10) -> None:
    from datasets import load_dataset

    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    rows = [
        {
            "question_id": f"math500:{i}",
            "dataset": "math500",
            "question": r["problem"],
            "gold": str(r["answer"]),
            "subject": r["subject"],
            "level": int(r["level"]),
            "unique_id": r["unique_id"],
        }
        for i, r in enumerate(ds)
    ]
    if len(rows) != 500 or len({r["question_id"] for r in rows}) != 500:
        raise ValueError(f"expected 500 unique MATH-500 rows, got {len(rows)}")
    write(out / "math500.jsonl", rows)
    by_level: dict[int, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_level[r["level"]].append(r)
    rng = random.Random(seed)
    subset = [r for lv in sorted(by_level) for r in rng.sample(by_level[lv], per_level)]
    subset.sort(key=lambda r: int(r["question_id"].split(":")[1]))
    write(out / f"math500_budget{len(subset)}.jsonl", subset)
    levels = collections.Counter(r["level"] for r in subset)
    print(f"  budget subset by level: {dict(sorted(levels.items()))}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/policy"))
    ap.add_argument("--subset", type=int, default=200, help="stage-1 screening size")
    ap.add_argument("--seed", type=int, default=20260922)
    ap.add_argument("--fresh", type=int, default=0, help="draw the confirmatory cohort instead")
    ap.add_argument("--math500", action="store_true", help="write the MATH-500 evaluation set")
    args = ap.parse_args()
    if args.math500:
        math500(args.out, args.seed)
    elif args.fresh:
        fresh(args.out, args.fresh, args.seed)
    else:
        development(args.out, args.subset, args.seed)


if __name__ == "__main__":
    main()
