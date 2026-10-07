#!/usr/bin/env python3
r"""Build data/baselines/hard128.jsonl (EXPERIMENT-reviewer-baselines.md §B4).

AIME 2025 minus aime2025:1 (N's source rollout) and aime2025:9 (Finding 11 dev cohort),
AIME 2024, AMC 23, HMMT February 2025. Every gold must self-grade under the D69 grader
(`grade("\boxed{gold}", gold)`); a gold that does not is dropped and reported. Question
ids for AIME 2025 / AMC 23 follow the project's existing 0-based scheme
(`data/distill/eval_aime_amc.jsonl`): AIME2025-I then AIME2025-II.

    uv run python scripts/baselines_build_hard.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from datasets import load_dataset  # noqa: E402

from reasoning_attention.grading import grade  # noqa: E402

OUT = Path("data/baselines/hard128.jsonl")
EXCLUDE = {"aime2025:1", "aime2025:9"}


def main() -> None:
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; the hard set is built once and locked")
    rows: list[dict] = []
    i = 0
    for subset in ("AIME2025-I", "AIME2025-II"):
        for ex in load_dataset("opencompass/AIME2025", subset, split="test"):
            rows.append({"question_id": f"aime2025:{i}", "dataset": "aime2025",
                         "question": ex["question"], "gold": str(ex["answer"]).strip(),
                         "source_id": f"{subset}#{i}"})
            i += 1
    for j, ex in enumerate(load_dataset("HuggingFaceH4/aime_2024", split="train")):
        rows.append({"question_id": f"aime2024:{j}", "dataset": "aime2024",
                     "question": ex["problem"], "gold": str(ex["answer"]).strip(),
                     "source_id": ex["url"]})
    for ex in load_dataset("math-ai/amc23", "default", split="test"):
        rows.append({"question_id": f"amc23:{ex['id']}", "dataset": "amc23",
                     "question": ex["question"], "gold": str(ex["answer"]).strip(),
                     "source_id": ex["url"]})
    for ex in load_dataset("MathArena/hmmt_feb_2025", split="train"):
        rows.append({"question_id": f"hmmt_feb2025:{ex['problem_idx']}", "dataset": "hmmt_feb2025",
                     "question": ex["problem"], "gold": str(ex["answer"]).strip(),
                     "source_id": f"problem_idx={ex['problem_idx']}"})
    counts = {}
    for r in rows:
        counts[r["dataset"]] = counts.get(r["dataset"], 0) + 1
    print("sources:", counts)
    if len({r["question_id"] for r in rows}) != len(rows):
        raise SystemExit("duplicate question ids")
    excluded = [r["question_id"] for r in rows if r["question_id"] in EXCLUDE]
    if sorted(excluded) != sorted(EXCLUDE):
        raise SystemExit(f"expected to exclude {EXCLUDE}, found {excluded}")
    keep, dropped = [], []
    for r in rows:
        if r["question_id"] in EXCLUDE:
            continue
        g = grade(f"\\boxed{{{r['gold']}}}", r["gold"])
        (keep if g.is_correct else dropped).append(r)
    print(f"excluded by construction: {excluded}")
    print(f"dropped (gold does not self-grade): {[(r['question_id'], r['gold']) for r in dropped]}")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("".join(json.dumps(r) + "\n" for r in keep))
    by = {}
    for r in keep:
        by[r["dataset"]] = by.get(r["dataset"], 0) + 1
    report = {"n": len(keep), "by_dataset": by, "excluded": excluded,
              "dropped_self_grade": [{"question_id": r["question_id"], "gold": r["gold"]}
                                     for r in dropped],
              "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}
    Path("data/baselines/hard128_build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
