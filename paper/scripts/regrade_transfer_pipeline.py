#!/usr/bin/env python3
"""Recompute the transfer pipeline table from existing D69-regraded CSVs.

This script does not generate or grade model outputs. It reads the saved corrected
``correct`` flags. Run from any directory:

    uv run python paper/scripts/regrade_transfer_pipeline.py
    uv run python paper/scripts/regrade_transfer_pipeline.py --compare-historical

The historical mode additionally reads the original CSVs and reproduces Finding
15's pre-D69 table. Each policy spends two rollouts (seeds 0/1) first, then two
base rollouts (seeds 2/3) only if both initial answers are wrong. The shortest
correct trace among the rollouts actually spent is accepted. Generation cost
includes unsuccessful questions and is divided by the number accepted.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
MODELS = (
    (
        "1.7B",
        "N@a1.0d256",
        ("xfer8b/r17_confirm_h0", "xfer8b/r17_confirm_h1"),
    ),
    (
        "8B",
        "NU26@a0.75d256",
        ("xfer8b/r8b_confirm_h0", "xfer8b/r8b_confirm_h1", "xfer8b/r8b_confirm_L26"),
    ),
    (
        "30B-A3B",
        "NU34@a0.75d256",
        ("xfer30b/r30_confirm_base", "xfer30b/r30_confirm_arm"),
    ),
)
Row = tuple[int, int]  # correct, total generated tokens


@dataclass(frozen=True)
class Result:
    questions: int
    accepted: int
    accepted_tokens: int
    spent_tokens: int

    @property
    def coverage(self) -> float:
        return 100 * self.accepted / self.questions

    @property
    def mean_accepted_length(self) -> float:
        return self.accepted_tokens / self.accepted

    @property
    def tokens_per_accepted(self) -> float:
        return self.spent_tokens / self.accepted


def load_rows(root: Path, directories: tuple[str, ...], arm: str) -> dict:
    rows: dict[str, dict[tuple[str, int], Row]] = defaultdict(dict)
    for directory in directories:
        path = root / directory / "rollouts.csv"
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row["policy"] not in ("base", arm):
                    continue
                key = (row["policy"], int(row["seed"]))
                value = (int(row["correct"]), int(row["total_tokens"]))
                assert value[0] in (0, 1) and value[1] > 0, (path, row)
                question = rows[row["question_id"]]
                if key in question:
                    assert question[key] == value, (path, row["question_id"], key)
                question[key] = value
    expected = {(policy, seed) for policy in ("base", arm) for seed in range(4)}
    assert len(rows) == 400, (root, arm, len(rows))
    for question_id, question in rows.items():
        assert question.keys() == expected, (question_id, question.keys() ^ expected)
    return dict(rows)


def simulate(rows: dict, first_policy: str) -> Result:
    accepted_lengths: list[int] = []
    total_spent = 0
    for question in rows.values():
        used = [question[(first_policy, seed)] for seed in (0, 1)]
        if not any(correct for correct, _ in used):
            used += [question[("base", seed)] for seed in (2, 3)]
        total_spent += sum(tokens for _, tokens in used)
        correct_lengths = [tokens for correct, tokens in used if correct]
        if correct_lengths:
            accepted_lengths.append(min(correct_lengths))
    return Result(len(rows), len(accepted_lengths), sum(accepted_lengths), total_spent)


def report(root: Path, title: str) -> None:
    print(f"\n{title}: {root}")
    print("model | accepted base/pipeline | coverage % | accepted length | cost/accepted | cost saved %")
    for model, arm, directories in MODELS:
        rows = load_rows(root, directories, arm)
        base, pipeline = simulate(rows, "base"), simulate(rows, arm)
        saved = 100 * (1 - pipeline.tokens_per_accepted / base.tokens_per_accepted)
        print(
            f"{model} | {base.accepted}/{pipeline.accepted} of {base.questions} | "
            f"{base.coverage:.2f} -> {pipeline.coverage:.2f} | "
            f"{base.mean_accepted_length:.3f} -> {pipeline.mean_accepted_length:.3f} | "
            f"{base.tokens_per_accepted:.3f} -> {pipeline.tokens_per_accepted:.3f} | "
            f"{saved:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare-historical",
        action="store_true",
        help="also reproduce the original pre-D69 pipeline table",
    )
    args = parser.parse_args()
    report(REPO_ROOT / "data/regraded/data", "Corrected D69 grading")
    if args.compare_historical:
        report(REPO_ROOT / "data", "Historical grading")


if __name__ == "__main__":
    main()
