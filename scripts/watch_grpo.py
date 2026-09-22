#!/usr/bin/env python3
"""Tail a miles GRPO log as a metrics table.

miles has no wandb wired for these runs, so the log file is the only record.
It interleaves two per-rollout dict lines that must be joined on the step id:

    log_utils.py:52  - rollout N: {'rollout/raw_reward': ..., ...}
    log_utils.py:429 - step    N: {'train/fve_nrm': ..., ...}
    train_metric_utils.py:44 - perf N: {'perf/step_time': ..., ...}

The actor and the critic BOTH emit `step N:` under the same key names, so a
blind merge lets the actor's `train/loss` (pg_loss, ~0.01 and meaningless on
policy) overwrite the critic's reconstruction MSE. The critic's line is the one
carrying `train/fve_nrm`; its loss is re-filed as `critic/loss`.

Usage:
    python3 watch_grpo.py /workspace/data/grpo_probe.log          # table so far
    python3 watch_grpo.py /workspace/data/grpo_probe.log -f       # follow
"""

from __future__ import annotations

import argparse
import ast
import re
import sys
import time

ANSI = re.compile(r"\x1b\[[0-9;]*m")
LINE = re.compile(r"(rollout|step|perf) (\d+): (\{.*\})\s*$")

# (header, dict key, format). fve_nrm is the headline reconstruction metric;
# raw_reward is -log||h - AR(z)||^2, so it rises toward 0.
COLS = [
    ("reward", "rollout/raw_reward", "{:>8.3f}"),
    ("fve_nrm", "train/fve_nrm", "{:>8.3f}"),
    ("mse", "critic/loss", "{:>7.3f}"),
    ("grad", "train/grad_norm", "{:>6.2f}"),
    # train<->rollout logprob mismatch. ~0.001 healthy; ~0.20 means the rollout
    # transport is corrupting the injected embeddings (TRAINING_NOTES.md:277).
    # Only present with --get-mismatch-metrics (TIS_METRICS=1).
    ("tis_k3", "train/tis_k3", "{:>8.4f}"),
    ("resp_len", "rollout/total_lengths", "{:>8.1f}"),
    ("trunc", "rollout/truncated", "{:>6.2f}"),
    ("lr", "train/lr-pg_0", "{:>9.2e}"),
    ("sec", "perf/step_time", "{:>6.0f}"),
]


def parse(path: str, rows: dict[int, dict[str, float]]) -> None:
    with open(path, encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            m = LINE.search(ANSI.sub("", raw.rstrip()))
            if not m:
                continue
            try:
                payload = ast.literal_eval(m.group(3))
            except (ValueError, SyntaxError):
                continue
            if not isinstance(payload, dict):
                continue
            if "train/fve_nrm" in payload and "train/loss" in payload:
                payload = dict(payload, **{"critic/loss": payload["train/loss"]})
            rows.setdefault(int(m.group(2)), {}).update(payload)


def render(rows: dict[int, dict[str, float]], since: int) -> int:
    for step in sorted(k for k in rows if k > since):
        row = rows[step]
        cells = [
            fmt.format(row[key]) if isinstance(row.get(key), (int, float)) else " " * 8
            for _, key, fmt in COLS
        ]
        print(f"{step:>5} " + " ".join(cells), flush=True)
        since = step
    return since


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("-f", "--follow", action="store_true")
    ap.add_argument("-n", type=int, default=0, help="show only the last N steps")
    args = ap.parse_args()

    print("step  " + " ".join(f"{h:>8}" for h, _, _ in COLS), file=sys.stderr)
    rows: dict[int, dict[str, float]] = {}
    parse(args.log, rows)
    start = max(rows) - args.n if args.n and rows else -1
    seen = render(rows, start)
    while args.follow:
        time.sleep(20)
        parse(args.log, rows)
        seen = render(rows, seen)


if __name__ == "__main__":
    main()
