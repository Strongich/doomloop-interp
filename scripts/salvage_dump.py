#!/usr/bin/env python3
"""Rebuild an arm's CSV from its texts dump, so a killed run keeps its work.

`suppress_answer.py` flushes every generated trace to its `--dump` jsonl as it
goes, but (before the resumable-partial change) only wrote the CSV at the very
end. An arm killed at 90% therefore had all its text on disk and nothing scored.

Everything the report needs is a pure function of that text: grading, doubt
counts, and — by re-tokenizing — length and whether the trace hit the cap. Only
`injections` is unrecoverable, and nothing reads it, so it is written as -1.

The output is `<out>.partial`, which is exactly the file the patched
`suppress_answer.py` resumes from: salvage an interrupted arm, re-run
`run_tier2.sh`, and it continues from the last salvaged trace instead of
restarting.

    uv run python scripts/salvage_dump.py \
        --dump data/tier2/A_1trace_texts.jsonl --out data/tier2/A_1trace.csv \
        --conditions none suppress random
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suppress_answer import doubt_stats, think_block  # noqa: E402

from reasoning_attention.config import MODEL_ID  # noqa: E402
from reasoning_attention.grading import grade  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--traces", type=Path, default=Path("data/hard_sample_heldout.jsonl"))
    p.add_argument("--conditions", nargs="+", default=["none", "suppress", "random"])
    p.add_argument("--max-new-tokens", type=int, default=12288)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = {}
    for line in args.traces.open():
        r = json.loads(line)
        meta[r["question_id"]] = r["dataset"]

    recs = []
    for line in args.dump.open():
        line = line.strip()
        if not line:
            continue
        try:
            recs.append(json.loads(line))          # tolerate a truncated last line
        except json.JSONDecodeError:
            print("skipping a truncated final record")
    print(f"{len(recs)} traces in the dump")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    rows: list[dict[str, Any]] = []
    for rec in recs:
        row: dict[str, Any] = {
            "question_id": rec["question_id"],
            "dataset": meta.get(rec["question_id"], "?"),
            "gold": rec["gold"],
        }
        for cond in args.conditions:
            text = rec[cond]
            g = grade(text, rec["gold"])
            n_mark, n_blocks, n_doubt = doubt_stats(text)
            n_tok = len(tok(text, add_special_tokens=False)["input_ids"])
            n_think = len(tok(think_block(text), add_special_tokens=False)["input_ids"])
            row.update(
                {
                    f"{cond}_correct": int(g.is_correct),
                    f"{cond}_has_answer": int(g.has_answer),
                    f"{cond}_status": g.status,
                    f"{cond}_tokens": n_tok,
                    # `capped` was "generation never emitted EOS". Re-tokenizing is
                    # within a token or two of the original count, so allow slack.
                    f"{cond}_think_tokens": n_think if "</think>" in text else 0,
                    f"{cond}_capped": int(n_tok >= args.max_new_tokens - 8),
                    f"{cond}_injections": -1,
                    f"{cond}_markers": n_mark,
                    f"{cond}_blocks": n_blocks,
                    f"{cond}_doubt_blocks": n_doubt,
                }
            )
        rows.append(row)

    partial = Path(str(args.out) + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with partial.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} rows -> {partial}")
    n = len(rows)
    for cond in args.conditions:
        c = sum(r[f"{cond}_correct"] for r in rows)
        cap = sum(r[f"{cond}_capped"] for r in rows)
        print(f"  {cond:<10} {c}/{n} = {100*c/n:4.1f}%   capped {100*cap/n:4.0f}%")


if __name__ == "__main__":
    main()
