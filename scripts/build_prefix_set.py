#!/usr/bin/env python3
"""Freeze each baseline trace at the boundary where the model has its answer.

Reads `probe_candidates.py`'s forced-exit probes and picks the detection point,
then writes the exact prefix text every arm of the branching experiment will
share, plus a human-readable audit file.

## Which detection rule, and why

The probes give, at every paragraph boundary, the answer the model would commit
to if it stopped there. A candidate is "settled" once that answer stops moving.
How long it has to hold still is the one free parameter, and it is set by a
validation signal that never touches the experiment: on `all_right` questions --
ones the model solved in all four historical rollouts -- a correctly detected
candidate should equal the gold answer. Measured over 400 traces:

    rule        coverage   freeze @   all_right candidate correct
    run >= 2      100%       119 tok        48%
    run >= 3       98%       237 tok        73%
    run >= 4       96%       321 tok        88%
    run >= 6       84%       382 tok        91%
    terminal       90%       518 tok        98%

`run >= 4` is the default, at 96% coverage.

**Where the freeze lands is a separate choice from when the rule fires, and
getting it wrong costs the online claim.** A run of four agreements is only
*known* at the fourth probe. Freezing at the first one -- which Finding 10 did --
uses the three later probes to select a point before them, so the prefix is
chosen with hindsight. `run4` therefore freezes at the fourth agreeing probe
(median 546 think tokens); `run4@first` reproduces the retrospective version
(median 321). The detected candidate is identical either way, in 385 of 385
questions, since agreement is what defines the run -- so the validation column
above is unchanged and only the freeze position moves.

The `terminal` rule scores better still but needs to see the whole trace, so it
could not be deployed; its boundary is recorded alongside for a sensitivity
check that needs no new probing.

## Ordering

Rows are emitted round-robin over the (band x candidate-correct) cells, so
`--limit N` on the generation script yields a balanced sample for any N rather
than all of one cell.

    uv run python scripts/build_prefix_set.py --rule run4
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_attention.config import MODEL_ID  # noqa: E402

BANDS = ["all_right", "mixed", "mostly_wrong", "all_wrong"]


def runs(probes: list[dict[str, Any]]) -> list[tuple[int, int]]:
    """(start index, length) for each maximal run of one repeated candidate."""
    out: list[tuple[int, int]] = []
    i = 0
    while i < len(probes):
        j = i
        while (j + 1 < len(probes) and probes[j + 1]["candidate"]
               and probes[j + 1]["candidate"] == probes[i]["candidate"]):
            j += 1
        if probes[i]["candidate"]:
            out.append((i, j - i + 1))
        i = j + 1
    return out


def pick(probes: list[dict[str, Any]], rule: str) -> int | None:
    """Index of the boundary to freeze at, or None if the rule never fires.

    `run4` freezes at the FOURTH agreeing probe -- the first boundary at which
    four agreements have actually been observed. `run4@first` freezes at the
    first probe of that run, which is what Finding 10 shipped: it needs the
    three later probes to know the run exists, so the freeze point is chosen
    with hindsight and cannot be computed online. Both are kept so the old
    result stays reproducible; `run4` is the deployable one.
    """
    rs = runs(probes)
    if rule.startswith("run"):
        spec = rule[3:]
        at_first = spec.endswith("@first")
        k = int(spec.removesuffix("@first"))
        hit = next(((a, n) for a, n in rs if n >= k), None)
        if hit is None:
            return None
        return hit[0] if at_first else hit[0] + k - 1
    if rule == "terminal":  # the candidate never abandoned inside the probe window
        if rs and rs[-1][0] + rs[-1][1] == len(probes) and rs[-1][1] >= 2:
            return rs[-1][0]
        return None
    raise SystemExit(f"unknown rule {rule}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--candidates", type=Path, default=Path("data/prefix/candidates.jsonl"))
    p.add_argument("--traces", type=Path, default=Path("data/after_answer_400.jsonl"))
    p.add_argument("--texts", type=Path, default=Path("data/afteranswer/baseline_texts.jsonl"))
    p.add_argument("--key", default="none")
    p.add_argument("--out", type=Path, default=Path("data/prefix/prefixes.jsonl"))
    p.add_argument("--audit", type=Path, default=Path("data/prefix/audit.md"))
    p.add_argument("--rule", default="run4",
                   help="run4 = freeze at the 4th agreeing probe (online); "
                        "run4@first = at the 1st (what Finding 10 used)")
    p.add_argument("--audit-n", type=int, default=60)
    # The matched non-answer control RESEARCH-DIRECTION asks for: without it,
    # "steering after the answer works" cannot be told apart from "steering late
    # in a trace works".
    p.add_argument("--shuffle-freeze", action="store_true",
                   help="freeze at a DEPTH-MATCHED boundary taken from another "
                        "question, so freeze depth keeps its distribution but "
                        "stops coinciding with where the candidate settles")
    p.add_argument("--shuffle-seed", type=int, default=0)
    p.add_argument("--min-shift", type=int, default=2,
                   help="boundaries the control freeze must differ by")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    meta = {t["question_id"]: t for t in (json.loads(x) for x in args.traces.open())}
    body_of = {}
    for x in args.texts.open():
        r = json.loads(x)
        body_of[r["question_id"]] = r[args.key].split("</think>")[0]

    cells: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    skipped = 0
    for line in args.candidates.open():
        c = json.loads(line)
        idx = pick(c["probes"], args.rule)
        if idx is None:
            skipped += 1
            continue
        p = c["probes"][idx]
        term = pick(c["probes"], "terminal")
        ids = tok(body_of[c["question_id"]], add_special_tokens=False)["input_ids"]
        # +1: the boundary token itself belongs to the prefix, so every arm
        # resumes from the same "start of a fresh paragraph" state.
        prefix = tok.decode(ids[: p["boundary"] + 1], skip_special_tokens=False)
        cells[(c["band"], p["correct"])].append({
            "question_id": c["question_id"], "dataset": c["dataset"],
            "gold": c["gold"], "band": c["band"],
            "question": meta[c["question_id"]]["question"],
            "cand": p["candidate"], "cand_correct": p["correct"],
            "freeze_at": p["boundary"], "think_tokens": c["think_tokens"],
            "terminal_at": c["probes"][term]["boundary"] if term is not None else -1,
            "prefix": prefix,
        })

    if args.shuffle_freeze:
        pool = [r for c in cells.values() for r in c]
        rng = random.Random(args.shuffle_seed)
        depths = [r["freeze_at"] for r in pool]
        rng.shuffle(depths)
        by_id = {r["question_id"]: r for r in pool}
        probes_of = {json.loads(x)["question_id"]: json.loads(x)["probes"]
                     for x in args.candidates.open()}
        cells = defaultdict(list)
        n_shift = 0
        for r, want in zip(pool, depths):
            pr = probes_of[r["question_id"]]
            own = next(i for i, p in enumerate(pr) if p["boundary"] == r["freeze_at"])
            # Nearest boundary to the partner's depth, at least --min-shift
            # boundaries away from this question's own detection point.
            opts = [i for i in range(len(pr)) if abs(i - own) >= args.min_shift]
            if not opts:
                continue
            i2 = min(opts, key=lambda i: abs(pr[i]["boundary"] - want))
            p2 = pr[i2]
            ids = tok(body_of[r["question_id"]], add_special_tokens=False)["input_ids"]
            n_shift += 1
            cells[(r["band"], p2["correct"])].append({
                **r,
                "cand": p2["candidate"], "cand_correct": p2["correct"],
                "freeze_at": p2["boundary"], "shifted_from": r["freeze_at"],
                "prefix": tok.decode(ids[: p2["boundary"] + 1], skip_special_tokens=False),
            })
        print(f"shuffled freeze: {n_shift} of {len(pool)} questions kept")
        del by_id

    order = [(b, k) for b in BANDS for k in (1, 0)]
    rows: list[dict[str, Any]] = []
    i = 0
    while any(cells[c] for c in order):
        cell = order[i % len(order)]
        if cells[cell]:
            rows.append(cells[cell].pop(0))
        i += 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    if not rows:
        raise SystemExit(f"rule={args.rule}: no prefix survived detection "
                         f"({skipped} undetected) -- nothing to write")
    n_bad = sum(1 for r in rows if not r["cand_correct"])
    print(f"rule={args.rule}: {len(rows)} prefixes, {skipped} undetected")
    print(f"  incorrect candidates: {n_bad} ({n_bad / len(rows):.0%})")
    for b in BANDS:
        ok = sum(1 for r in rows if r["band"] == b and r["cand_correct"])
        bad = sum(1 for r in rows if r["band"] == b and not r["cand_correct"])
        print(f"  {b:13s} correct {ok:3d}  incorrect {bad:3d}")
    fz = sorted(r["freeze_at"] for r in rows)
    print(f"  freeze token: median {fz[len(fz) // 2]}  p90 {fz[int(len(fz) * 0.9)]}")

    # The audit the plan asks for: read the last 400 characters before each
    # freeze and judge whether the model really had stated that answer.
    with args.audit.open("w") as fh:
        fh.write(f"# Freeze-point audit ({args.rule})\n\nRead the tail of the prefix and "
                 "ask: had the model reached this candidate, or is it a premise, an "
                 "intermediate, or an answer to the wrong quantity?\n\n")
        step = max(len(rows) // args.audit_n, 1)
        for r in rows[::step][: args.audit_n]:
            fh.write(f"## {r['question_id']} ({r['band']}) gold={r['gold']} "
                     f"candidate={r['cand']} "
                     f"{'CORRECT' if r['cand_correct'] else 'INCORRECT'} "
                     f"freeze@{r['freeze_at']}/{r['think_tokens']}\n\n")
            fh.write("```\n..." + r["prefix"][-500:].strip() + "\n```\n\n")
    print(f"wrote {args.out} and {args.audit}")


if __name__ == "__main__":
    main()
