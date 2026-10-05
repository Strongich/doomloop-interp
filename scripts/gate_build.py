#!/usr/bin/env python3
r"""EXPERIMENT-probe-gated.md §G7: the primary test split and the lock.

  split  data/gated/mathtest_gate400.jsonl -- 200 L4 + 200 L5 hendrycks_math TEST problems
         (prefix `mathgate:`); excluded by subject:index AND by text: confirm_mathtest400,
         mathtest_fresh1000, mathtest_sel400, mathtest_pipe1000; by text: every question of
         any MATH/GSM jsonl in data/ (MATH-500 and every training pool included); golds
         self-grade (D69); seed 20261003.
  lock   (--lock, after gate_probe.py wrote probe.pt, before any gated rollout)
         data/gated/protocol_locked.json -- sha256 of the split, probe.pt, the protocol.

    uv run python scripts/gate_build.py            # split
    uv run python scripts/gate_build.py --lock     # lock
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sel_build import SUBJECTS, last_boxed, sha  # noqa: E402

SEED = 20261003
OUT = ROOT / "data/gated"
SPLIT = OUT / "mathtest_gate400.jsonl"
USED_TEST = ["data/xfer8b/confirm_mathtest400.jsonl", "data/distill_v3/mathtest_fresh1000.jsonl",
             "data/selective/mathtest_sel400.jsonl", "data/pipeline_sft/mathtest_pipe1000.jsonl"]


def build_split() -> dict:
    from datasets import load_dataset

    from reasoning_attention.grading import grade

    used_ids, texts = set(), set()
    for f in USED_TEST:
        for line in open(ROOT / f):
            r = json.loads(line)
            used_ids.add(r["question_id"].split(":", 1)[1])  # subject:index
            texts.add(r["question"].strip())
    n_files = 0
    for f in glob.glob(str(ROOT / "data/**/*.jsonl"), recursive=True):
        p = Path(f)
        if p.stat().st_size > 200e6 or "rollouts" in p.name or p.is_relative_to(OUT):
            continue
        with p.open() as fh:
            first = fh.readline()
            if '"question"' not in first:
                continue
            n_files += 1
            for line in [first, *fh]:
                try:
                    q = json.loads(line).get("question")
                except json.JSONDecodeError:
                    continue
                if isinstance(q, str):
                    texts.add(q.strip())
    by = collections.defaultdict(list)
    excl: collections.Counter = collections.Counter()
    for s in SUBJECTS:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", s, split="test")):
            lv = r["level"].removeprefix("Level ")
            if lv not in ("4", "5"):
                continue
            g = last_boxed(r["solution"])
            if f"{s}:{i}" in used_ids:
                excl["used_id"] += 1
                continue
            if r["problem"].strip() in texts:
                excl["used_text"] += 1
                continue
            if not g or not grade(f"\\boxed{{{g}}}", g).is_correct:
                excl["gold_not_self_grading"] += 1
                continue
            by[int(lv)].append({"question_id": f"mathgate:{s}:{i}", "dataset": "mathgate",
                                "question": r["problem"], "gold": g, "subject": s,
                                "level": int(lv)})
    rng = random.Random(SEED)
    rows = []
    for lv in (4, 5):
        xs = sorted(by[lv], key=lambda r: r["question_id"])
        rng.shuffle(xs)
        rows += xs[:200]
    SPLIT.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"path": str(SPLIT.relative_to(ROOT)), "usable": {lv: len(v) for lv, v in by.items()},
            "excluded": dict(excl), "text_files_scanned": n_files, "n": len(rows),
            "by_level": dict(collections.Counter(r["level"] for r in rows))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "protocol_locked.json"
    info_path = OUT / "split_info.json"
    if lock.exists():
        raise SystemExit(f"{lock} exists; already locked")
    if not args.lock:
        if SPLIT.exists():
            raise SystemExit(f"{SPLIT} exists")
        info = {"written": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED, "split": build_split()}
        info_path.write_text(json.dumps(info, indent=1) + "\n")
        print(json.dumps(info, indent=1))
        return
    info = json.loads(info_path.read_text())
    info["protocol"] = "EXPERIMENT-probe-gated.md (locked 2026-10-03)"
    info["locked"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["sha256"] = {"split": sha(SPLIT), "probe": sha(OUT / "probe.pt"),
                      "probe_report": sha(OUT / "probe_report.json"),
                      "protocol_md": sha(ROOT / "EXPERIMENT-probe-gated.md")}
    lock.write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
