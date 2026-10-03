#!/usr/bin/env python3
r"""EXPERIMENT-selective-doubt.md §S2 + §S6: source sample, test split and the lock.

Run once, before any probe, state or direction exists.

  §S6 test   data/selective/mathtest_sel400.jsonl -- 200 L4 + 200 L5 hendrycks_math TEST
             problems; excluded: MATH-500 (by text), confirm_mathtest400 + mathtest_fresh1000
             (by subject:index and by text), every question text of any MATH pool file in
             data/ (train pools included, as a duplicate guard); golds self-grade (D69).
  §S2 source data/selective/source1000.jsonl -- base rollouts of the distillation v2
             generation, mathtrain levels 3-5, doubt_blocks >= 1, at most 2 per question:
             all candidates shuffled (seed 20261002), taken in order while the question has
             < 2, until 1,000.
  lock       data/selective/protocol_locked.json -- sha256 of both files and the protocol.

    uv run python scripts/sel_build.py
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED = 20261002
OUT = ROOT / "data/selective"
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
USED_TEST = ["data/xfer8b/confirm_mathtest400.jsonl", "data/distill_v3/mathtest_fresh1000.jsonl"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def last_boxed(s: str) -> str | None:  # as in xfer_build_confirm.py
    i = s.rfind("\\boxed")
    j = s.find("{", i) if i >= 0 else -1
    if j < 0:
        return None
    d = 0
    for k in range(j, len(s)):
        d += (s[k] == "{") - (s[k] == "}")
        if d == 0:
            return s[j + 1 : k]
    return None


def build_test() -> dict:
    from datasets import load_dataset

    from reasoning_attention.grading import grade

    used_ids = set()
    texts = set()
    for f in USED_TEST:
        for line in open(ROOT / f):
            r = json.loads(line)
            used_ids.add(r["question_id"].split(":", 1)[1])  # subject:index
            texts.add(r["question"].strip())
    # Every small jsonl in data/ that carries question texts (pools, cohorts, MATH-500).
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
    excl = collections.Counter()
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
            by[int(lv)].append({"question_id": f"mathsel:{s}:{i}", "dataset": "mathsel",
                                "question": r["problem"], "gold": g, "subject": s,
                                "level": int(lv)})
    rng = random.Random(SEED)
    rows = []
    for lv in (4, 5):
        xs = sorted(by[lv], key=lambda r: r["question_id"])
        rng.shuffle(xs)
        rows += xs[:200]
    path = OUT / "mathtest_sel400.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"path": str(path.relative_to(ROOT)), "usable": {lv: len(v) for lv, v in by.items()},
            "excluded": dict(excl), "text_files_scanned": n_files, "n": len(rows)}


def build_source() -> dict:
    pool = {json.loads(x)["question_id"]: json.loads(x) for x in open(ROOT / "data/distill/train.jsonl")}
    cand = []
    for d in sorted(p for p in glob.glob(str(ROOT / "data/distill/gen/shard*")) if Path(p).is_dir()):
        for line in open(Path(d) / "rollouts.jsonl"):
            r = json.loads(line)
            q = pool.get(r["question_id"])
            if (r["policy"] == "base" and q and q["dataset"] == "mathtrain"
                    and q.get("level") in (3, 4, 5) and r["doubt_blocks"] >= 1):
                cand.append({"question_id": r["question_id"], "seed": r["seed"],
                             "level": q["level"], "doubt_blocks": r["doubt_blocks"],
                             "total_tokens": r["total_tokens"], "correct": r["correct"],
                             "capped": r["capped"], "shard": Path(d).name})
    cand.sort(key=lambda r: (r["question_id"], r["seed"]))
    random.Random(SEED).shuffle(cand)
    per: collections.Counter = collections.Counter()
    out = []
    for r in cand:
        if per[r["question_id"]] < 2:
            per[r["question_id"]] += 1
            out.append(r)
            if len(out) == 1000:
                break
    path = OUT / "source1000.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in out))
    lv = collections.Counter(r["level"] for r in out)
    return {"path": str(path.relative_to(ROOT)), "candidates": len(cand), "n": len(out),
            "questions": len(per), "by_level": dict(sorted(lv.items())),
            "correct": sum(r["correct"] for r in out), "capped": sum(r["capped"] for r in out),
            "mean_tokens": round(sum(r["total_tokens"] for r in out) / len(out), 1),
            "mean_doubt_blocks": round(sum(r["doubt_blocks"] for r in out) / len(out), 2)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "protocol_locked.json"
    if lock.exists():
        raise SystemExit(f"{lock} exists; the split is already locked")
    info = {"protocol": "EXPERIMENT-selective-doubt.md (locked 2026-10-02)",
            "written": time.strftime("%Y-%m-%d %H:%M:%S"), "seed": SEED,
            "test": build_test(), "source": build_source()}
    info["sha256"] = {k: sha(ROOT / info[k]["path"]) for k in ("test", "source")}
    info["sha256"]["protocol_md"] = sha(ROOT / "EXPERIMENT-selective-doubt.md")
    lock.write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
