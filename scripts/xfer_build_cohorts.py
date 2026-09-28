#!/usr/bin/env python3
"""8B transfer cohorts from MATH *train* (disjoint from MATH-500, which is drawn from test).

  dev   250 problems, 50 per level -- alpha / layer / method selection for the 8B
  fit   400 problems, 80 per level -- 8B-native traces for the paired-activation map
                                     and the 8B-native diff-of-means direction
Problems whose text appears in MATH-500 are dropped (none expected: train vs test).
Golds are the last \\boxed{} of the reference solution, braces preserved.
"""
import json, random, collections, sys
from pathlib import Path
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reasoning_attention.grading import grade  # noqa: E402


def last_boxed(s: str) -> str | None:
    i = s.rfind("\\boxed")
    if i < 0:
        return None
    j = s.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(s)):
        depth += s[k] == "{"
        depth -= s[k] == "}"
        if depth == 0:
            return s[j + 1 : k]
    return None


SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]
m5 = {json.loads(l)["question"].strip() for l in open("data/policy/math500.jsonl")}
rows = []
for s in SUBJECTS:
    for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", s, split="train")):
        if r["problem"].strip() in m5:
            continue
        gold = last_boxed(r["solution"])
        lv = r["level"].removeprefix("Level ")
        if not gold or not lv.isdigit():
            continue
        # The gold must grade itself correct, or the question can never be scored (D66).
        if not grade(f"\\boxed{{{gold}}}", gold).is_correct:
            continue
        rows.append({"question_id": f"mathtrain:{s}:{i}", "dataset": "mathtrain",
                     "question": r["problem"], "gold": gold, "subject": s, "level": int(lv)})
print(len(rows), "usable MATH-train problems")
rng = random.Random(20260925)
by = collections.defaultdict(list)
for r in rows:
    by[r["level"]].append(r)
dev, fit = [], []
for lv in sorted(by):
    pool = by[lv][:]
    rng.shuffle(pool)
    dev += pool[:50]
    fit += pool[50:130]
out = Path("data/xfer8b")
out.mkdir(parents=True, exist_ok=True)
for name, rs in (("dev_math250", dev), ("fit_math400", fit)):
    (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    print(name, len(rs), dict(collections.Counter(r["level"] for r in rs)))
