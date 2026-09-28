#!/usr/bin/env python3
"""Confirmation cohort: 400 MATH *test* problems NOT in MATH-500, level-stratified to
MATH-500's level mix (43/90/105/128/134 per 500 -> x0.8). Never used by any run."""
import collections, json, random, sys
from pathlib import Path
from datasets import load_dataset
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reasoning_attention.grading import grade  # noqa: E402

def last_boxed(s):
    i = s.rfind("\\boxed"); j = s.find("{", i) if i >= 0 else -1
    if j < 0: return None
    d = 0
    for k in range(j, len(s)):
        d += (s[k] == "{") - (s[k] == "}")
        if d == 0: return s[j + 1:k]

m5 = {json.loads(l)["question"].strip() for l in open("data/policy/math500.jsonl")}
subs = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
        "number_theory", "prealgebra", "precalculus"]
by = collections.defaultdict(list)
for s in subs:
    for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", s, split="test")):
        g, lv = last_boxed(r["solution"]), r["level"].removeprefix("Level ")
        if r["problem"].strip() in m5 or not g or not lv.isdigit(): continue
        if not grade(f"\\boxed{{{g}}}", g).is_correct: continue
        by[int(lv)].append({"question_id": f"mathtest:{s}:{i}", "dataset": "mathtest",
                            "question": r["problem"], "gold": g, "subject": s, "level": int(lv)})
rng = random.Random(20260926)
quota = {1: 34, 2: 72, 3: 84, 4: 103, 5: 107}
rows = []
for lv, q in quota.items():
    rng.shuffle(by[lv]); rows += by[lv][:q]
Path("data/xfer8b/confirm_mathtest400.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
print(len(rows), {lv: len(v) for lv, v in sorted(by.items())})
