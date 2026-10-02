#!/usr/bin/env python3
r"""Offline-distillation pools, LOCKED PROTOCOL v2 §L2 (EXPERIMENT-offline-distillation.md).

  gsm8k_train  official GSM8K train (combined-index ids gsm8k:0..7472), minus every
               question in data/policy/dev400.jsonl / dev200.jsonl (by id and by text),
               minus any text equal to a GSM8K test question
  math_train   EleutherAI/hendrycks_math train, gold = last \boxed{} of the solution,
               gold must grade itself correct under the D69 grader; minus
               data/xfer8b/dev_math250.jsonl / fit_math400.jsonl (by id and text) and any
               text equal (whitespace-normalized) to a MATH-500 or confirm_mathtest400 problem
  val          150 GSM8K-train + 150 MATH-train (30 per level), seed 20260930, removed
               from the training pools

Writes data/distill/{gsm8k_train,math_train,train,val}.jsonl and the lock file
data/distill/protocol_v2_locked.json (question ids, sha256s, git commit, model/teacher).

    uv run python scripts/distill_build_pools.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
import subprocess
import sys
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from reasoning_attention.data.math_datasets import _gsm8k_final_answer  # noqa: E402
from reasoning_attention.grading import grade  # noqa: E402

# Copied from scripts/xfer_build_cohorts.py, which runs its whole body at import time
# (it would rewrite the 8B dev/fit cohorts).
SUBJECTS = ["algebra", "counting_and_probability", "geometry", "intermediate_algebra",
            "number_theory", "prealgebra", "precalculus"]


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

SEED = 20260930
OUT = ROOT / "data/distill"
N_TRAIN_GSM8K = 7473  # combined-index ids >= this are GSM8K test


def norm(s: str) -> str:
    return " ".join(s.split())


def read(p: str) -> list[dict]:
    return [json.loads(x) for x in (ROOT / p).read_text().splitlines() if x.strip()]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gsm_ex = read("data/policy/dev400.jsonl") + read("data/policy/dev200.jsonl")
    gsm_ex_ids = {r["question_id"] for r in gsm_ex}
    gsm_ex_txt = {norm(r["question"]) for r in gsm_ex}
    gtest = load_dataset("openai/gsm8k", "main", split="test")
    gsm_ex_txt |= {norm(r["question"]) for r in gtest}
    gsm = []
    for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="train")):
        qid = f"gsm8k:{i}"
        if qid in gsm_ex_ids or norm(r["question"]) in gsm_ex_txt:
            continue
        gsm.append({"question_id": qid, "dataset": "gsm8k", "question": r["question"],
                    "gold": _gsm8k_final_answer(r["answer"]), "split": "train"})
    print(f"gsm8k train: {len(gsm)} kept of 7473 "
          f"({len([q for q in gsm_ex_ids if int(q.split(':')[1]) < N_TRAIN_GSM8K])} dev ids)")

    m_ex = read("data/xfer8b/dev_math250.jsonl") + read("data/xfer8b/fit_math400.jsonl")
    m_ex_ids = {r["question_id"] for r in m_ex}
    m_ex_txt = {norm(r["question"]) for r in m_ex}
    m_ex_txt |= {norm(r["question"]) for r in read("data/policy/math500.jsonl")}
    m_ex_txt |= {norm(r["question"]) for r in read("data/xfer8b/confirm_mathtest400.jsonl")}
    math, drop = [], collections.Counter()
    for s in SUBJECTS:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", s, split="train")):
            qid = f"mathtrain:{s}:{i}"
            if qid in m_ex_ids or norm(r["problem"]) in m_ex_txt:
                drop["excluded"] += 1
                continue
            gold = last_boxed(r["solution"])
            lv = r["level"].removeprefix("Level ")
            if not gold or not lv.isdigit():
                drop["no gold/level"] += 1
                continue
            if not grade(f"\\boxed{{{gold}}}", gold).is_correct:
                drop["gold fails D69 self-grade"] += 1
                continue
            math.append({"question_id": qid, "dataset": "mathtrain", "question": r["problem"],
                         "gold": gold, "subject": s, "level": int(lv)})
    print(f"math train: {len(math)} kept; dropped {dict(drop)}")
    # Duplicate texts inside a pool would put one question in both train and val.
    for name, pool in (("gsm8k", gsm), ("math", math)):
        seen: dict[str, str] = {}
        dup = [q["question_id"] for q in pool if seen.setdefault(norm(q["question"]),
                                                                 q["question_id"]) != q["question_id"]]
        if dup:
            print(f"{name}: dropping {len(dup)} in-pool duplicate texts")
            pool[:] = [q for q in pool if q["question_id"] not in set(dup)]

    rng = random.Random(SEED)
    val_ids = set(rng.sample([q["question_id"] for q in gsm], 150))
    for lv in range(1, 6):
        val_ids |= set(rng.sample(sorted(q["question_id"] for q in math if q["level"] == lv), 30))
    val = [q for q in gsm + math if q["question_id"] in val_ids]
    gsm_tr = [q for q in gsm if q["question_id"] not in val_ids]
    math_tr = [q for q in math if q["question_id"] not in val_ids]
    assert not ({norm(q["question"]) for q in val}
                & {norm(q["question"]) for q in gsm_tr + math_tr})
    files = {"gsm8k_train": gsm_tr, "math_train": math_tr, "train": gsm_tr + math_tr, "val": val}
    for name, rows in files.items():
        (OUT / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        print(f"{name}: {len(rows)}", dict(collections.Counter(
            r.get("level", r["dataset"]) for r in rows)) if name != "gsm8k_train" else "")

    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)
    code = ["scripts/distill_build_pools.py", "scripts/reasoning_policy_vllm.py",
            "src/reasoning_attention/serving/vllm_steering.py",
            "src/reasoning_attention/grading.py"]
    lock = {
        "protocol": "EXPERIMENT-offline-distillation.md LOCKED PROTOCOL v2",
        "created": "2026-09-30",
        "git_commit": commit, "git_dirty": bool(dirty.strip()),
        "seed": SEED,
        "student_model": "Qwen/Qwen3-1.7B",
        "teacher": {"model": "Qwen/Qwen3-1.7B", "layer": 20, "policy": "N@a1.0d256",
                    "direction": "data/pool/dir_A_1trace.pt",
                    "direction_sha256": sha(ROOT / "data/pool/dir_A_1trace.pt")},
        "generation": {"per_question": {"base": 4, "N@a1.0d256": 4}, "max_new_tokens": 32768,
                       "temperature": 0.6, "top_p": 0.95, "top_k": 20,
                       "seed_scheme": "sha256(question_id:seed)"},
        "files": {f"data/distill/{n}.jsonl": sha(OUT / f"{n}.jsonl") for n in files},
        "exclusion_files": {p: sha(ROOT / p) for p in (
            "data/policy/dev400.jsonl", "data/policy/dev200.jsonl",
            "data/xfer8b/dev_math250.jsonl", "data/xfer8b/fit_math400.jsonl",
            "data/policy/math500.jsonl", "data/xfer8b/confirm_mathtest400.jsonl")},
        "code_sha256": {p: sha(ROOT / p) for p in code},
        "question_ids": {n: [r["question_id"] for r in rows] for n, rows in files.items()
                         if n != "train"},
    }
    lock_path = OUT / "protocol_v2_locked.json"
    if lock_path.exists():
        raise SystemExit(f"{lock_path} exists; the lock is written once")
    lock_path.write_text(json.dumps(lock, indent=1) + "\n")
    print(f"wrote {lock_path}")


if __name__ == "__main__":
    main()
