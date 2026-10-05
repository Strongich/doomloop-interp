#!/usr/bin/env python3
r"""PROTOCOL v4 §M2 lock (EXPERIMENT-offline-distillation.md), written before 8B generation.

  pool    data/distill/math_train.jsonl (6,588 MATH-train questions, the v2 pool)
  val     data/distill_8b/val_math150.jsonl = the 150 MATH questions of data/distill/val.jsonl
  shards  data/distill_8b/gen/shard{k}.jsonl, round-robin over the pool ordered by
          (level, question_id), so every shard carries the same level mix
  lock    data/distill_8b/protocol_v4_locked.json: sha256 of pool, val, every test set,
          the direction file and the code; git commit; model ids

    uv run python scripts/distill8b_lock.py --shards 6
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data/distill_8b"
TESTS = {"mathfresh": "data/distill_v3/mathtest_fresh1000.jsonl",
         "mathgate": "data/gated/mathtest_gate400.jsonl",
         "aime_amc": "data/distill/eval_aime_amc.jsonl",
         "math500": "data/policy/math500.jsonl",
         "gsm8k_test": "data/distill/eval_gsm8k_test.jsonl"}
CODE = ["scripts/reasoning_policy_vllm.py", "src/reasoning_attention/serving/vllm_steering.py",
        "src/reasoning_attention/grading.py", "scripts/distill_build_sets.py",
        "scripts/distill_train.py", "scripts/distill_eval.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", type=int, default=6)
    args = ap.parse_args()
    lock = OUT / "protocol_v4_locked.json"
    if lock.exists():
        raise SystemExit(f"{lock} exists")
    (OUT / "gen").mkdir(parents=True, exist_ok=True)
    pool = [json.loads(x) for x in open(ROOT / "data/distill/math_train.jsonl")]
    val = [json.loads(x) for x in open(ROOT / "data/distill/val.jsonl")]
    val = [r for r in val if r["dataset"] == "mathtrain"]
    assert len(pool) == 6588 and len(val) == 150
    ids = {r["question_id"] for r in pool}
    assert not ids & {r["question_id"] for r in val}
    texts = {r["question"].strip() for r in pool}
    overlap = {}
    for name, p in TESTS.items():
        rs = [json.loads(x) for x in open(ROOT / p)]
        overlap[name] = sum(r["question"].strip() in texts for r in rs)
    assert not any(overlap.values()), overlap
    (OUT / "val_math150.jsonl").write_text("".join(json.dumps(r) + "\n" for r in val))
    rows = sorted(pool, key=lambda r: (r["level"], r["question_id"]))
    for k in range(args.shards):
        (OUT / f"gen/shard{k}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows[k :: args.shards]))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    info = {
        "protocol": "EXPERIMENT-offline-distillation.md PROTOCOL v4 (8B phase), locked 2026-10-03",
        "written": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": "Qwen/Qwen3-8B", "teacher": "NU26@a0.75d256", "layer": 26,
        "direction": {"path": "data/xfer8b/dirs_clean/L26_NU.pt",
                      "sha256": sha(ROOT / "data/xfer8b/dirs_clean/L26_NU.pt")},
        "pool": {"path": "data/distill/math_train.jsonl", "n": len(pool),
                 "sha256": sha(ROOT / "data/distill/math_train.jsonl")},
        "val": {"path": "data/distill_8b/val_math150.jsonl", "n": len(val),
                "sha256": sha(OUT / "val_math150.jsonl")},
        "tests": {k: {"path": p, "sha256": sha(ROOT / p)} for k, p in TESTS.items()},
        "test_text_overlap_with_pool": overlap,
        "shards": {k: sha(OUT / f"gen/shard{k}.jsonl") for k in range(args.shards)},
        "git_commit": commit, "git_dirty": dirty,
        "code_sha256": {p: sha(ROOT / p) for p in CODE},
        "protocol_md_sha256": sha(ROOT / "EXPERIMENT-offline-distillation.md"),
    }
    lock.write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
