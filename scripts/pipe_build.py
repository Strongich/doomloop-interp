#!/usr/bin/env python3
r"""EXPERIMENT-pipeline-sft.md §P2 sets + §P4 fresh split + lock.

  split   data/pipeline_sft/mathtest_pipe1000.jsonl: hendrycks_math TEST, 200 per level, seed
          20261003; excluded by subject:index and text: confirm400, mathtest_fresh1000,
          mathtest_sel400; by text: MATH-500 and every question-carrying jsonl in data/
          (training pools included). Golds self-grade (D69).
  sets    data/pipeline_sft/sets/sft_{pipeline,gated,mix}.jsonl (v2 format; stored ids),
          v2 seeds 0-3 only, v2 §L4 eligibility.
          pipeline  shortest eligible steered of seeds 0-1, else shortest eligible base of
                    seeds 2-3, else drop
          gated     s(q) >= b(q) (correct counts, seeds 0-3) and an eligible steered exists ->
                    shortest eligible steered; else shortest eligible base; else drop
          mix       §P9 item 1: same questions and base count k as pipeline. Questions with
                    no eligible steered in seeds 0-3 are forced to base; the remaining base
                    slots are filled uniformly at random (seed 20261002) among questions
                    with an eligible base in seeds 2-3. Base group: shortest eligible base
                    of seeds 2-3; steered group: shortest eligible steered of seeds 0-1,
                    else of seeds 2-3.

    uv run python scripts/pipe_build.py            # split + sets + lock
"""

from __future__ import annotations

import collections
import glob
import hashlib
import json
import random
import statistics as st
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from sel_build import SUBJECTS, last_boxed  # noqa: E402

CTX = 16384
BASE, STEER = "base", "N@a1.0d256"
OUT = ROOT / "data/pipeline_sft"
USED_TEST = ["data/xfer8b/confirm_mathtest400.jsonl", "data/distill_v3/mathtest_fresh1000.jsonl",
             "data/selective/mathtest_sel400.jsonl"]


def sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build_split() -> dict:
    from datasets import load_dataset

    from reasoning_attention.grading import grade

    used_ids, texts = set(), set()
    for f in USED_TEST:
        for line in open(ROOT / f):
            r = json.loads(line)
            used_ids.add(r["question_id"].split(":", 1)[1])
            texts.add(r["question"].strip())
    for f in glob.glob(str(ROOT / "data/**/*.jsonl"), recursive=True):
        p = Path(f)
        if p.stat().st_size > 200e6 or "rollouts" in p.name or p.is_relative_to(OUT):
            continue
        with p.open() as fh:
            first = fh.readline()
            if '"question"' not in first:
                continue
            for line in [first, *fh]:
                try:
                    q = json.loads(line).get("question")
                except json.JSONDecodeError:
                    continue
                if isinstance(q, str):
                    texts.add(q.strip())
    by, excl = collections.defaultdict(list), collections.Counter()
    for s in SUBJECTS:
        for i, r in enumerate(load_dataset("EleutherAI/hendrycks_math", s, split="test")):
            lv = r["level"].removeprefix("Level ")
            if not lv.isdigit():
                excl["no_level"] += 1
                continue
            g = last_boxed(r["solution"])
            if f"{s}:{i}" in used_ids:
                excl["used_id"] += 1
            elif r["problem"].strip() in texts:
                excl["used_text"] += 1
            elif not g or not grade(f"\\boxed{{{g}}}", g).is_correct:
                excl["gold_not_self_grading"] += 1
            else:
                by[int(lv)].append({"question_id": f"mathpipe:{s}:{i}", "dataset": "mathpipe",
                                    "question": r["problem"], "gold": g, "subject": s,
                                    "level": int(lv)})
    rng = random.Random(20261003)
    rows, short = [], {}
    for lv in range(1, 6):
        xs = sorted(by[lv], key=lambda r: r["question_id"])
        rng.shuffle(xs)
        if len(xs) < 200:
            short[lv] = 200 - len(xs)
        rows += xs[:200]
    path = OUT / "mathtest_pipe1000.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"path": str(path.relative_to(ROOT)), "n": len(rows), "excluded": dict(excl),
            "usable": {lv: len(v) for lv, v in sorted(by.items())}, "shortfall": short,
            "question_ids": [r["question_id"] for r in rows]}


def build_sets() -> dict:
    from transformers import AutoTokenizer

    from sel_common import Paragraphs

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
    P = Paragraphs(tok)
    pool = {json.loads(x)["question_id"]: json.loads(x) for x in open(ROOT / "data/distill/train.jsonl")}
    by: dict = collections.defaultdict(lambda: {BASE: {}, STEER: {}})
    for d in sorted(p for p in glob.glob(str(ROOT / "data/distill/gen/shard*")) if Path(p).is_dir()):
        for line in open(Path(d) / "rollouts.jsonl"):
            r = json.loads(line)
            if r["seed"] < 4:
                by[r["question_id"]][r["policy"]][r["seed"]] = r
    assert set(by) == set(pool) and all(
        sorted(v[p]) == [0, 1, 2, 3] for v in by.values() for p in (BASE, STEER)), "v2 gen incomplete"

    def elig(r: dict) -> bool:
        return bool(r["correct"] and r["closed"] and r["finish_reason"] == "stop"
                    and r["has_answer"] and r["prompt_tokens"] + r["total_tokens"] <= CTX)

    def shortest(q: str, pol: str, seeds: tuple[int, ...]) -> dict | None:
        xs = [by[q][pol][s] for s in seeds if elig(by[q][pol][s])]
        return min(xs, key=lambda r: (r["total_tokens"], r["seed"])) if xs else None

    arms: dict[str, dict[str, dict]] = {"pipeline": {}, "gated": {}, "mix": {}}
    for q in sorted(pool):
        c = shortest(q, STEER, (0, 1)) or shortest(q, BASE, (2, 3))
        if c:
            arms["pipeline"][q] = c
        s_ = sum(by[q][STEER][i]["correct"] for i in range(4))
        b_ = sum(by[q][BASE][i]["correct"] for i in range(4))
        es, eb = shortest(q, STEER, (0, 1, 2, 3)), shortest(q, BASE, (0, 1, 2, 3))
        g = es if (s_ >= b_ and es) else eb
        if g:
            arms["gated"][q] = g
    qp = sorted(arms["pipeline"])
    k = sum(arms["pipeline"][q]["policy"] == BASE for q in qp)
    forced = [q for q in qp if shortest(q, STEER, (0, 1, 2, 3)) is None]
    free = [q for q in qp if q not in set(forced) and shortest(q, BASE, (2, 3))]
    rng = random.Random(20261002)
    rng.shuffle(free)
    if len(forced) > k or len(forced) + len(free) < k:
        raise SystemExit(f"mix infeasible: k {k}, forced {len(forced)}, free {len(free)}")
    base_group = set(forced) | set(free[: k - len(forced)])
    for q in qp:
        arms["mix"][q] = (shortest(q, BASE, (2, 3)) if q in base_group
                          else shortest(q, STEER, (0, 1)) or shortest(q, STEER, (2, 3)))
    assert sum(r["policy"] == BASE for r in arms["mix"].values()) == k
    overlap = len(base_group & {q for q in qp if arms["pipeline"][q]["policy"] == BASE})

    v2q = {json.loads(x)["question_id"] for x in open(ROOT / "data/distill/sets/sft_short.jsonl")}
    (OUT / "sets").mkdir(parents=True, exist_ok=True)
    prompts: dict[str, list[int]] = {}
    stats: dict = {"pool": len(pool), "k_pipeline_base": k, "mix_forced_base": len(forced),
                   "mix_base_overlap_with_pipeline_base": overlap}

    def lvl(q: str) -> str:
        return "gsm8k" if pool[q]["dataset"] == "gsm8k" else f"L{pool[q]['level']}"

    for arm, rows in arms.items():
        with (OUT / "sets" / f"sft_{arm}.jsonl").open("w") as f:
            for q in sorted(rows):
                r = rows[q]
                if q not in prompts:
                    prompts[q] = P.prompt_ids(pool[q]["question"])
                    assert len(prompts[q]) == r["prompt_tokens"], q
                f.write(json.dumps({"question_id": q, "dataset": pool[q]["dataset"],
                                    "level": pool[q].get("level", 0), "policy": r["policy"],
                                    "seed": r["seed"], "prompt_ids": prompts[q],
                                    "completion_ids": r["token_ids"]}) + "\n")
        lens = [r["total_tokens"] for r in rows.values()]
        cell = collections.defaultdict(lambda: [0, 0])
        for q, r in rows.items():
            cell[lvl(q)][0] += 1
            cell[lvl(q)][1] += r["policy"] == BASE
        stats[arm] = {
            "questions": len(rows), "base_sourced": sum(r["policy"] == BASE for r in rows.values()),
            "base_share_by_level": {c: {"n": n, "base": b, "share": round(b / n, 4)}
                                    for c, (n, b) in sorted(cell.items())},
            "completion_tokens": sum(lens), "mean_len": round(st.mean(lens), 1),
            "median_len": st.median(lens),
            "overlap_v2_qsft": len(set(rows) & v2q), "v2_qsft": len(v2q)}
    return stats


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lock = OUT / "protocol_locked.json"
    if lock.exists():
        raise SystemExit(f"{lock} exists")
    info: dict = {"protocol": "EXPERIMENT-pipeline-sft.md (locked 2026-10-02)",
                  "written": time.strftime("%Y-%m-%d %H:%M:%S"),
                  "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                                               capture_output=True, text=True).stdout.strip(),
                  "split": build_split()}
    info["sets"] = build_sets()
    info["sha256"] = {"split": sha(ROOT / info["split"]["path"]),
                      "protocol_md": sha(ROOT / "EXPERIMENT-pipeline-sft.md"),
                      **{f"sft_{a}": sha(OUT / "sets" / f"sft_{a}.jsonl")
                         for a in ("pipeline", "gated", "mix")}}
    lock.write_text(json.dumps(info, indent=1) + "\n")
    (OUT / "sets" / "sets_stats.json").write_text(json.dumps(info["sets"], indent=1) + "\n")
    print(json.dumps({k: v for k, v in info.items() if k != "split"}, indent=1))
    print(json.dumps({k: v for k, v in info["split"].items() if k != "question_ids"}, indent=1))


if __name__ == "__main__":
    main()
