#!/usr/bin/env python3
r"""PROTOCOL v3 §V4 training sets (EXPERIMENT-offline-distillation.md).

  SFT (easy = base 4/4 in v2)   one trace per question; Q = easy questions with >= 1
                                eligible base AND >= 1 eligible steered (v2 seeds 0-3)
      natural  shortest eligible base        nla  shortest eligible steered
  DPO (hard = base <= 2/4)      v2 seeds 0-3 + v3 top-up seeds 4-15 (16 + 16 rollouts)
      A  chosen = eligible base, longest first (cycled); rejected = wrong trace of the twin's
         source (natural: base, nla: steered) with len <= 0.67 x chosen, shortest first
      B  chosen = the twin's eligible traces, shortest first; rejected = wrong trace of the
         same source with len in [0.67, 1.5] x chosen, closest in length
      up to k = 4 pairs per type per question, rejected traces distinct; per-type counts
      matched across twins (subsample the larger, seed 20260930)

Eligible / wrong / length as in v2 (distill_build_sets.py). Stored token ids only.

    uv run python scripts/distill_v3_build_sets.py
"""

from __future__ import annotations

import collections
import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED, CTX, K = 20260930, 16384, 4
BASE, STEER = "base", "N@a1.0d256"
FORMAT = "Give your final answer in \\boxed{}."
OUT = ROOT / "data/distill_v3/sets"


def main() -> None:
    sft_only = "--sft-only" in sys.argv  # SFT needs only the easy questions (v2 seeds 0-3)
    from transformers import AutoTokenizer

    from reasoning_attention.data.math_datasets import build_messages

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    def prompt_ids(q: str) -> list[int]:
        m = [{"role": "system", "content": FORMAT}, *build_messages(q)]
        t = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True,
                                    enable_thinking=True)
        return list(tok(t, add_special_tokens=False)["input_ids"])

    easy = {json.loads(x)["question_id"]: json.loads(x)
            for x in open(ROOT / "data/distill_v3/easy.jsonl")}
    hard = {json.loads(x)["question_id"]: json.loads(x)
            for x in open(ROOT / "data/distill_v3/hard.jsonl")}
    by: dict = collections.defaultdict(lambda: {BASE: [], STEER: []})
    for pat in ("data/distill/gen/shard*/rollouts.jsonl", "data/distill_v3/gen/shard*/rollouts.jsonl"):
        for f in sorted(glob.glob(str(ROOT / pat))):
            for line in open(f):
                r = json.loads(line)
                if r["question_id"] in easy or r["question_id"] in hard:
                    by[r["question_id"]][r["policy"]].append(r)
    for q in ([] if sft_only else hard):
        for pol in (BASE, STEER):
            seeds = sorted(r["seed"] for r in by[q][pol])
            if seeds != list(range(16)):
                raise SystemExit(f"{q} {pol}: seeds {seeds[:5]}... (top-up incomplete?)")

    fits = lambda r: r["prompt_tokens"] + r["total_tokens"] <= CTX  # noqa: E731
    elig = lambda r: bool(r["correct"] and r["closed"] and r["finish_reason"] == "stop"  # noqa: E731
                          and r["has_answer"] and fits(r))
    wrong = lambda r: (not r["correct"]) and fits(r)  # noqa: E731
    key = lambda r: (r["total_tokens"], r["seed"])  # noqa: E731
    prompts: dict[str, list[int]] = {}

    def P(q: str, pool: dict) -> list[int]:
        if q not in prompts:
            prompts[q] = prompt_ids(pool[q]["question"])
            if any(r["prompt_tokens"] != len(prompts[q]) for v in by[q].values() for r in v):
                raise SystemExit(f"prompt length mismatch {q}")
        return prompts[q]

    ex = lambda r: {"policy": r["policy"], "seed": r["seed"], "len": r["total_tokens"],  # noqa: E731
                    "ids": r["token_ids"]}
    sft: dict[str, list] = {"natural": [], "nla": []}
    for q in sorted(easy):
        eb = sorted(filter(elig, by[q][BASE]), key=key)
        es = sorted(filter(elig, by[q][STEER]), key=key)
        if eb and es:
            meta = {"question_id": q, "dataset": easy[q]["dataset"], "level": easy[q].get("level", 0)}
            sft["natural"].append({**meta, **ex(eb[0])})
            sft["nla"].append({**meta, **ex(es[0])})

    dpo: dict = {t: {"A": [], "B": []} for t in ("natural", "nla")}
    for q in ([] if sft_only else sorted(hard)):
        meta = {"question_id": q, "dataset": hard[q]["dataset"], "level": hard[q].get("level", 0)}
        eb = sorted(filter(elig, by[q][BASE]), key=key)
        for twin, src in (("natural", BASE), ("nla", STEER)):
            wr = sorted(filter(wrong, by[q][src]), key=key)
            # A: long base correct vs short wrong of the twin's source. Rejected traces are
            # taken shortest first; chosen traces cycle through eligible base, longest first.
            desc = sorted(eb, key=lambda r: (-r["total_tokens"], r["seed"]))
            i = n_a = 0
            for w in wr:
                if n_a >= K or not desc:
                    break
                for j in range(len(desc)):
                    c = desc[(i + j) % len(desc)]
                    if w["total_tokens"] <= 0.67 * c["total_tokens"]:
                        dpo[twin]["A"].append({**meta, "chosen": ex(c), "rejected": ex(w)})
                        i, n_a = i + j + 1, n_a + 1
                        break
            # B: twin's short correct vs wrong of the same length band
            usedb: set = set()
            for c in sorted(filter(elig, by[q][src]), key=key):
                if len(usedb) >= K:
                    break
                cand = [w for w in wr if w["seed"] not in usedb
                        and 0.67 * c["total_tokens"] <= w["total_tokens"] <= 1.5 * c["total_tokens"]]
                if cand:
                    w = min(cand, key=lambda w: (abs(w["total_tokens"] - c["total_tokens"]), w["seed"]))
                    usedb.add(w["seed"])
                    dpo[twin]["B"].append({**meta, "chosen": ex(c), "rejected": ex(w)})

    raw = {t: {k: len(v) for k, v in d.items()} for t, d in dpo.items()}
    for typ in ("A", "B"):
        n = min(len(dpo["natural"][typ]), len(dpo["nla"][typ]))
        for twin in dpo:
            xs = dpo[twin][typ]
            if len(xs) > n:
                keep = set(random.Random(f"{SEED}:v3sub:{twin}:{typ}").sample(range(len(xs)), n))
                dpo[twin][typ] = [x for i, x in enumerate(xs) if i in keep]

    OUT.mkdir(parents=True, exist_ok=True)
    stats: dict = {"easy_pool": len(easy), "hard_pool": len(hard), "dpo_before_matching": raw}
    for twin, rows in sft.items():
        with (OUT / f"sft_{twin}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps({"question_id": r["question_id"], "dataset": r["dataset"],
                                    "level": r["level"], "policy": r["policy"], "seed": r["seed"],
                                    "prompt_ids": P(r["question_id"], easy),
                                    "completion_ids": r["ids"]}) + "\n")
        lens = [r["len"] for r in rows]
        stats[f"sft_{twin}"] = {"n": len(rows), "completion_tokens": sum(lens),
                                "mean_len": round(st.mean(lens), 1), "median_len": st.median(lens)}
    for twin, types in ({} if sft_only else dpo).items():
        with (OUT / f"dpo_{twin}.jsonl").open("w") as f:
            for typ in ("A", "B"):
                for r in types[typ]:
                    f.write(json.dumps({
                        "question_id": r["question_id"], "dataset": r["dataset"], "level": r["level"],
                        "type": typ, "prompt_ids": P(r["question_id"], hard),
                        "chosen_ids": r["chosen"]["ids"], "rejected_ids": r["rejected"]["ids"],
                        "chosen": {k: r["chosen"][k] for k in ("policy", "seed", "len")},
                        "rejected": {k: r["rejected"][k] for k in ("policy", "seed", "len")}}) + "\n")
        stats[f"dpo_{twin}"] = {
            **{t: len(x) for t, x in types.items()},
            "questions": len({r["question_id"] for x in types.values() for r in x}),
            "mean_chosen_len": {t: round(st.mean([r["chosen"]["len"] for r in x]), 1) for t, x in types.items() if x},
            "mean_rejected_len": {t: round(st.mean([r["rejected"]["len"] for r in x]), 1) for t, x in types.items() if x}}
    (OUT / ("sft_stats.json" if sft_only else "sets_stats.json")).write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
