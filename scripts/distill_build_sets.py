#!/usr/bin/env python3
r"""SFT and DPO training sets, LOCKED PROTOCOL v2 §L4-§L5 (EXPERIMENT-offline-distillation.md).

Reads the teacher-generation journals (base + N@a1.0d256, 4 each per question) and writes
data/distill/sets/<arm>.jsonl with the stored token ids -- prompt_ids + completion ids,
never a re-render through the chat template -- plus sets_stats.json.

  eligible (§L4)  correct (D69) & </think> closed & finish stop & boxed answer
                  & prompt + generated <= 16,384
  wrong           correct == 0; used only as a DPO rejected trace. It must also fit the
                  16,384 training context (§L10 deviation 1), so capped traces (>= 32k)
                  never qualify.
  length          generated tokens (total_tokens)

Prompt ids are rebuilt with the runner's exact prompt function and checked against the
journaled prompt length. All random picks use random.Random(f"{SEED}:{qid}:{purpose}").

    uv run python scripts/distill_build_sets.py --runs 'data/distill/gen/shard*'
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

SEED = 20260930
CTX = 16384
BASE, STEER = "base", "N@a1.0d256"
FORMAT = "Give your final answer in \\boxed{}."


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="data/distill/gen/shard*")
    ap.add_argument("--pool", default="data/distill/train.jsonl")
    ap.add_argument("--out", type=Path, default=ROOT / "data/distill/sets")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    from reasoning_attention.data.math_datasets import build_messages

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

    def prompt_ids(question: str) -> list[int]:  # == reasoning_policy_vllm.prompt_ids(q, None)
        messages = [{"role": "system", "content": FORMAT}, *build_messages(question)]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                       enable_thinking=True)
        return list(tok(text, add_special_tokens=False)["input_ids"])

    pool = {json.loads(x)["question_id"]: json.loads(x)
            for x in (ROOT / args.pool).read_text().splitlines() if x.strip()}
    by: dict[str, dict[str, list[dict]]] = collections.defaultdict(lambda: {BASE: [], STEER: []})
    n_rec = 0
    for d in sorted(p for p in glob.glob(str(ROOT / args.runs)) if Path(p).is_dir()):
        for line in open(Path(d) / "rollouts.jsonl"):
            r = json.loads(line)
            by[r["question_id"]][r["policy"]].append(r)
            n_rec += 1
    missing = set(pool) - set(by)
    incomplete = [q for q, v in by.items() if len(v[BASE]) != 4 or len(v[STEER]) != 4]
    print(f"{n_rec} rollouts over {len(by)} questions; {len(missing)} pool questions missing, "
          f"{len(incomplete)} incomplete")
    if missing or incomplete or set(by) - set(pool):
        raise SystemExit("generation is not complete / not the locked pool")

    def fits(r: dict) -> bool:
        return r["prompt_tokens"] + r["total_tokens"] <= CTX

    def eligible_nolen(r: dict) -> bool:
        return bool(r["correct"] and r["closed"] and r["finish_reason"] == "stop"
                    and r["has_answer"])

    def pick(qid: str, purpose: str, xs: list[dict]) -> dict:
        return random.Random(f"{SEED}:{qid}:{purpose}").choice(
            sorted(xs, key=lambda r: r["seed"]))

    def shortest(xs: list[dict]) -> dict:
        return min(xs, key=lambda r: (r["total_tokens"], r["seed"]))

    def longest(xs: list[dict]) -> dict:
        return max(xs, key=lambda r: (r["total_tokens"], -r["seed"]))

    excl = {BASE: 0, STEER: 0}  # eligible except for length
    prompts: dict[str, list[int]] = {}
    sft: dict[str, list[dict]] = {k: [] for k in ("ordinary", "short", "steered", "steered_short")}
    dpo: dict[str, dict[str, list[dict]]] = {"natural": {"L": [], "K": [], "P": []},
                                             "steered": {"L": [], "K": [], "P": []}}
    yield_hits, yield_n = 0, 0
    for qid in sorted(by):
        v = by[qid]
        el = {}
        for pol in (BASE, STEER):
            ok = [r for r in v[pol] if eligible_nolen(r)]
            excl[pol] += sum(not fits(r) for r in ok)
            el[pol] = [r for r in ok if fits(r)]
        wrong_b = [r for r in v[BASE] if not r["correct"] and fits(r)]
        wrong_s = [r for r in v[STEER] if not r["correct"] and fits(r)]
        if not (el[BASE] or el[STEER] or wrong_b or wrong_s):
            continue
        p = prompt_ids(pool[qid]["question"])
        if any(r["prompt_tokens"] != len(p) for r in v[BASE] + v[STEER]):
            raise SystemExit(f"prompt length mismatch for {qid}")
        prompts[qid] = p
        meta = {"question_id": qid, "dataset": pool[qid]["dataset"],
                "level": pool[qid].get("level", 0)}

        def ex(r: dict) -> dict:
            return {"policy": r["policy"], "seed": r["seed"], "len": r["total_tokens"],
                    "ids": r["token_ids"]}

        # --- SFT (§L5): Q_SFT = >= 1 eligible base AND >= 1 eligible steered
        if el[BASE] and el[STEER]:
            sft["ordinary"].append({**meta, **ex(pick(qid, "sft_ordinary", el[BASE]))})
            sft["short"].append({**meta, **ex(shortest(el[BASE]))})
            sft["steered"].append({**meta, **ex(pick(qid, "sft_steered", el[STEER]))})
            sft["steered_short"].append({**meta, **ex(shortest(el[STEER]))})
            yield_n += 1
            yield_hits += pick(qid, "sft_steered", el[STEER])["total_tokens"] < \
                shortest(el[BASE])["total_tokens"]
        # --- DPO (§L5)
        chosen_l = {}
        if len(el[BASE]) >= 1:
            c, rj = shortest(el[BASE]), longest(el[BASE])
            if rj["total_tokens"] >= 1.5 * c["total_tokens"]:
                dpo["natural"]["L"].append({**meta, "type": "L", "chosen": ex(c), "rejected": ex(rj)})
                chosen_l["natural"] = c
        if el[STEER] and el[BASE]:
            c, rj = shortest(el[STEER]), longest(el[BASE])
            if rj["total_tokens"] >= 1.5 * c["total_tokens"]:
                dpo["steered"]["L"].append({**meta, "type": "L", "chosen": ex(c), "rejected": ex(rj)})
                chosen_l["steered"] = c
        if wrong_b:
            rj = longest(wrong_b)
            for arm, cpool in (("natural", el[BASE]), ("steered", el[STEER])):
                c = chosen_l.get(arm) or (pick(qid, f"dpo_K_{arm}", cpool) if cpool else None)
                if c is not None:
                    dpo[arm]["K"].append({**meta, "type": "K", "chosen": ex(c), "rejected": ex(rj)})
        if el[BASE]:
            c = pick(qid, "dpo_P_chosen", el[BASE])
            for arm, wpool in (("natural", wrong_b), ("steered", wrong_s)):
                short_w = [r for r in wpool if r["total_tokens"] <= c["total_tokens"]]
                if short_w:
                    dpo[arm]["P"].append({**meta, "type": "P", "chosen": ex(c),
                                          "rejected": ex(shortest(short_w))})

    # Pair-type mix held fixed across the two DPO arms: subsample the larger to the smaller.
    dpo_counts_raw = {a: {t: len(x) for t, x in v.items()} for a, v in dpo.items()}
    for t in ("L", "K", "P"):
        n = min(len(dpo["natural"][t]), len(dpo["steered"][t]))
        for arm in dpo:
            xs = dpo[arm][t]
            if len(xs) > n:
                keep = set(random.Random(f"{SEED}:dpo_sub:{arm}:{t}").sample(range(len(xs)), n))
                dpo[arm][t] = [x for i, x in enumerate(xs) if i in keep]

    args.out.mkdir(parents=True, exist_ok=True)
    stats: dict = {"rollouts": n_rec, "questions": len(by), "context": CTX,
                   "excluded_eligible_but_too_long": excl, "q_sft": len(sft["short"]),
                   "q_sft_share": round(len(sft["short"]) / len(by), 4)}
    lv = collections.Counter(str(r["level"]) if r["dataset"] == "mathtrain" else "gsm8k"
                             for r in sft["short"])
    pool_lv = collections.Counter(str(q.get("level")) if q["dataset"] == "mathtrain" else "gsm8k"
                                  for q in pool.values())
    stats["q_sft_mix"] = {k: {"kept": lv[k], "pool": pool_lv[k],
                              "share": round(lv[k] / pool_lv[k], 4)} for k in sorted(pool_lv)}
    stats["eligible_rollouts"] = {
        pol: sum(eligible_nolen(r) and fits(r) for v in by.values() for r in v[pol])
        for pol in (BASE, STEER)}
    stats["steered_yield"] = {"desc": "P(random eligible steered < shortest of eligible base), "
                                      "over Q_SFT", "value": round(yield_hits / max(yield_n, 1), 4)}
    for arm, rows in sft.items():
        path = args.out / f"sft_{arm}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps({"question_id": r["question_id"], "dataset": r["dataset"],
                                    "level": r["level"], "policy": r["policy"], "seed": r["seed"],
                                    "prompt_ids": prompts[r["question_id"]],
                                    "completion_ids": r["ids"]}) + "\n")
        lens = [r["len"] for r in rows]
        stats[f"sft_{arm}"] = {"n": len(rows), "completion_tokens": sum(lens),
                               "total_tokens": sum(lens) + sum(len(prompts[r["question_id"]])
                                                               for r in rows),
                               "mean_len": round(st.mean(lens), 1),
                               "median_len": st.median(lens)}
    stats["dpo_counts_before_matching"] = dpo_counts_raw
    for arm, types in dpo.items():
        path = args.out / f"dpo_{arm}.jsonl"
        n_tok = 0
        with path.open("w") as f:
            for t in ("L", "K", "P"):
                for r in types[t]:
                    n_tok += r["chosen"]["len"] + r["rejected"]["len"]
                    f.write(json.dumps({
                        "question_id": r["question_id"], "dataset": r["dataset"],
                        "level": r["level"], "type": t,
                        "prompt_ids": prompts[r["question_id"]],
                        "chosen_ids": r["chosen"]["ids"], "rejected_ids": r["rejected"]["ids"],
                        "chosen": {k: r["chosen"][k] for k in ("policy", "seed", "len")},
                        "rejected": {k: r["rejected"][k] for k in ("policy", "seed", "len")},
                    }) + "\n")
        stats[f"dpo_{arm}"] = {**{t: len(x) for t, x in types.items()},
                               "pairs": sum(len(x) for x in types.values()),
                               "completion_tokens": n_tok,
                               "mean_chosen_len": {t: round(st.mean([r["chosen"]["len"] for r in x]), 1)
                                                   for t, x in types.items() if x},
                               "mean_rejected_len": {t: round(st.mean([r["rejected"]["len"] for r in x]), 1)
                                                     for t, x in types.items() if x}}
    (args.out / "sets_stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
