"""Blind audit of reasoning quality in KEPT (correct) rollouts: steered N vs base.

The pipeline keeps any rollout whose boxed answer matches gold. A correct answer
does not certify the reasoning, and on hard problems N skips checks base makes
(Finding 12). This asks a judge model whether the kept traces are sound.

Design
- Pairs: MATH-500 (question, seed) where BOTH `base` and `N@a1.0d256` are correct,
  one pair per question, stratified by level (default 40 per level). Same question
  and seed, so difficulty is matched.
- Absolute grading (primary): every trace is judged ALONE, in its own call, with
  the problem and gold answer. The judge never sees the other arm, so it cannot
  compare lengths. Rubric: uncorrected errors, missed conditions, whether the
  reasoning establishes the answer, "lucky" answers, soundness 1-5, and whether it
  is an acceptable training example.
- Pairwise (secondary): both traces side by side, asked in BOTH orders to cancel
  position bias. Length is visible here, so this is the weaker measurement.
- Calibration: wrong-answer traces from both arms, graded the same way. A judge
  that marks them justified is not detecting anything.
- Blindness: prompts carry no arm, policy or id; the item -> arm key is a separate
  file; item order is shuffled.

    uv run python scripts/audit_trace_quality.py build   # writes items + key
    uv run python scripts/audit_trace_quality.py judge --limit 6   # pilot
    uv run python scripts/audit_trace_quality.py judge   # full run, resumable
    uv run python scripts/audit_trace_quality.py report
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re

import numpy as np
import pandas as pd

OUT = "data/eval_reports/trace_audit"
ROLL = "data/eval_math500/stage2_shard*/rollouts.jsonl"
COHORT = "data/policy/math500.jsonl"
ARMS = ("base", "N@a1.0d256")

ABSOLUTE = """You are auditing a worked solution written by a small language model, to decide whether it is fit to be used as a training example. The model reasons inside <think>...</think> and then writes a final answer.

Grade the REASONING. Do not reward or penalize length or style. Exploration, hesitation and self-correction are fine when the final line of reasoning is valid. An error that the solution itself later catches and fixes does not count.

<problem>
{question}
</problem>

<reference_answer>
{gold}
</reference_answer>

<solution>
{text}
</solution>

Return a JSON object inside <json></json> with exactly these fields:
- "final_answer_matches": true if the solution's final answer is equivalent to the reference answer.
- "uncorrected_errors": list of errors that remain in the line of reasoning the final answer rests on. Each item: {{"quote": "<at most 25 words from the solution>", "type": "arithmetic" | "algebraic" | "logical" | "unjustified_claim" | "other", "affects_answer": true | false}}. Empty list if none.
- "missed_conditions": list of short strings, one per condition or case the problem requires and the solution never checks (domain restrictions, excluded or extraneous values, boundary cases, verifying every constraint). Empty list if none.
- "answer_justified": "full" if the reasoning as written establishes the final answer, "partial" if it reaches it with a gap, "none" if it does not support it.
- "lucky": true if the final answer is right but the reasoning leading to it is wrong or incomplete in a way that would ordinarily produce a different answer.
- "soundness": integer 1-5. 5 = every step the answer depends on is valid and every required case is handled; 3 = correct approach with a real gap or unverified step; 1 = the answer is not supported.
- "training_example_ok": true if you would accept this solution as a worked example to teach a student.
"""

PAIRWISE = """Two solutions to the same problem, written by a small language model, both reaching the reference answer. Each reasons inside <think>...</think> and then writes a final answer.

Decide which is the better worked example for teaching a student to reason CORRECTLY and RIGOROUSLY: valid steps, every required case handled, the answer actually established. Do not prefer a solution for being longer or shorter; judge only the soundness of the reasoning. Answer "tie" if they are equally sound.

<problem>
{question}
</problem>

<reference_answer>
{gold}
</reference_answer>

<solution_A>
{a}
</solution_A>

<solution_B>
{b}
</solution_B>

Return a JSON object inside <json></json>: {{"better": "A" | "B" | "tie", "reason": "<one sentence>"}}
"""


def _load() -> tuple[pd.DataFrame, dict]:
    rows = [json.loads(line) for f in sorted(glob.glob(ROLL)) for line in open(f)]
    df = pd.DataFrame([r for r in rows if r["policy"] in ARMS])
    cohort = {(r := json.loads(line))["question_id"]: r for line in open(COHORT)}
    return df, cohort


def build(args: argparse.Namespace) -> None:
    os.makedirs(OUT, exist_ok=True)
    rng = random.Random(args.seed)
    df, cohort = _load()
    df["level"] = df.question_id.map(lambda q: cohort[q]["level"])
    by = {(r.question_id, r.policy, r.seed): r for r in df.itertuples()}

    pairs = []
    for level in sorted(df.level.unique()):
        cands = {}
        for (q, pol, s), r in by.items():
            if pol != "base" or r.level != level or not r.correct:
                continue
            n = by.get((q, "N@a1.0d256", s))
            if n is not None and n.correct:
                cands.setdefault(q, []).append(s)
        qs = sorted(cands)
        rng.shuffle(qs)
        for q in sorted(qs[: args.per_level]):
            pairs.append((q, rng.choice(sorted(cands[q])), int(level)))

    # Calibration: wrong-answer traces from hard levels, half per arm.
    wrong = sorted(
        (r.question_id, r.policy, r.seed)
        for r in df.itertuples()
        if not r.correct and r.level >= 4 and r.has_answer and r.status == "ok"
    )
    calib = []
    for arm in ARMS:
        pool = [w for w in wrong if w[1] == arm]
        calib += rng.sample(pool, args.calibration // 2)

    items, key = [], {}

    def new_id() -> str:
        return "%012x" % rng.getrandbits(48)

    def add(kind: str, prompt: str, meta: dict) -> None:
        iid = new_id()
        items.append({"id": iid, "kind": kind, "prompt": prompt})
        key[iid] = {"kind": kind, **meta}

    for q, s, level in pairs:
        c = cohort[q]
        tb, tn = by[(q, "base", s)].text, by[(q, "N@a1.0d256", s)].text
        for arm, t in (("base", tb), ("N@a1.0d256", tn)):
            add(
                "absolute",
                ABSOLUTE.format(question=c["question"], gold=c["gold"], text=t),
                {"question_id": q, "seed": s, "level": level, "arm": arm, "tokens": len(t)},
            )
        for first in ARMS:  # both orders
            a, b = (tb, tn) if first == "base" else (tn, tb)
            add(
                "pairwise",
                PAIRWISE.format(question=c["question"], gold=c["gold"], a=a, b=b),
                {"question_id": q, "seed": s, "level": level, "A": first},
            )
    for q, arm, s in calib:
        c = cohort[q]
        add(
            "calibration",
            ABSOLUTE.format(question=c["question"], gold=c["gold"], text=by[(q, arm, s)].text),
            {"question_id": q, "seed": s, "level": cohort[q]["level"], "arm": arm},
        )

    rng.shuffle(items)
    with open(f"{OUT}/items.jsonl", "w") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    with open(f"{OUT}/key.json", "w") as f:
        json.dump(key, f, indent=1)
    chars = sum(len(it["prompt"]) for it in items)
    kinds = pd.Series([it["kind"] for it in items]).value_counts().to_dict()
    print(f"{len(pairs)} pairs; items {kinds}; ~{chars / 3.6 / 1e6:.1f}M input tokens")
    print(f"wrote {OUT}/items.jsonl and {OUT}/key.json (the key never goes to the judge)")


def _parse(text: str | None) -> dict | None:
    if not text:
        return None
    # Tagged JSON; failing that, the outermost {...} (the judge sometimes drops the tags).
    m = re.search(r"<json>\s*(\{.*\})\s*</json>", text, re.S) or re.search(
        r"(\{.*\})", text, re.S
    )
    if not m:
        return None
    # Judges quote LaTeX (`\(`, `\frac`, `\neq`) inside JSON strings. Every backslash
    # except an escaped quote or an already-doubled backslash is LaTeX: `\f` and `\n`
    # are valid JSON escapes that would silently eat the command, `\u` invalid ones.
    body = re.sub(
        r'\\\\|\\"|\\', lambda t: t.group(0) if len(t.group(0)) == 2 else "\\\\", m.group(1)
    )
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def judge(args: argparse.Namespace) -> None:
    from reasoning_attention.config import ExplainerConfig
    from reasoning_attention.datagen.providers import OpenAIProvider

    items = [json.loads(line) for line in open(f"{OUT}/items.jsonl")]
    path = f"{OUT}/judgments.jsonl"
    done = set()
    if os.path.exists(path):
        done = {json.loads(line)["id"] for line in open(path) if _parse(json.loads(line)["raw"])}
    todo = [it for it in items if it["id"] not in done]
    if args.kind:
        todo = [it for it in todo if it["kind"] == args.kind]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(done)} done, {len(todo)} to judge")
    provider = OpenAIProvider(
        ExplainerConfig(
            reasoning_effort=args.effort,
            max_output_tokens=args.max_output,
            concurrency=args.concurrency,
        )
    )
    for i in range(0, len(todo), args.chunk):
        chunk = todo[i : i + args.chunk]
        outs = provider.complete([it["prompt"] for it in chunk])
        with open(path, "a") as f:
            for it, raw in zip(chunk, outs, strict=True):
                f.write(json.dumps({"id": it["id"], "raw": raw}) + "\n")
        bad = sum(_parse(o) is None for o in outs)
        print(f"  {i + len(chunk)}/{len(todo)} judged ({bad} unparsed in this chunk)", flush=True)


def _boot(x: np.ndarray, b: int = 10_000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    m = x[rng.integers(0, len(x), size=(b, len(x)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def report(args: argparse.Namespace) -> None:
    key = json.load(open(f"{OUT}/key.json"))
    parsed: dict[str, dict] = {}  # last parseable judgment per item (retries append)
    for line in open(f"{OUT}/judgments.jsonl"):
        r = json.loads(line)
        j = _parse(r["raw"])
        if j is not None:
            parsed[r["id"]] = {**key[r["id"]], **j}
    rows = list(parsed.values())
    df = pd.DataFrame(rows)
    ab = df[df.kind == "absolute"].copy()
    ab["major"] = ab.uncorrected_errors.map(lambda e: any(x.get("affects_answer") for x in e))
    ab["n_missed"] = ab.missed_conditions.map(len)
    ab["full"] = ab.answer_justified == "full"
    metrics = ["soundness", "full", "lucky", "major", "n_missed", "training_example_ok"]

    print(f"absolute judgments: {len(ab)}  (pairs with both arms judged below)")
    w = ab.pivot_table(index=["question_id", "level"], columns="arm", values=metrics, aggfunc="first")
    if not all((m, arm) in w.columns for m in metrics for arm in ARMS):
        w = w.iloc[0:0]
    w = w.dropna()
    print(f"{len(w)} complete pairs\n")
    if len(w):
        print(f"{'metric':22s} {'base':>7s} {'N':>7s}   N - base [95% CI]")
    for lab, sel in (("all", None), ("L1-3", [1, 2, 3]), ("L4", [4]), ("L5", [5])):
        s = w if sel is None else w[w.index.get_level_values("level").isin(sel)]
        if not len(s):
            continue
        print(f"-- {lab} (n={len(s)})")
        for m in metrics:
            bv = s[(m, "base")].astype(float).to_numpy()
            nv = s[(m, "N@a1.0d256")].astype(float).to_numpy()
            lo, hi = _boot(nv - bv)
            print(f"  {m:20s} {bv.mean():7.3f} {nv.mean():7.3f}   {(nv - bv).mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")

    pw = df[df.kind == "pairwise"].copy()
    if len(pw):
        # Score from N's perspective: +1 N better, -1 base better, 0 tie; average both orders.
        def n_score(r: pd.Series) -> float:
            if r.better == "tie":
                return 0.0
            winner = r.A if r.better == "A" else ("base" if r.A != "base" else "N@a1.0d256")
            return 1.0 if winner == "N@a1.0d256" else -1.0

        pw["n_score"] = pw.apply(n_score, axis=1)
        pq = pw.groupby(["question_id", "level"]).n_score.mean()
        print("\n-- pairwise (N perspective: +1 N better, -1 base better; both orders averaged)")
        for lab, sel in (("all", None), ("L1-3", [1, 2, 3]), ("L4", [4]), ("L5", [5])):
            s = pq if sel is None else pq[pq.index.get_level_values("level").isin(sel)]
            lo, hi = _boot(s.to_numpy())
            print(f"  {lab:5s} n={len(s):3d}  mean {s.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
        flips = pw.groupby("question_id").apply(lambda g: g.n_score.nunique() > 1).mean()
        print(f"  order sensitivity: verdict changes with order on {flips:.0%} of pairs")

    cal = df[df.kind == "calibration"]
    if len(cal):
        print("\n-- calibration (wrong-answer traces; should NOT be justified)")
        print(f"  n={len(cal)}  final_answer_matches {cal.final_answer_matches.mean():.2f}  "
              f"answer_justified=full {(cal.answer_justified == 'full').mean():.2f}  "
              f"training_example_ok {cal.training_example_ok.mean():.2f}  "
              f"soundness {cal.soundness.mean():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--per-level", type=int, default=40)
    b.add_argument("--calibration", type=int, default=20)
    b.add_argument("--seed", type=int, default=20260925)
    j = sub.add_parser("judge")
    j.add_argument("--limit", type=int, default=0)
    j.add_argument("--kind", choices=("absolute", "pairwise", "calibration"), default=None)
    j.add_argument("--effort", default="high")
    j.add_argument("--max-output", type=int, default=32_000)
    j.add_argument("--concurrency", type=int, default=16)
    j.add_argument("--chunk", type=int, default=32)
    sub.add_parser("report")
    args = ap.parse_args()
    {"build": build, "judge": judge, "report": report}[args.cmd](args)


if __name__ == "__main__":
    main()
