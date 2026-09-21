#!/usr/bin/env python3
"""When does the model first HAVE an answer? Ask it, don't pattern-match.

Finding 9 triggered on the gold value appearing in the text. That is an oracle,
and RESEARCH-DIRECTION.md's audit showed the heuristic also fires on premises,
on intermediate quantities and (before the completeness guard) on half-written
numbers. Every one of those failure modes comes from the same mistake: guessing
the model's candidate answer from its prose.

This asks the model instead. At each paragraph boundary inside `<think>` it
appends `</think>` and an answer stem, and reads what the model puts in the box.
Whatever comes out IS its candidate answer at that boundary -- by construction,
with no regex, no gold, and no guessing.

Forcing the box means every boundary returns *something*, so "it produced an
answer" cannot be the detection rule. The rule is STABILITY: the candidate is
detected at the first boundary whose forced answer is repeated at the next
boundary. Before the model has settled, consecutive probes disagree; once it has
a candidate, they stop disagreeing. That criterion never looks at gold -- gold is
used only afterwards, to label the detected candidate correct or incorrect.

The forcing stem is needed because an unforced `</think>` often produces a
complete, correct, and entirely unboxed response ("**Answer:** 100 words"),
which the `\\boxed{}`-only grader scores as no answer at all.

The by-product is the `commit` comparator of the fixed-prefix experiment: the
answer the model would have given had it stopped thinking right there.

Reads the Finding 9 baseline traces, so no trace generation is needed. Nothing
is injected, so this runs on vLLM.

    uv run python scripts/probe_candidates.py --limit 400
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suppress_answer import NEWLINE_CHAR  # noqa: E402

from reasoning_attention.config import MODEL_ID, VLLMConfig  # noqa: E402
from reasoning_attention.grading import grade  # noqa: E402
from reasoning_attention.tokenview import _chat_header  # noqa: E402

# What gets appended to a frozen prefix to read off the model's candidate. The
# open brace is part of the stem: the model only has to fill in the value.
STEM = "\n</think>\n\n**Final Answer:** $\\boxed{"


def boxed_value(text: str) -> str:
    """The contents of the forced box, up to its matching close.

    An UNCLOSED box returns "" -- unknown, not a candidate. Accepting the
    truncated prefix instead would turn a probe that ran out of budget into an
    answer, and two such probes could then "agree" on a half-written expression.
    """
    depth = 1
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[:i].strip()
        elif ch in "\n$" and depth == 1:
            # A bare newline or `$` at depth 1 closes the expression in practice;
            # anything still nested is an incomplete \frac{..}{..} and is unknown.
            return text[:i].strip()
    return ""


def normalize(v: str) -> str:
    """Enough normalization that `1,100`, `1100 ` and `1100.0` are one answer.

    Braces are PRESERVED. An earlier version stripped every `}`, which turned
    `\\sqrt{2}` into the malformed `\\sqrt{2` -- harmless on integer-answer GSM8K
    but fatal for any symbolic answer, so it silently capped the detector at one
    dataset.
    """
    v = v.replace(",", "").replace("\\!", "").replace("\\,", "").replace("$", "")
    v = v.replace("\\ ", " ").strip()
    # \text{ minutes} is a unit, not part of the value.
    v = re.sub(r"\\(?:text|mathrm|mbox)\s*\{[^{}]*\}", "", v)
    v = " ".join(v.split())
    if re.fullmatch(r"-?\d+\.0+", v):
        v = v.split(".")[0]
    return v.rstrip(".") if re.fullmatch(r"-?[\d.]+\.", v) else v


def same_answer(a: str, b: str) -> bool:
    """Do two probes name the same answer?

    String equality first because it is free and settles almost everything, then
    symbolic equivalence, so `0.5` and `\\frac{1}{2}` count as agreement instead
    of resetting the stability run. Erring toward "different" only delays
    detection, so a failed comparison is a non-match.
    """
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        return bool(grade("\\boxed{" + a + "}", b).is_correct)
    except Exception:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/after_answer_400.jsonl"))
    p.add_argument("--texts", type=Path, default=Path("data/afteranswer/baseline_texts.jsonl"))
    p.add_argument("--key", default="none", help="which arm in the texts dump to read")
    p.add_argument("--out", type=Path, default=Path("data/prefix/candidates.jsonl"))
    p.add_argument("--limit", type=int, default=400)
    # A boundary budget, not a token budget: the detection point has to be found
    # early enough that a continuation experiment still has something to remove.
    p.add_argument("--max-boundaries", type=int, default=24)
    p.add_argument("--max-prefix-tokens", type=int, default=2048)
    # 32 was enough for an integer but truncates a symbolic answer mid-brace,
    # and a truncated box is now discarded rather than guessed at.
    p.add_argument("--probe-tokens", type=int, default=96)
    p.add_argument("--chunk", type=int, default=40, help="questions per checkpoint")
    p.add_argument("--gpu-util", type=float, default=0.86)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    meta = {t["question_id"]: t for t in (json.loads(x) for x in args.traces.open())}
    texts = [json.loads(x) for x in args.texts.open()][: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)

    done: set[str] = set()
    if args.out.exists():
        done = {json.loads(x)["question_id"] for x in args.out.open()}
        print(f"resuming: {len(done)} questions already probed")
    todo = [r for r in texts if r["question_id"] not in done and r["question_id"] in meta]
    print(f"{len(texts)} traces, {len(todo)} to probe")
    if not todo:
        return

    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))

    def boundaries(ids: list[int]) -> list[int]:
        """Token indices carrying a paragraph break -- the injection sites."""
        return [i for i, t in enumerate(ids) if pieces[t] and pieces[t].count(NEWLINE_CHAR) >= 2]

    from reasoning_attention.serving.vllm_server import build_llm
    from vllm import SamplingParams

    cfg = replace(VLLMConfig(), gpu_memory_utilization=args.gpu_util,
                  max_model_len=args.max_prefix_tokens + 1024)
    llm = build_llm(cfg)
    # Greedy: the candidate must be a property of the prefix, not of a seed.
    sp = SamplingParams(temperature=0.0, max_tokens=args.probe_tokens)

    fh = args.out.open("a")
    for start in range(0, len(todo), args.chunk):
        chunk = todo[start : start + args.chunk]
        prompts: list[str] = []
        index: list[tuple[int, int]] = []  # (row in chunk, boundary token index)
        per: list[dict[str, Any]] = []
        for j, r in enumerate(chunk):
            t = meta[r["question_id"]]
            body = r[args.key].split("</think>")[0]
            ids = tok(body, add_special_tokens=False)["input_ids"]
            bs = [b for b in boundaries(ids) if b < args.max_prefix_tokens][: args.max_boundaries]
            header = _chat_header(tok, t["question"])
            per.append({"ids": ids, "bs": bs, "header": header, "meta": t})
            for b in bs:
                prefix = tok.decode(ids[: b + 1], skip_special_tokens=False)
                prompts.append(header + prefix.rstrip() + STEM)
                index.append((j, b))
        if not prompts:
            continue
        outs = llm.generate(prompts, sp)

        by_row: dict[int, list[dict[str, Any]]] = {j: [] for j in range(len(chunk))}
        for (j, b), o in zip(index, outs):
            raw = o.outputs[0].text
            cand = normalize(boxed_value(raw))
            # Correctness is a LABEL, computed after detection, never used by it.
            g = grade("\\boxed{" + cand + "}", per[j]["meta"]["gold"]) if cand else None
            by_row[j].append({
                "boundary": b, "candidate": cand,
                "correct": int(bool(g and g.is_correct)), "raw": raw[:120],
            })

        for j, r in enumerate(chunk):
            probes = by_row[j]
            t = per[j]["meta"]
            first = next((p for p in probes if p["candidate"]), None)
            # DETECTION: first boundary whose forced answer survives one more
            # boundary. Gold plays no part in this.
            runs: list[tuple[int, int]] = []  # (start index, run length)
            i = 0
            while i < len(probes):
                j2 = i
                while (j2 + 1 < len(probes)
                       and same_answer(probes[j2 + 1]["candidate"], probes[i]["candidate"])):
                    j2 += 1
                if probes[i]["candidate"]:
                    runs.append((i, j2 - i + 1))
                i = j2 + 1
            stable = next((probes[a] for a, n in runs if n >= 2), None)
            run_len = next((n for _, n in runs if n >= 2), 0)
            # A run of 3 is the sensitivity check: a pair can agree by accident
            # while the model is still moving, as `8, 8, 32, 32, ...` does.
            stable3 = next((probes[a] for a, n in runs if n >= 3), None)
            fh.write(json.dumps({
                "question_id": r["question_id"], "dataset": t["dataset"],
                "gold": str(t["gold"]), "band": r.get("band", t.get("band", "-")),
                "n_boundaries": len(probes),
                "think_tokens": len(per[j]["ids"]),
                "first_boundary": first["boundary"] if first else -1,
                "first_candidate": first["candidate"] if first else "",
                "first_correct": first["correct"] if first else -1,
                "stable_boundary": stable["boundary"] if stable else -1,
                "stable_candidate": stable["candidate"] if stable else "",
                "stable_correct": stable["correct"] if stable else -1,
                "stable_run": run_len,
                "stable3_boundary": stable3["boundary"] if stable3 else -1,
                "stable3_candidate": stable3["candidate"] if stable3 else "",
                "stable3_correct": stable3["correct"] if stable3 else -1,
                "probes": probes,
            }) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
        print(f"[{start + len(chunk)}/{len(todo)}] {len(prompts)} probes", flush=True)
    fh.close()
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
