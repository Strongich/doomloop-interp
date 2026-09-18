#!/usr/bin/env python3
"""Rebuild an after-answer CSV from its dumped texts, with corrected definitions.

The baseline arm injects nothing, so its generated text is unaffected by the
trigger bugs -- only the derived columns were wrong. This recomputes them from
the texts instead of spending two GPU-hours regenerating identical output.

Corrections applied:
  * trigger requires an ANSWER-CLAIM context, not merely the gold value appearing
    (a premise "a test of 100 questions" is not the model stating its answer)
  * post_share is the CHARACTER share of the <think> body after the gold value,
    matching loop_metrics.py's 69% baseline -- not a token share of the whole
    completion, which includes the post-</think> answer and reads lower

    uv run python scripts/rescore_after_answer.py --arm baseline
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suppress_after_answer import answer_claim_span, gold_occurrences  # noqa: E402
from suppress_answer import doubt_stats  # noqa: E402

from reasoning_attention.config import MODEL_ID  # noqa: E402
from reasoning_attention.grading import grade  # noqa: E402
from reasoning_attention.loops import gold_span  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", default="baseline")
    p.add_argument("--dir", type=Path, default=Path("data/afteranswer"))
    p.add_argument("--traces", type=Path, default=Path("data/after_answer_400.jsonl"))
    p.add_argument("--max-new-tokens", type=int, default=12288)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    meta = {json.loads(x)["question_id"]: json.loads(x) for x in args.traces.open()}
    recs = [json.loads(x) for x in (args.dir / f"{args.arm}_texts.jsonl").open()]
    print(f"{len(recs)} texts for arm {args.arm}")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    key = "none" if args.arm == "baseline" else "after"
    rows: list[dict[str, Any]] = []
    for r in recs:
        q = r["question_id"]
        t = r.get(key) or r.get("none") or r.get("after")
        m = meta[q]
        gold = str(m["gold"])
        body = t.split("</think>")[0]
        ids = tok(t, add_special_tokens=False)["input_ids"]
        think_ids = tok(body, add_special_tokens=False)["input_ids"] if "</think>" in t else None
        g = grade(t, gold)
        nm, nb, nd = doubt_stats(t)
        claim = answer_claim_span(body, gold)
        first = gold_span(body, gold)
        trig_char = claim[0] if claim else -1
        # token index of the trigger: tokenize the prefix up to the claim
        trig_tok = len(tok(body[:trig_char], add_special_tokens=False)["input_ids"]) \
            if trig_char >= 0 else -1
        rows.append({
            "question_id": q, "dataset": m["dataset"], "gold": gold, "band": m["band"],
            "correct": int(g.is_correct), "has_answer": int(g.has_answer),
            "status": g.status, "tokens": len(ids),
            "think_tokens": len(think_ids) if think_ids is not None else -1,
            "capped": int(len(ids) >= args.max_new_tokens - 2),
            "triggered": int(claim is not None), "trigger_at": trig_tok,
            "post_share": round(1.0 - first[0] / max(len(body), 1), 4) if first else -1.0,
            "post_tok_share": round((len(ids) - trig_tok) / len(ids), 4)
            if trig_tok >= 0 and ids else -1.0,
            "gold_in_question": int(bool(gold_occurrences(m["question"], gold))),
            "gold_occurrences": len(gold_occurrences(body, gold)),
            "injections": 0, "markers": nm, "blocks": nb, "doubt_blocks": nd,
        })
    out = args.dir / f"{args.arm}.csv"
    partial = Path(str(out) + ".partial")
    with partial.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {partial} ({len(rows)} rows)")
    n = len(rows)
    tr = sum(r["triggered"] for r in rows)
    pa = [r["post_share"] for r in rows if r["post_share"] >= 0]
    print(f"  triggered {tr}/{n} ({100*tr/n:.0f}%)   "
          f"post-answer share (chars in <think>) {100*sum(pa)/len(pa):.0f}%")
    if os.environ.get("FINALISE"):
        os.replace(partial, out)
        print(f"  finalised {out}")


if __name__ == "__main__":
    main()
