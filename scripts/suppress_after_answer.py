#!/usr/bin/env python3
"""Suppress doubt only AFTER the model has already written the correct answer.

Findings 5/6 inject at every paragraph break of the whole trace. `loop_metrics.py`
then showed where the waste actually sits: 69% of baseline reasoning happens
after the gold value has already been written, and the every-boundary
intervention only brings that to 53%.

This targets the condition instead of the symptom. Generation runs untouched
until the gold value first appears; from that token on, the same direction is
injected at that token and at every subsequent paragraph break. Injection sites,
layer, alpha and direction are all unchanged from Finding 6 — only the START
condition differs, so the comparison isolates one variable.

ORACLE PROBE, NOT A METHOD. Triggering on the *gold* value uses information the
model does not have. The scientific question — is post-answer reasoning removable
without damage? — is answerable this way; a deployable version would trigger on
the model's own first stated answer and is the follow-up.

Traces that never state the gold get no injection and are reported separately;
including them in the paired analysis would dilute a real effect with rows where
nothing happened.

    uv run python scripts/suppress_after_answer.py \
        --traces data/after_answer_400.jsonl --direction data/pool/dir_A_1trace.pt
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suppress_answer import NEWLINE_CHAR, doubt_stats, sample_next  # noqa: E402

from reasoning_attention.config import MODEL_ID, NLAConfig, SamplingDefaults  # noqa: E402
from reasoning_attention.grading import grade  # noqa: E402
from reasoning_attention.loops import gold_span  # noqa: E402
from reasoning_attention.nla.arch import inner_transformer  # noqa: E402
from reasoning_attention.tokenview import _chat_header  # noqa: E402

WINDOW = 48  # tokens of rolling context searched for the gold value

# A number is only an answer claim if something in front of it says so. Without
# this, the trigger fires on premises ("a test of 100 questions") and on
# intermediate quantities, neither of which is the model stating its answer.
ANSWER_CUE = re.compile(
    r"(?:=|\bis\b|\bare\b|\bwas\b|\bwere\b|\bequals?\b|\bgives?\b|\bgets?\b|"
    r"\btotal\b|\banswer\b|\bso\b|\btherefore\b|\bthus\b|\bleft\b)"
    r"[^0-9A-Za-z]{0,12}$",
    re.IGNORECASE,
)


def gold_occurrences(text: str, gold: str) -> list[tuple[int, int]]:
    """Every standalone occurrence, using gold_span's own boundary rules."""
    g = str(gold).strip()
    if not g:
        return []
    return [m.span() for m in
            re.finditer(rf"(?<![\w.]){re.escape(g)}(?!\w)(?!\.\d)", text)]


def answer_claim_span(text: str, gold: str, *, require_complete: bool = False
                      ) -> tuple[int, int] | None:
    """First occurrence of `gold` that reads as an answer claim.

    `require_complete` is for STREAMING use: at the end of a partial buffer every
    number looks word-final, so "the limit is 100" matches gold 100 one token
    before the model writes "1000". Requiring at least one decoded character
    after the match lets gold_span's own `(?!\w)` guard do its job.
    """
    for a, b in gold_occurrences(text, gold):
        if require_complete and b >= len(text):
            continue
        if ANSWER_CUE.search(text[max(0, a - 40):a]):
            return (a, b)
    return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/after_answer_400.jsonl"))
    p.add_argument("--direction", type=Path, default=Path("data/pool/dir_A_1trace.pt"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--dump", type=Path, required=True)
    p.add_argument("--cond", default="after", choices=["after", "none"])
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=400)
    p.add_argument("--batch", type=int, default=4)
    p.add_argument("--max-new-tokens", type=int, default=12288)
    p.add_argument("--seed", type=int, default=0)
    # Injecting AT the trigger token would add a mid-paragraph intervention site
    # that Findings 5/6 never had, so "only the start condition changed" would
    # stop being true. Default keeps the site rule identical: paragraph breaks
    # only, just starting later.
    p.add_argument("--inject-at-trigger", action="store_true",
                   help="also inject on the trigger token itself (off by default)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = NLAConfig()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    traces = [json.loads(x) for x in args.traces.open()][: args.limit]
    traces.sort(key=lambda t: len(t["response"]))
    print(f"{len(traces)} questions, condition={args.cond}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    trunk = inner_transformer(model)
    unit = torch.load(args.direction, map_location="cpu", weights_only=False)["unit"]
    unit = unit.to(model.device, torch.float32)

    n_logits = int(model.config.vocab_size)
    pieces = tok.convert_ids_to_tokens(list(range(len(tok))))
    break_mask = torch.zeros(n_logits, dtype=torch.bool, device=model.device)
    for i, piece in enumerate(pieces):
        if piece and piece.count(NEWLINE_CHAR) >= 2:
            break_mask[i] = True
    think_close = int(tok.convert_tokens_to_ids("</think>"))
    eos_ids = {int(tok.eos_token_id), int(tok.convert_tokens_to_ids("<|endoftext|>"))}

    state: dict[str, Any] = {"mask": None, "dirs": None, "alpha": 0.0}

    def hook(_m: Any, _i: Any, output: Any) -> Any:
        mask = state["mask"]
        if mask is None or state["dirs"] is None or not bool(mask.any()):
            return output
        hidden = output[0] if isinstance(output, tuple) else output
        hidden = hidden.clone()
        h = hidden[:, -1, :]
        push = (state["alpha"] * h.float().norm(dim=-1, keepdim=True)) * state["dirs"]
        hidden[:, -1, :] = h + torch.where(
            mask[:, None], push.to(h.dtype), torch.zeros_like(h, dtype=h.dtype)
        )
        return (hidden, *output[1:]) if isinstance(output, tuple) else hidden

    trunk.layers[cfg.extraction_layer].register_forward_hook(hook)
    sampling = SamplingDefaults()

    partial = Path(str(args.out) + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    str_cols = {"question_id", "dataset", "gold", "band", "status"}
    if partial.exists():
        rows = [{k: v if k in str_cols else (float(v) if "." in v else int(v))
                 for k, v in r.items()} for r in csv.DictReader(partial.open())]
        print(f"resuming: {len(rows)} done")
    done = {r["question_id"] for r in rows}
    dump = args.dump.open("a" if rows else "w")
    pf = partial.open("a" if rows else "w", newline="")
    pw: Any = csv.DictWriter(pf, fieldnames=list(rows[0])) if rows else None
    t0 = time.time()

    todo = [t for t in traces if t["question_id"] not in done]
    for start in range(0, len(todo), args.batch):
        chunk = todo[start : start + args.batch]
        prompts = [_chat_header(tok, t["question"]) for t in chunk]
        golds = [str(t["gold"]).strip() for t in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left").to(model.device)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        B = ids.shape[0]
        pos = attn.long().cumsum(-1) - 1
        pos.masked_fill_(attn == 0, 1)
        torch.manual_seed(args.seed + start)

        state["dirs"] = None if args.cond == "none" else unit[None].expand(B, -1)
        state["alpha"] = args.alpha
        state["mask"] = None
        past: Any = None
        cur, cur_pos = ids, pos
        fin = torch.zeros(B, dtype=torch.bool, device=model.device)
        in_think = torch.ones(B, dtype=torch.bool, device=model.device)
        trig = torch.zeros(B, dtype=torch.bool, device=model.device)
        inject = torch.zeros(B, dtype=torch.bool, device=model.device)
        n_inj = torch.zeros(B, dtype=torch.long, device=model.device)
        out: list[list[int]] = [[] for _ in range(B)]
        think_len = [-1] * B
        trig_at = [-1] * B

        for step in range(args.max_new_tokens):
            state["mask"] = inject if step else None
            # no_grad is load-bearing: without it the autograd graph accumulates
            # across every decode step and OOMs within a few hundred tokens.
            with torch.no_grad():
                res = model(input_ids=cur, attention_mask=attn, position_ids=cur_pos,
                            past_key_values=past, use_cache=True, logits_to_keep=1)
                past = res.past_key_values
                nxt = sample_next(res.logits[:, -1], sampling.temperature,
                                  sampling.top_p, sampling.top_k)
            alive = ~fin
            for i in range(B):
                if alive[i]:
                    out[i].append(int(nxt[i]))
            closing = alive & (nxt == think_close)
            for i in range(B):
                if closing[i] and think_len[i] < 0:
                    think_len[i] = len(out[i])

            # trigger: gold value newly visible in the rolling tail
            just = torch.zeros(B, dtype=torch.bool, device=model.device)
            for i in range(B):
                if not alive[i] or trig[i] or not in_think[i]:
                    continue
                tail = tok.decode(out[i][-WINDOW:], skip_special_tokens=True)
                if answer_claim_span(tail, golds[i], require_complete=True):
                    trig[i] = True
                    just[i] = True
                    trig_at[i] = len(out[i])

            sites = break_mask[nxt] | just if args.inject_at_trigger else break_mask[nxt]
            inject = alive & in_think & trig & sites
            n_inj += inject.long()
            in_think &= ~closing
            for e in eos_ids:
                fin |= nxt == e
            if bool(fin.all()):
                break
            cur = nxt[:, None]
            cur_pos = cur_pos[:, -1:] + 1
            attn = torch.cat([attn, torch.ones(B, 1, dtype=attn.dtype, device=model.device)], 1)

        texts = [tok.decode(o, skip_special_tokens=False) for o in out]
        n_before = len(rows)
        for i, t in enumerate(chunk):
            g = grade(texts[i], t["gold"])
            nm, nb, nd = doubt_stats(texts[i])
            total = len(out[i])
            # Two different quantities, kept apart because they are not
            # comparable: `post_share` is the CHARACTER share of the <think>
            # body after the gold value first appears -- the same definition
            # loop_metrics.py uses for its 69% baseline. `post_tok_share` is the
            # TOKEN share of the whole completion after the trigger fired, which
            # includes the post-</think> answer text and so reads lower.
            body = texts[i].split("</think>")[0]
            sp = gold_span(body, str(t["gold"]))
            post = 1.0 - sp[0] / max(len(body), 1) if sp else -1.0
            post_tok = (total - trig_at[i]) / total if trig_at[i] >= 0 and total else -1.0
            qtext = t["question"]
            first = gold_occurrences(body, str(t["gold"]))
            rows.append({
                "question_id": t["question_id"], "dataset": t["dataset"],
                "gold": str(t["gold"]), "band": t["band"],
                "correct": int(g.is_correct), "has_answer": int(g.has_answer),
                "status": g.status, "tokens": total,
                "think_tokens": think_len[i], "capped": int(not bool(fin[i])),
                "triggered": int(trig_at[i] >= 0), "trigger_at": trig_at[i],
                "post_share": round(post, 4), "post_tok_share": round(post_tok, 4),
                "gold_in_question": int(bool(gold_occurrences(qtext, str(t["gold"])))),
                "gold_occurrences": len(first),
                "injections": int(n_inj[i]),
                "markers": nm, "blocks": nb, "doubt_blocks": nd,
            })
            dump.write(json.dumps({"question_id": t["question_id"], "gold": t["gold"],
                                   "band": t["band"], args.cond: texts[i]}) + "\n")
        dump.flush()
        if pw is None:
            pw = csv.DictWriter(pf, fieldnames=list(rows[0]))
            pw.writeheader()
        pw.writerows(rows[n_before:])
        pf.flush()
        os.fsync(pf.fileno())
        d = len(rows)
        tr = sum(r["triggered"] for r in rows)
        co = sum(r["correct"] for r in rows)
        print(f"[{d}/{len(traces)}] {(time.time()-t0)/max(d-len(done),1):.1f}s/trace  "
              f"correct {co}/{d}  triggered {tr}/{d}", flush=True)

    dump.close()
    pf.close()
    os.replace(partial, args.out)
    print(f"\nwrote {args.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
