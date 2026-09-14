#!/usr/bin/env python3
"""Suppress the self-doubt at every block boundary of a whole trace, then grade it.

Finding 3 showed a one-token edit at one block boundary changes what the model
writes next. This asks the question that has something at stake: if the doubt is
suppressed at **every** boundary of a trace that was known to answer correctly,
and generation runs all the way to `</think>` and a `\\boxed{}`, does the answer
survive?

    accuracy holds  -> the doubt was decorative (the representational counterpart
                       to arXiv:2510.24941's length reduction, by intervention
                       rather than by pruning-and-retraining)
    accuracy drops  -> the doubt is load-bearing, which reverses the
                       "overthinking" framing for this model

Three conditions per trace, all regenerated from the same question prompt at the
rollout's own sampling settings (T=0.6, top_p=0.95, top_k=20) — the un-steered
condition is **re-generated**, never read off the original rollout, because
sampling is stochastic and the original was selected for being correct:

    none      no injection
    suppress  h -> h + alpha*||h||*u_suppress   at every paragraph break in <think>
    random    the same push along a random unit direction (matched norm)

`random` is what separates "suppressing doubt costs accuracy" from "poking the
residual stream 40 times costs accuracy".

## The direction

Per-probe Δ (Finding 3's recipe) needs the AV to verbalize each boundary online —
~250 generated tokens per boundary, tens of boundaries per trace, three
conditions. That is hours of AV decoding for a experiment whose signal is the
answer, so the direction is computed **once**, as the mean of the per-probe
Δ_suppress over the `doubt_wait` probes:

    u = normalize( mean_i [ AR(edit(e_i, CONTINUE)) - AR(e_i) ] )

The script prints the cosine of each Δ_i to that mean, so whether a single
direction is a fair summary is visible rather than assumed. `steer_sweep.py
--delta mean` measures the same substitution directly against per-probe Δ on the
suppression benchmark.

## The site

Injection happens at the token that *carries* the paragraph break — the position
whose next-token distribution chooses the first word of the next block. This is
the SAME site Finding 3 uses, not an approximation of it: Qwen3 tokenizes the
block-final period together with the break (".\n\n" is one token, "ĊĊ"), so the
"last token before the \n\n" that `block_boundaries` returns and the break token
coincide on 295 of the 297 `doubt_wait` probes. Online detection is therefore
exact rather than off-by-one.

    uv run python scripts/suppress_answer.py --limit 120 --alpha 1.0
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steer_demo import (  # noqa: E402
    CONTINUE_PARAGRAPH,
    DOUBT_PARAGRAPH,
    edit_explanation,
    load_ar,
    reconstruct,
)

from reasoning_attention.config import MODEL_ID, NLAConfig, SamplingDefaults  # noqa: E402
from reasoning_attention.grading import grade  # noqa: E402
from reasoning_attention.loops import DOUBT_MARKERS, marker_matches  # noqa: E402
from reasoning_attention.nla.arch import inner_transformer  # noqa: E402
from reasoning_attention.tokenview import _chat_header  # noqa: E402

NEWLINE_CHAR = "Ċ"  # byte-level BPE's stand-in for "\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/correct_sample.jsonl"))
    p.add_argument("--explanations", type=Path, default=Path("data/block_explanations_3way.csv"))
    p.add_argument("--base", default=MODEL_ID)
    p.add_argument("--ar", type=Path, default=Path("checkpoints/ar_rl_ep1"))
    p.add_argument("--out", type=Path, default=Path("data/suppress_answer.csv"))
    p.add_argument("--dump", type=Path, default=Path("data/suppress_answer_texts.jsonl"))
    p.add_argument("--direction", type=Path, default=Path("data/delta_suppress_mean.pt"))
    p.add_argument(
        "--edit",
        default="continue",
        choices=("continue", "doubt"),
        help="which paragraph the mean direction points toward. `continue` is "
        "suppression; `doubt` is available as the opposite-sign sanity check.",
    )
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=120)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--max-new-tokens", type=int, default=6144)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--conditions",
        nargs="+",
        default=["none", "suppress", "random"],
        choices=["none", "suppress", "random"],
    )
    p.add_argument(
        "--max-question-chars",
        type=int,
        default=2000,
        help="skip pathological prompts; nothing in the sample is near this",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# the direction
# --------------------------------------------------------------------------- #
def build_direction(args: argparse.Namespace) -> tuple[torch.Tensor, dict[str, float]]:
    """Mean of the per-probe Δ over the `doubt_wait` probes, as a unit vector."""
    import csv

    paragraph = CONTINUE_PARAGRAPH if args.edit == "continue" else DOUBT_PARAGRAPH
    probes = [
        r
        for r in csv.DictReader(args.explanations.open())
        if r["kind"] == "doubt_wait" and r["explanation"].count("\n\n") >= 1
    ]
    ar_tok, ar_backbone, affine = load_ar(args.ar)
    deltas = []
    with torch.no_grad():
        for row in probes:
            e0 = row["explanation"]
            try:
                edited = edit_explanation(e0, paragraph)
            except SystemExit:
                continue
            a = reconstruct(ar_tok, ar_backbone, affine, e0)
            b = reconstruct(ar_tok, ar_backbone, affine, edited)
            deltas.append((b - a).float().cpu())
    del ar_backbone, affine
    torch.cuda.empty_cache()

    d = torch.stack(deltas)  # [n, d_model]
    mean = d.mean(0)
    unit = mean / mean.norm()
    cos_to_mean = torch.nn.functional.cosine_similarity(d, mean[None], dim=-1)
    stats = {
        "n": float(len(d)),
        "mean_norm": float(d.norm(dim=-1).mean()),
        "norm_of_mean": float(mean.norm()),
        "cos_to_mean_mean": float(cos_to_mean.mean()),
        "cos_to_mean_min": float(cos_to_mean.min()),
        "cos_to_mean_p10": float(cos_to_mean.quantile(0.10)),
    }
    return unit, stats


# --------------------------------------------------------------------------- #
# batched decode with injection at paragraph breaks
# --------------------------------------------------------------------------- #
def sample_next(logits: torch.Tensor, temperature: float, top_p: float, top_k: int) -> torch.Tensor:
    """Temperature / top-k / top-p sampling, matching the rollout recipe."""
    logits = logits.float() / temperature
    k = min(top_k, logits.shape[-1])
    vals, idx = torch.topk(logits, k, dim=-1)  # already sorted descending
    probs = torch.softmax(vals, dim=-1)
    cum = probs.cumsum(-1)
    # keep the smallest prefix whose mass reaches top_p (always >= 1 token)
    drop = (cum - probs) >= top_p
    probs = probs.masked_fill(drop, 0.0)
    probs = probs / probs.sum(-1, keepdim=True)
    pick = torch.multinomial(probs, 1)
    return idx.gather(-1, pick).squeeze(-1)


@torch.no_grad()
def generate_batch(
    model: Any,
    tokenizer: Any,
    state: dict[str, Any],
    prompts: list[str],
    dirs: torch.Tensor | None,
    alpha: float,
    max_new: int,
    break_mask: torch.Tensor,
    think_close_id: int,
    eos_ids: set[int],
    sampling: SamplingDefaults,
) -> tuple[list[str], list[int], list[int], list[bool], list[int]]:
    """Sample to completion, injecting at every paragraph break inside <think>.

    A hand-rolled loop rather than `model.generate`: the injection has to fire on
    the forward pass whose *input* is the break token, and a `LogitsProcessor`
    only ever sees a step's input token after that step's forward has already
    run. The lag is exactly one step and it is the step that matters.

    `logits_to_keep=1` is load-bearing — the full causal-LM head over a
    [B, S, 151936] prefill is several hundred MB of logits nobody reads.
    """
    device = model.device
    enc = tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").to(device)
    input_ids, attn = enc["input_ids"], enc["attention_mask"]
    batch = input_ids.shape[0]

    position_ids = attn.long().cumsum(-1) - 1
    position_ids.masked_fill_(attn == 0, 1)

    state["dirs"] = dirs
    state["alpha"] = alpha
    state["mask"] = None

    past: Any = None
    cur, cur_pos = input_ids, position_ids
    finished = torch.zeros(batch, dtype=torch.bool, device=device)
    in_think = torch.ones(batch, dtype=torch.bool, device=device)
    inject = torch.zeros(batch, dtype=torch.bool, device=device)
    n_inject = torch.zeros(batch, dtype=torch.long, device=device)
    out: list[list[int]] = [[] for _ in range(batch)]
    think_len = [-1] * batch

    for step in range(max_new):
        state["mask"] = inject if step else None
        res = model(
            input_ids=cur,
            attention_mask=attn,
            position_ids=cur_pos,
            past_key_values=past,
            use_cache=True,
            logits_to_keep=1,
        )
        past = res.past_key_values
        nxt = sample_next(res.logits[:, -1], sampling.temperature, sampling.top_p, sampling.top_k)

        alive = ~finished
        for i in range(batch):
            if alive[i]:
                out[i].append(int(nxt[i]))
        closing = alive & (nxt == think_close_id)
        for i in range(batch):
            if closing[i] and think_len[i] < 0:
                think_len[i] = len(out[i])

        inject = alive & in_think & break_mask[nxt]
        n_inject += inject.long()
        in_think &= ~closing
        for tok_id in eos_ids:
            finished |= nxt == tok_id
        if bool(finished.all()):
            break

        cur = nxt[:, None]
        cur_pos = cur_pos[:, -1:] + 1
        attn = torch.cat([attn, torch.ones(batch, 1, dtype=attn.dtype, device=device)], dim=1)

    texts = [tokenizer.decode(o, skip_special_tokens=False) for o in out]
    hit_cap = [not bool(finished[i]) for i in range(batch)]
    return texts, [len(o) for o in out], [int(n) for n in n_inject], hit_cap, think_len


# --------------------------------------------------------------------------- #
# trace-level doubt accounting
# --------------------------------------------------------------------------- #
def think_block(text: str) -> str:
    body = text.split("</think>")[0]
    return body.split("<think>")[-1]


def doubt_stats(text: str) -> tuple[int, int, int]:
    """(doubt markers in <think>, paragraph breaks, breaks followed by a marker).

    The third is Finding 1's criterion applied to a whole generated trace: the
    next block's FIRST SENTENCE must open with a marker.
    """
    body = think_block(text)
    n_markers = sum(len(marker_matches(body, m, case_sensitive=False)) for m in DOUBT_MARKERS)
    blocks = [b for b in body.split("\n\n")]
    n_doubt_blocks = 0
    import re as _re

    for b in blocks[1:]:
        first = _re.split(r"(?<=[.!?])\s", b.strip(), maxsplit=1)[0]
        if any(marker_matches(first, m, case_sensitive=False) for m in DOUBT_MARKERS):
            n_doubt_blocks += 1
    return n_markers, max(len(blocks) - 1, 0), n_doubt_blocks


def mcnemar(a: list[bool], b: list[bool]) -> tuple[int, int, float, float]:
    bb = sum(1 for x, y in zip(a, b, strict=True) if x and not y)
    cc = sum(1 for x, y in zip(a, b, strict=True) if y and not x)
    chi = (abs(bb - cc) - 1) ** 2 / (bb + cc) if bb + cc else 0.0
    p = math.erfc(math.sqrt(chi / 2)) if chi > 0 else 1.0
    return bb, cc, chi, p


def main() -> None:
    args = parse_args()
    cfg = NLAConfig()
    sampling = SamplingDefaults()

    if args.direction.exists():
        blob = torch.load(args.direction)
        unit, stats = blob["unit"], blob["stats"]
        print(f"loaded direction from {args.direction}")
    else:
        unit, stats = build_direction(args)
        args.direction.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"unit": unit, "stats": stats, "edit": args.edit}, args.direction)
    # stats may carry non-numeric provenance (e.g. which depths went into the
    # pool), so format defensively rather than assuming every value is a float.
    print(
        f"direction ({args.edit}): "
        + "  ".join(
            f"{k}={v:.4g}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in stats.items()
        )
    )

    traces = [json.loads(line) for line in args.traces.open()]
    traces = [t for t in traces if len(t["question"]) <= args.max_question_chars]
    rng = random.Random(args.seed)
    rng.shuffle(traces)
    traces = traces[: args.limit]
    # Batch runtime is the LONGEST member's, so group similar lengths together.
    # The original rollout's length is only a proxy for the regenerated one, but
    # it is a good one and it costs nothing.
    traces.sort(key=lambda t: len(t["response"]))
    print(f"{len(traces)} traces, batched by length")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    trunk = inner_transformer(model)

    # Sized to the LOGIT dimension, not len(tokenizer): Qwen3 pads its embedding
    # matrix past the last real token, and those ids are reachable by sampling.
    n_logits = int(model.config.vocab_size)
    pieces = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
    break_mask = torch.zeros(n_logits, dtype=torch.bool, device=model.device)
    for i, piece in enumerate(pieces):
        if piece and piece.count(NEWLINE_CHAR) >= 2:
            break_mask[i] = True
    print(f"{int(break_mask.sum())} tokens carry a paragraph break")
    think_close_id = int(tokenizer.convert_tokens_to_ids("</think>"))
    eos_ids = {int(tokenizer.eos_token_id), int(tokenizer.convert_tokens_to_ids("<|endoftext|>"))}

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

    unit = unit.to(model.device, torch.float32)

    # --- crash-safe, resumable output ----------------------------------------
    # Rows are appended to <out>.partial the moment each batch finishes, and the
    # file is renamed to <out> only on success. A killed run therefore loses at
    # most one batch, and re-running the same command picks up where it stopped.
    # run_tier2.sh's "skip this arm if its CSV exists" test stays correct because
    # the final name never appears until the arm is complete.
    #
    # The resume is exact, not approximate: the trace order is a seeded shuffle
    # followed by a length sort, and each batch reseeds with args.seed + start, so
    # the batches that still have to run sample exactly as they would have in an
    # uninterrupted run.
    partial = Path(str(args.out) + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    str_cols = {"question_id", "dataset", "gold"}
    rows: list[dict[str, Any]] = []
    if partial.exists():
        rows = [
            {k: v if k in str_cols or k.endswith("_status") else int(v) for k, v in r.items()}
            for r in csv.DictReader(partial.open())
        ]
        print(f"resuming from {partial}: {len(rows)} traces already done", flush=True)
    done_ids = {r["question_id"] for r in rows}
    dump = args.dump.open("a" if rows else "w")
    pf = partial.open("a" if rows else "w", newline="")
    pw = csv.DictWriter(pf, fieldnames=list(rows[0])) if rows else None
    t0 = time.time()

    for start in range(0, len(traces), args.batch):
        chunk = traces[start : start + args.batch]
        if done_ids and all(t["question_id"] in done_ids for t in chunk):
            continue
        n_before = len(rows)
        prompts = [_chat_header(tokenizer, t["question"]) for t in chunk]
        rand_dirs = torch.randn(len(chunk), unit.shape[0], device=model.device)
        rand_dirs = rand_dirs / rand_dirs.norm(dim=-1, keepdim=True)
        per: dict[str, Any] = {}
        for cond in args.conditions:
            dirs = (
                None
                if cond == "none"
                else (unit[None].expand(len(chunk), -1) if cond == "suppress" else rand_dirs)
            )
            torch.manual_seed(args.seed + start)
            texts, lens, n_inj, capped, think_len = generate_batch(
                model,
                tokenizer,
                state,
                prompts,
                dirs,
                args.alpha,
                args.max_new_tokens,
                break_mask,
                think_close_id,
                eos_ids,
                sampling,
            )
            per[cond] = (texts, lens, n_inj, capped, think_len)

        for i, tr in enumerate(chunk):
            row: dict[str, Any] = {
                "question_id": tr["question_id"],
                "dataset": tr["dataset"],
                "gold": tr["gold"],
            }
            rec: dict[str, Any] = {"question_id": tr["question_id"], "gold": tr["gold"]}
            for cond in args.conditions:
                texts, lens, n_inj, capped, think_len = per[cond]
                g = grade(texts[i], tr["gold"])
                n_mark, n_blocks, n_doubt_blocks = doubt_stats(texts[i])
                row.update(
                    {
                        f"{cond}_correct": int(g.is_correct),
                        f"{cond}_has_answer": int(g.has_answer),
                        f"{cond}_status": g.status,
                        f"{cond}_tokens": lens[i],
                        f"{cond}_think_tokens": think_len[i],
                        f"{cond}_capped": int(capped[i]),
                        f"{cond}_injections": n_inj[i],
                        f"{cond}_markers": n_mark,
                        f"{cond}_blocks": n_blocks,
                        f"{cond}_doubt_blocks": n_doubt_blocks,
                    }
                )
                rec[cond] = texts[i]
            rows.append(row)
            dump.write(json.dumps(rec) + "\n")
        dump.flush()
        if pw is None:
            pw = csv.DictWriter(pf, fieldnames=list(rows[0]))
            pw.writeheader()
        pw.writerows(rows[n_before:])
        pf.flush()
        os.fsync(pf.fileno())
        done = len(rows)
        rate = (time.time() - t0) / done
        print(
            f"[{done}/{len(traces)}] {rate:.1f}s/trace  "
            + "  ".join(
                f"{c} {sum(r[f'{c}_correct'] for r in rows)}/{done}" for c in args.conditions
            ),
            flush=True,
        )

    dump.close()
    pf.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    os.replace(partial, args.out)

    n = len(rows)
    print(f"\nwrote {args.out} ({n} traces)  alpha = {args.alpha}\n")
    head = f"{'condition':<10}{'correct':>10}{'answered':>11}{'capped':>8}"
    head += (
        f"{'inj/trace':>11}{'markers':>9}{'doubt blk':>11}"
        f"{'mean think':>11}{'med think':>10}"
    )
    print(head)
    for c in args.conditions:
        corr = sum(r[f"{c}_correct"] for r in rows)
        ans = sum(r[f"{c}_has_answer"] for r in rows)
        cap = sum(r[f"{c}_capped"] for r in rows)
        inj = sum(r[f"{c}_injections"] for r in rows) / n
        mk = sum(r[f"{c}_markers"] for r in rows) / n
        db = sum(r[f"{c}_doubt_blocks"] for r in rows) / n
        toks = sorted(r[f"{c}_think_tokens"] for r in rows if r[f"{c}_think_tokens"] > 0)
        mean_t = sum(toks) / len(toks) if toks else float("nan")
        med_t = toks[len(toks) // 2] if toks else -1
        print(
            f"{c:<10}{corr}/{n} = {100 * corr / n:4.1f}%{ans:>8}{cap:>8}"
            f"{inj:>11.1f}{mk:>9.1f}{db:>11.1f}"
            f"{mean_t:>11.0f}{med_t:>10}"
        )

    print("\npaired McNemar on correctness:")
    per_b = {c: [bool(r[f"{c}_correct"]) for r in rows] for c in args.conditions}
    for a, b in (("suppress", "none"), ("random", "none"), ("suppress", "random")):
        if a in per_b and b in per_b:
            bb, cc, chi, p = mcnemar(per_b[a], per_b[b])
            print(f"  {a} vs {b:<10} b={bb:<4} c={cc:<4} chi2={chi:6.2f}  p = {p:.3g}")


if __name__ == "__main__":
    main()
