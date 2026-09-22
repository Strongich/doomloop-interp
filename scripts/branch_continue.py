#!/usr/bin/env python3
"""Same prefix, four futures: does the removed reasoning ever earn its tokens?

Finding 9 showed that steering after an answer appears cuts thinking almost in
half with no detectable accuracy change. That is not the same as showing the
removed continuation was worthless, because two effects cancel in an aggregate:
a continuation can rescue a wrong candidate, and it can talk the model out of a
right one. Neither is visible unless the candidate is known BEFORE the
continuation runs.

So `probe_candidates.py` freezes each trace at the paragraph boundary where the
model first has a settled answer, and labels that candidate correct or incorrect
using gold AFTER the fact. This script branches that one frozen prefix into:

    base   ordinary continuation
    N      the NLA-derived direction, injected at every later paragraph break
    D      the diff-of-means direction, same sites, same alpha
    exit   `</think>` immediately -- commit to the candidate and stop

Every arm starts from a byte-identical prefix, so a difference between them is
caused by the intervention and by nothing else. `commit` -- the answer the model
would have given at the freeze -- is already known from the probe and costs
nothing, so the deterministic comparator comes for free in the report.

The readouts that matter are per-stratum: on initially CORRECT candidates,
correct->wrong is the damage the intervention does; on initially INCORRECT ones,
wrong->correct is the repair it prevents. k seeds per prefix estimate how much
of each is just sampling.

All arms use one engine. The default vLLM backend installs a version-checked
worker hook. Use --backend transformers to reproduce the historical engine.
Outputs from different backends must live in separate directories.

    uv run python scripts/branch_continue.py --seeds 4 --arms base N D exit
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from suppress_answer import NEWLINE_CHAR, doubt_stats, sample_next  # noqa: E402

from reasoning_attention.config import MODEL_ID, NLAConfig, SamplingDefaults  # noqa: E402
from reasoning_attention.grading import grade  # noqa: E402
from reasoning_attention.nla.arch import inner_transformer  # noqa: E402
from reasoning_attention.tokenview import _chat_header  # noqa: E402

DIRECTIONS = {
    "N": "data/pool/dir_A_1trace.pt",
    "D": "data/dom/dir_D_diffmeans.pt",
    "P": "data/dom/dir_P_probe.pt",
}
FIELDS = [
    "question_id", "dataset", "gold", "band", "arm", "seed",
    "cand", "cand_correct", "freeze_at",
    "correct", "has_answer", "status", "flip",
    "cont_tokens", "cont_think_tokens", "capped", "closed",
    "boundaries", "injections", "markers", "doubt_blocks", "doubt_rate",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prefixes", type=Path, default=Path("data/prefix/prefixes.jsonl"))
    p.add_argument("--outdir", type=Path, default=None)
    p.add_argument("--backend", choices=["vllm", "transformers"], default="vllm")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    p.add_argument("--max-num-batched-tokens", type=int, default=2048)
    p.add_argument("--checkpoint-every", type=int, default=128,
                   help="vLLM requests per durable checkpoint; scheduling uses --batch")
    p.add_argument("--arms", nargs="+", default=["base", "N", "D", "exit"])
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--limit", type=int, default=0)
    # The decode loop is Python-overhead bound: one step costs ~60 ms at batch 6
    # and barely more at batch 20, so batch size is limited by KV memory alone.
    # Sequences are therefore packed to a token-slot budget instead of a fixed
    # count -- short traces run 20 at a time, long ones 4.
    p.add_argument("--batch", type=int, default=24, help="hard cap on batch size")
    p.add_argument("--kv-budget", type=int, default=80000,
                   help="token slots per batch (batch x estimated sequence length)")
    p.add_argument("--retire-every", type=int, default=64,
                   help="steps between dropping finished sequences from the batch")
    p.add_argument("--max-new-tokens", type=int, default=8192)
    # 512 truncated 13% of forced-exit responses, and a truncated response has
    # no \boxed{}, so the comparator was scoring its own budget rather than the
    # policy. 4096 caps essentially none of them and costs little: the arm skips
    # the whole thinking phase.
    p.add_argument("--exit-tokens", type=int, default=4096)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.outdir is None:
        args.outdir = Path("data/prefix_vllm" if args.backend == "vllm" else "data/prefix")
    if args.backend == "vllm":
        from branch_continue_vllm import run
        run(args)
        return
    if (args.outdir / "run_manifest.json").exists():
        raise ValueError("Refusing Transformers output into a manifested vLLM directory")
    cfg = NLAConfig()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rows_in = [json.loads(x) for x in args.prefixes.open()]
    if args.limit:
        rows_in = rows_in[: args.limit]
    # A batch runs until its slowest member stops, so group traces that are
    # likely to stop together. The baseline trace's own length is the best
    # available predictor of how long a continuation from it will run.
    rows_in.sort(key=lambda r: r["think_tokens"])
    print(f"{len(rows_in)} prefixes x {len(args.arms)} arms x {args.seeds} seeds")

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    trunk = inner_transformer(model)
    units = {}
    for a in args.arms:
        if a in DIRECTIONS:
            u = torch.load(DIRECTIONS[a], map_location="cpu", weights_only=False)["unit"]
            units[a] = u.to(model.device, torch.float32)

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

    args.outdir.mkdir(parents=True, exist_ok=True)
    out = args.outdir / "branches.csv"
    partial = Path(str(out) + ".partial")
    dumpf = args.outdir / "branches_texts.jsonl"
    done: set[tuple[str, str, int]] = set()
    if partial.exists():
        for r in csv.DictReader(partial.open()):
            done.add((r["question_id"], r["arm"], int(r["seed"])))
        print(f"resuming: {len(done)} branches already generated")
    pf = partial.open("a" if done else "w", newline="")
    pw = csv.DictWriter(pf, fieldnames=FIELDS)
    if not done:
        pw.writeheader()
    dump = dumpf.open("a")
    t0, n_done = time.time(), 0

    def generate(chunk: list[dict[str, Any]], arm: str, max_new: int, tag: int) -> Any:
        """Decode one batch from its frozen prefixes. Returns per-row raw results.

        Finished sequences are dropped from the batch every `--retire-every`
        steps. Without that a batch costs `max(length)` steps at full width, and
        continuation lengths here span two orders of magnitude, so most of the
        compute would go on padding out one runaway trace.
        """
        suffix = "\n</think>\n\n" if arm == "exit" else ""
        prompts = [_chat_header(tok, r["question"]) + r["prefix"] + suffix for r in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True,
                  padding_side="left").to(model.device)
        ids, attn = enc["input_ids"], enc["attention_mask"]
        B0 = ids.shape[0]
        pos = attn.long().cumsum(-1) - 1
        pos.masked_fill_(attn == 0, 1)
        torch.manual_seed(tag)

        unit = units.get(arm)
        state["dirs"] = unit[None].expand(B0, -1) if unit is not None else None
        state["alpha"] = args.alpha
        state["mask"] = None
        past: Any = None
        cur, cur_pos = ids, pos
        live = list(range(B0))  # batch slot -> row in `chunk`
        fin = torch.zeros(B0, dtype=torch.bool, device=model.device)
        in_think = torch.full((B0,), arm != "exit", dtype=torch.bool, device=model.device)
        inject = torch.zeros(B0, dtype=torch.bool, device=model.device)
        n_inj = torch.zeros(B0, dtype=torch.long, device=model.device)
        n_brk = torch.zeros(B0, dtype=torch.long, device=model.device)
        out_ids: list[list[int]] = [[] for _ in range(B0)]
        think_len = [-1] * B0
        capped = [1] * B0
        brk_of = [0] * B0
        inj_of = [0] * B0

        def retire(slots: list[int]) -> None:
            for sl in slots:
                r = live[sl]
                capped[r] = int(not bool(fin[sl]))
                brk_of[r] = int(n_brk[sl])
                inj_of[r] = int(n_inj[sl])

        for step in range(max_new):
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
            for sl, r in enumerate(live):
                if alive[sl]:
                    out_ids[r].append(int(nxt[sl]))
            closing = alive & (nxt == think_close)
            for sl, r in enumerate(live):
                if closing[sl] and think_len[r] < 0:
                    think_len[r] = len(out_ids[r])
            sites = alive & in_think & break_mask[nxt]
            n_brk += sites.long()
            inject = sites if unit is not None else sites & False
            n_inj += inject.long()
            in_think &= ~closing
            for e in eos_ids:
                fin |= nxt == e
            if bool(fin.all()):
                retire(list(range(len(live))))
                break
            if (step + 1) % args.retire_every == 0 and bool(fin.any()):
                keep = (~fin).nonzero(as_tuple=True)[0]
                retire(fin.nonzero(as_tuple=True)[0].tolist())
                live = [live[sl] for sl in keep.tolist()]
                past.batch_select_indices(keep)
                attn, cur_pos, nxt = attn[keep], cur_pos[keep], nxt[keep]
                fin, in_think, inject = fin[keep], in_think[keep], inject[keep]
                n_inj, n_brk = n_inj[keep], n_brk[keep]
                if unit is not None:
                    state["dirs"] = unit[None].expand(len(live), -1)
            cur = nxt[:, None]
            cur_pos = cur_pos[:, -1:] + 1
            attn = torch.cat(
                [attn, torch.ones(attn.shape[0], 1, dtype=attn.dtype, device=model.device)], 1)
        else:
            retire(list(range(len(live))))

        return [{
            "text": tok.decode(o, skip_special_tokens=False),
            "tokens": len(o), "think_len": think_len[i],
            "capped": capped[i], "brk": brk_of[i], "inj": inj_of[i],
        } for i, o in enumerate(out_ids)]

    def generate_safe(chunk: list[dict[str, Any]], arm: str, max_new: int, tag: int) -> Any:
        """Same, but a batch that does not fit is split rather than lost.

        Packing estimates continuation length from the baseline trace; a run that
        loops far past its estimate can still overflow the card. Halving and
        retrying costs one wasted batch instead of the whole run.
        """
        try:
            return generate(chunk, arm, max_new, tag)
        except torch.OutOfMemoryError:
            if len(chunk) == 1:
                raise
            torch.cuda.empty_cache()
            half = len(chunk) // 2
            print(f"  OOM at B={len(chunk)}, splitting", flush=True)
            return (generate_safe(chunk[:half], arm, max_new, tag)
                    + generate_safe(chunk[half:], arm, max_new, tag + 1))

    # Seed-major: after seed 0 every arm has a complete k=1 result, so a run
    # stopped early is still a finished (smaller) experiment.
    for seed in range(args.seeds):
        for arm in args.arms:
            todo = [r for r in rows_in if (r["question_id"], arm, seed) not in done]
            if not todo:
                print(f"seed {seed} arm {arm}: already complete")
                continue
            print(f"=== seed {seed} arm {arm}: {len(todo)} prefixes", flush=True)
            max_new = args.exit_tokens if arm == "exit" else args.max_new_tokens

            # Packing estimate. Deliberately not conservative: over-estimating
            # shrinks every batch, while an under-estimate is caught by the OOM
            # split. N roughly halves the continuation, which is the whole point
            # of the arm, so it packs tighter.
            grow = 0.7 if arm == "N" else 1.2

            def est(r: dict[str, Any]) -> int:
                """Rough sequence length this row will occupy, for packing."""
                if arm == "exit":
                    return r["freeze_at"] + args.exit_tokens + 300
                tail = max(r["think_tokens"] - r["freeze_at"], 200)
                return r["freeze_at"] + min(int(grow * tail) + 500, max_new) + 300

            chunks: list[list[dict[str, Any]]] = []
            cur_chunk: list[dict[str, Any]] = []
            cur_est = 0
            for r in todo:  # already sorted by think_tokens, so lengths cluster
                e = max(est(r), cur_est)
                if cur_chunk and ((len(cur_chunk) + 1) * e > args.kv_budget
                                  or len(cur_chunk) >= args.batch):
                    chunks.append(cur_chunk)
                    cur_chunk, cur_est = [], 0
                cur_chunk.append(r)
                cur_est = max(cur_est, est(r))
            if cur_chunk:
                chunks.append(cur_chunk)

            seen = 0
            for chunk in chunks:
                start = seen
                seen += len(chunk)
                res = generate_safe(chunk, arm, max_new, seed * 100003 + start)
                batch_rows = []

                for i, r in enumerate(chunk):
                    o = res[i]
                    g = grade(o["text"], r["gold"])
                    # doubt_stats reads between <think> and </think>. The
                    # continuation carries no opening tag, so one is prepended to
                    # scope the count to post-freeze text: the prefix's own doubt
                    # is not this arm's doing. `exit` never thinks again, so its
                    # think body is closed immediately and reads zero.
                    head = "<think>\n</think>\n" if arm == "exit" else "<think>\n"
                    nm, _nb, nd = doubt_stats(head + o["text"])
                    brk = o["brk"]
                    cc = int(r["cand_correct"])
                    flip = ("keep" if cc and g.is_correct else
                            "break" if cc and not g.is_correct else
                            "repair" if not cc and g.is_correct else "stuck")
                    batch_rows.append({
                        "question_id": r["question_id"], "dataset": r["dataset"],
                        "gold": str(r["gold"]), "band": r["band"], "arm": arm, "seed": seed,
                        "cand": r["cand"], "cand_correct": cc, "freeze_at": r["freeze_at"],
                        "correct": int(g.is_correct), "has_answer": int(g.has_answer),
                        "status": g.status, "flip": flip,
                        "cont_tokens": o["tokens"],
                        "cont_think_tokens": o["think_len"],
                        "capped": o["capped"],
                        "closed": int(o["think_len"] >= 0),
                        "boundaries": brk, "injections": o["inj"],
                        "markers": nm, "doubt_blocks": nd,
                        "doubt_rate": round(nd / brk, 4) if brk else -1.0,
                    })
                    dump.write(json.dumps({
                        "question_id": r["question_id"], "arm": arm, "seed": seed,
                        "gold": r["gold"], "band": r["band"], "cand": r["cand"],
                        "text": o["text"]}) + "\n")
                pw.writerows(batch_rows)
                pf.flush()
                os.fsync(pf.fileno())
                dump.flush()
                n_done += len(chunk)
                el = time.time() - t0
                print(f"  [{seen}/{len(todo)}] B={len(chunk)} {el / n_done:.1f}s/branch "
                      f"median_cont={sorted(x['cont_tokens'] for x in batch_rows)[len(chunk)//2]}",
                      flush=True)

    dump.close()
    pf.close()
    os.replace(partial, out)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
