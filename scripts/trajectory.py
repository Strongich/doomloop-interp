#!/usr/bin/env python3
"""Does a doom loop stop moving in representation space?

The doubt marker turned out to carry no signal about whether a rollout recovers
(matched on the question, the AV cannot tell them apart). So the phenomenon is
probably not *at* a token but in the dynamics: "long and useless" would mean the
residual stream re-enters states it has already visited, computing nothing new
while the decoder keeps emitting text.

This measures that directly, with no AV involved — so it is unaffected by the
~37% of `h_l` the reconstructor cannot capture:

  velocity   cos(h_t, h_{t-1})              — how far the state moves per token
  recurrence max_{s <= t-gap} cos(h_t, h_s)  — is this state a state we were in?
  eff_rank   participation ratio of a window — has the trajectory collapsed onto
                                               a low-dimensional cycle?

`h_l` at every position comes from ONE forward pass per trace: the layer-20 hook
sees the whole [1, S, d] tensor, so a trajectory costs the same as the single
vector we were extracting before.

    uv run python scripts/trajectory.py --out data/trajectories

Compares, per looping trace, a window before its onset against a window after,
and against a non-looping rollout of the SAME question at the same relative
depth — holding the problem fixed, as the paired CSVs do.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from reasoning_attention.config import MODEL_ID, NLAConfig  # noqa: E402
from reasoning_attention.data.math_datasets import build_messages  # noqa: E402
from reasoning_attention.loops import (  # noqa: E402
    find_inner_repetition,
    onset_token_index,
    tokenize_with_spans,
)
from reasoning_attention.nla.arch import inner_transformer  # noqa: E402

# Positions closer together than this are excluded from the recurrence max: any
# two adjacent residual streams are similar, and counting that as "revisiting"
# would report every trace as a loop.
DEFAULT_MIN_GAP = 32
# Cap on positions kept per trace. A 32k-token trace strided to 8k still resolves
# loops with periods of tens of tokens, and the recurrence matmul is O(S^2).
MAX_POSITIONS = 8192
WINDOW = 384


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, default=Path("data/traces"))
    p.add_argument("--datasets", nargs="+", default=["aime2025", "amc23", "gsm8k"])
    p.add_argument("--out", type=Path, default=Path("data/trajectories"))
    p.add_argument("--base", default=MODEL_ID)
    p.add_argument("--min-gap", type=int, default=DEFAULT_MIN_GAP)
    p.add_argument("--max-positions", type=int, default=MAX_POSITIONS)
    p.add_argument("--window", type=int, default=WINDOW)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument(
        "--no-controls",
        action="store_true",
        help="skip the matched non-looping rollouts (loopers only)",
    )
    return p.parse_args()


# --------------------------------------------------------------------------- #
# selecting traces
# --------------------------------------------------------------------------- #


def collect(args: argparse.Namespace, tokenizer: Any) -> list[dict[str, Any]]:
    """Looping traces plus, for each, a non-looping rollout of the same question."""
    by_question: dict[str, list[dict[str, Any]]] = defaultdict(list)
    loopers: list[dict[str, Any]] = []

    for name in args.datasets:
        path = args.traces / f"{name}.jsonl"
        if not path.exists():
            continue
        with path.open() as fh:
            for line in fh:
                row = json.loads(line)
                hit = find_inner_repetition(row["response"])
                by_question[row["question_id"]].append({"row": row, "looped": hit is not None})
                if hit is None:
                    continue
                spans = tokenize_with_spans(tokenizer, row["response"])
                onset = onset_token_index(spans, hit)
                if onset is None:
                    continue
                loopers.append(
                    {
                        "row": row,
                        "kind": "loop",
                        "onset_token_index": onset,
                        "onset_char": spans.offsets[onset][0],
                        # This trace's own token count. Needed for the control's
                        # relative depth — reading it off the loop variable
                        # `spans` later would use whichever trace happened to be
                        # tokenized last, which silently produced fractions > 1.
                        "n_response_tokens": len(spans.ids),
                        "period": hit.period,
                        "repeats": hit.repeats,
                    }
                )

    if args.limit:
        loopers = loopers[: args.limit]
    if args.no_controls:
        return loopers

    out = list(loopers)
    for item in loopers:
        qid = item["row"]["question_id"]
        # Prefer a control that also answered correctly: the contrast we want is
        # "same problem, one rollout looped and one solved it".
        pool = [c for c in by_question[qid] if not c["looped"]]
        pool.sort(key=lambda c: (not c["row"]["is_correct"],))
        if not pool:
            continue
        out.append(
            {
                "row": pool[0]["row"],
                "kind": "control",
                "pair_question_id": qid,
                # No onset of its own; probe the same relative depth so the
                # comparison is at a matched stage of the trace, not a matched
                # token offset (the loop trace is far longer).
                "onset_frac": item["onset_token_index"] / max(item["n_response_tokens"], 1),
            }
        )
    return out


# --------------------------------------------------------------------------- #
# trajectory extraction + metrics
# --------------------------------------------------------------------------- #


class Extractor:
    def __init__(self, base: str, layer: int) -> None:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(base)
        model = AutoModelForCausalLM.from_pretrained(base, dtype=torch.bfloat16, device_map="cuda")
        model.eval()
        self.trunk = inner_transformer(model)
        self.captured: dict[str, torch.Tensor] = {}
        self.trunk.layers[layer].register_forward_hook(self._hook)

    def _hook(self, _m: Any, _i: Any, output: Any) -> None:
        hidden = output[0] if isinstance(output, tuple) else output
        # Keep every position, not just the last: this IS the trajectory.
        self.captured["h"] = hidden[0].detach()

    @torch.no_grad()
    def run(self, question: str, response: str) -> tuple[torch.Tensor, int]:
        """Returns [S, d] layer-l states over the response, and the header length."""
        header = self.tokenizer.apply_chat_template(
            build_messages(question),
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        n_header = len(self.tokenizer(header, add_special_tokens=False)["input_ids"])
        enc = self.tokenizer(header + response, return_tensors="pt").to("cuda")
        self.trunk(**enc)
        return self.captured["h"], n_header


def metrics(states: torch.Tensor, *, min_gap: int, window: int) -> dict[str, np.ndarray]:
    """Velocity and recurrence per position, from unit-normalized states."""
    unit = torch.nn.functional.normalize(states.float(), dim=-1)
    n = unit.shape[0]

    velocity = torch.ones(n, device=unit.device)
    if n > 1:
        velocity[1:] = (unit[1:] * unit[:-1]).sum(-1)

    # Blocked so a 8k x 8k similarity matrix is never materialized at once.
    recurrence = torch.full((n,), float("nan"), device=unit.device)
    argrecur = torch.zeros(n, dtype=torch.long, device=unit.device)
    block = 512
    for start in range(0, n, block):
        stop = min(start + block, n)
        rows = unit[start:stop]
        sims = rows @ unit.T  # [b, n]
        idx = torch.arange(start, stop, device=unit.device)[:, None]
        cols = torch.arange(n, device=unit.device)[None, :]
        sims = sims.masked_fill(cols > idx - min_gap, float("-inf"))
        best, where = sims.max(dim=1)
        recurrence[start:stop] = best
        argrecur[start:stop] = where

    out = {
        "velocity": velocity.cpu().numpy(),
        "recurrence": recurrence.cpu().numpy(),
        "recurrence_at": argrecur.cpu().numpy(),
        "eff_rank": effective_rank(unit, window).cpu().numpy(),
    }
    return out


def effective_rank(unit: torch.Tensor, window: int) -> torch.Tensor:
    """Participation ratio of each window's singular spectrum.

    (sum s^2)^2 / sum s^4 — how many directions the trajectory actually uses in
    that stretch. A trajectory cycling through k states has effective rank ~k, so
    this is the "collapsed onto a cycle" measure. Computed on non-overlapping
    windows and broadcast back, which is enough to see a collapse.
    """
    n = unit.shape[0]
    out = torch.full((n,), float("nan"), device=unit.device)
    for start in range(0, n, window):
        stop = min(start + window, n)
        chunk = unit[start:stop]
        if chunk.shape[0] < 8:
            continue
        centered = chunk - chunk.mean(0, keepdim=True)
        sv = torch.linalg.svdvals(centered.float())
        p2 = (sv**2).sum()
        p4 = (sv**4).sum()
        out[start:stop] = (p2 * p2 / p4) if p4 > 0 else float("nan")
    return out


def summarize(m: dict[str, np.ndarray], onset: int, window: int) -> dict[str, float]:
    """Windowed means before and after the onset."""

    def win(arr: np.ndarray, lo: int, hi: int) -> float:
        lo, hi = max(0, lo), min(len(arr), hi)
        seg = arr[lo:hi]
        seg = seg[np.isfinite(seg)]
        return float(seg.mean()) if seg.size else float("nan")

    return {
        "velocity_pre": win(m["velocity"], onset - window, onset),
        "velocity_post": win(m["velocity"], onset, onset + window),
        "recurrence_pre": win(m["recurrence"], onset - window, onset),
        "recurrence_post": win(m["recurrence"], onset, onset + window),
        "eff_rank_pre": win(m["eff_rank"], onset - window, onset),
        "eff_rank_post": win(m["eff_rank"], onset, onset + window),
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = NLAConfig()

    from transformers import AutoTokenizer

    print("== selecting traces ==")
    items = collect(args, AutoTokenizer.from_pretrained(args.base))
    n_loop = sum(i["kind"] == "loop" for i in items)
    print(f"{n_loop} looping traces, {len(items) - n_loop} matched controls")

    ex = Extractor(args.base, cfg.extraction_layer)
    rows: list[dict[str, Any]] = []
    for i, item in enumerate(items, 1):
        row = item["row"]
        states, n_header = ex.run(row["question"], row["response"])
        states = states[n_header:]  # response positions only
        n_full = states.shape[0]
        stride = max(1, -(-n_full // args.max_positions))
        states = states[::stride]

        if item["kind"] == "loop":
            onset = item["onset_token_index"] // stride
        else:
            onset = int(item["onset_frac"] * states.shape[0])
        onset = max(0, min(onset, states.shape[0] - 1))

        m = metrics(states, min_gap=max(1, args.min_gap // stride), window=args.window)
        summary = summarize(m, onset, args.window)
        rec = {
            "question_id": row["question_id"],
            "rollout_index": row["rollout_index"],
            "dataset": row["dataset"],
            "kind": item["kind"],
            "outcome": row["outcome"],
            "is_correct": row["is_correct"],
            "n_tokens": n_full,
            "stride": stride,
            "onset": onset,
            "period_chars": item.get("period", ""),
            "repeats": item.get("repeats", ""),
            **summary,
        }
        rows.append(rec)
        np.savez_compressed(
            args.out / f"{row['question_id'].replace(':', '_')}_{row['rollout_index']}.npz",
            onset=onset,
            stride=stride,
            **m,
        )
        vel = f"{summary['velocity_pre']:.3f}->{summary['velocity_post']:.3f}"
        rec_s = f"{summary['recurrence_pre']:.3f}->{summary['recurrence_post']:.3f}"
        rank = f"{summary['eff_rank_pre']:.0f}->{summary['eff_rank_post']:.0f}"
        print(
            f"  [{i}/{len(items)}] {rec['kind']:<7} "
            f"{row['question_id']}#{row['rollout_index']} n={n_full} s={stride}"
            f"  vel {vel}  rec {rec_s}  rank {rank}"
        )
        del states
        torch.cuda.empty_cache()

    import csv

    path = args.out / "summary.csv"
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {path} ({len(rows)} rows) and {len(rows)} .npz trajectories")


if __name__ == "__main__":
    main()
