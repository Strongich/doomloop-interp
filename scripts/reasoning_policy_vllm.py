#!/usr/bin/env python3
r"""Whole-question generation under a token-delayed boundary-steering policy.

This is NOT `branch_continue.py`. That runner resumes from a frozen prefix chosen
by a candidate-answer probe, so its "start" is a detector output. Here the model
answers the original question from scratch and the intervention is withheld for a
fixed number of *generated thinking tokens*. No probe, no gold, no future text
enters the decision -- the policy is deployable as written.

    policy = (direction, alpha, start_delay)
    h' = h + alpha * ||h|| * unit          at every generated paragraph boundary
                                            inside <think> with g >= start_delay

`g` is the one-based generated-token index -- position minus prompt length plus
one -- so delay 0 and delay 1 are the same policy. See
`vllm_steering.boundary_positions`.

**The `<think>` opener is generated, not prompted.** Qwen3-1.7B's chat template
with `enable_thinking=True` ends the prompt at `<|im_start|>assistant\n`; the
model emits `<think>` itself, at g=1. The plan document assumed the opener sat in
the prompt, and it does not. The consequence is a fixed one-token offset between
`g` and "thinking tokens produced", which is immaterial against delays of 256 and
up and exactly zero at delay 0. It is left in rather than corrected away because
`g` is then a property of the request that the worker can evaluate without
parsing for a tag. `<think>` is followed by a single newline, never the double
newline a boundary token carries, so the opener is never itself a site.

Arms are named policies rather than a fixed list, because the sweep needs the
same code to run a bare baseline, an NLA direction, a diff-of-means direction,
three isotropic random controls, and two brevity prompts. A random control is a
*direction* control at the selected operating point; a brevity prompt is a
*prompt* control with no steering at all. Both belong here so that engine,
decoding and grading are identical across every comparator.

    uv run python scripts/reasoning_policy_vllm.py \
        --cohort data/policy/dev400.jsonl --outdir data/reasoning_policy_v1/smoke \
        --policies base N@a1.0d512 --seeds 1 --limit 16
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from branch_continue_vllm import digest, read_journal, request_seed  # noqa: E402

# Named directions. `base` and the brevity prompts steer nothing.
DIRECTIONS: dict[str, str] = {
    "N": "data/pool/dir_A_1trace.pt",
    "D": "data/dom/dir_D_diffmeans.pt",
    "P": "data/dom/dir_P_probe.pt",
    "R101": "data/dirs_random/dir_R101.pt",
    "R202": "data/dirs_random/dir_R202.pt",
    "R303": "data/dirs_random/dir_R303.pt",
}

# EVERY arm carries this, exactly as `scripts/prompt_baseline.py` does (Finding 7).
# Grading reads `\boxed{}` only, so an output that answers in prose -- "**Answer:**
# 26" -- scores as no answer at all. Finding 7 measured the consequence directly:
# without it, 10 of 16 prompted rollouts were graded wrong while carrying the
# correct value, and the experiment would have reported a formatting change as a
# 60-point accuracy collapse. Omitting it here reproduced that artifact -- brevityA
# lost 5.6 points of which 5.1 were unboxed correct answers.
#
# It goes on the baseline too. A format instruction present in one arm and absent
# from another measures the instruction, not the intervention.
FORMAT = "Give your final answer in \\boxed{}."

# Declared brevity instructions, tuned on development only. They condition the
# prompt and are practical baselines, not identical-prefix causal controls.
BREVITY: dict[str, str] = {
    "A": "Solve concisely. Avoid repeating calculations or explanations.",
    "B": (
        "Give the necessary reasoning, check the result briefly, "
        "and provide the final answer."
    ),
}

FIELDS = [
    "question_id", "dataset", "gold", "policy", "direction", "alpha", "delay",
    "brevity", "seed", "correct", "has_answer", "status", "total_tokens",
    "think_tokens", "answer_tokens", "capped", "closed", "boundaries",
    "eligible_sites", "injections", "first_injection_g", "markers",
    "doubt_blocks", "doubt_rate",
]

POLICY_RE = re.compile(
    r"^(?P<dir>[A-Za-z0-9]+)@a(?P<alpha>[0-9.]+)d(?P<delay>\d+)(?:g(?P<guard>\d+))?$"
)
REPEAT_MINLEN = 8  # tokens; a paragraph shorter than this never counts toward the guard


def parse_policy(spec: str) -> dict[str, Any]:
    """`base` | `brevityA` | `<DIR>@a<alpha>d<delay>`.

    The name is the identity of the arm and is stored with every row, so a
    results file can always be traced back to the exact policy that produced it
    without consulting the manifest.
    """
    if spec == "base":
        return {"name": spec, "direction": None, "alpha": 0.0, "delay": 0, "brevity": None,
                "guard": 0}
    if spec.startswith("brevity"):
        key = spec.removeprefix("brevity")
        if key not in BREVITY:
            raise ValueError(f"Unknown brevity prompt {key!r}; have {sorted(BREVITY)}")
        return {"name": spec, "direction": None, "alpha": 0.0, "delay": 0, "brevity": key,
                "guard": 0}
    m = POLICY_RE.match(spec)
    if not m:
        raise ValueError(f"Bad policy {spec!r}; use base, brevity<A|B>, or DIR@a<alpha>d<delay>")
    name = m.group("dir")
    if name not in DIRECTIONS:
        raise ValueError(f"Unknown direction {name!r}; have {sorted(DIRECTIONS)}")
    return {
        "name": spec,
        "direction": name,
        "alpha": float(m.group("alpha")),
        "delay": int(m.group("delay")),
        "brevity": None,
        # g<K>: stop steering once any paragraph occurs for the K-th time (loop guard).
        "guard": int(m.group("guard") or 0),
    }


def expected_sites(
    ids: list[int], prompt_len: int, boundary_set: set[int], close_id: int, delay: int,
    guard: int = 0,
) -> list[int]:
    """Reconstruct eligible sites from output IDs alone, to audit the worker.

    Deliberately written from the output tokens rather than by calling the
    worker's own selector: an audit that reuses the code under test cannot fail.
    The final sampled token is excluded because a finished request never consumes
    it to predict a successor.
    """
    close = ids.index(close_id) if close_id in ids else len(ids)
    cut = len(ids)
    if guard:
        seen: dict[tuple[int, ...], int] = {}
        start = 0
        for i, token in enumerate(ids[:-1]):
            if i >= close:
                break
            if token in boundary_set:
                para = tuple(ids[start : i + 1])
                if len(para) >= REPEAT_MINLEN:
                    seen[para] = seen.get(para, 0) + 1
                    if seen[para] >= guard:
                        cut = i
                        break
                start = i + 1
    return [
        prompt_len + i
        for i, token in enumerate(ids[:-1])
        if i < close and i < cut and token in boundary_set and (i + 1) >= delay
    ]


def build_random_directions(outdir: Path, d_model: int, seeds: tuple[int, ...]) -> None:
    """Isotropic unit directions with fixed seeds, written once and then hashed.

    Saved to disk rather than regenerated per run so the manifest can record a
    content hash: "seed 101" is only reproducible if the torch RNG, dtype and
    device agree, and a hash does not care.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        path = outdir / f"dir_R{seed}.pt"
        if path.exists():
            continue
        gen = torch.Generator().manual_seed(seed)
        v = torch.randn(d_model, generator=gen, dtype=torch.float32)
        torch.save({"unit": v / v.norm(), "seed": seed, "kind": "isotropic_random"}, path)
        print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True,
                    help="jsonl with question_id, question, gold, dataset")
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--policies", nargs="+", required=True)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--seed-start", type=int, default=0,
                    help="first seed index; seeds run seed_start..seeds-1 (top-ups)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-new-tokens", type=int, default=16384)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--checkpoint-every", type=int, default=128)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    # Cross-model transfer (EXPERIMENT-transfer-8b-log.md): a different target model,
    # its steering layer, and extra NAME=PATH directions in that model's residual space.
    ap.add_argument("--model", default=None, help="HF id; default config.MODEL_ID (1.7B)")
    ap.add_argument("--layer", type=int, default=None, help="decoder layer; default l=20")
    ap.add_argument("--dir", action="append", default=[], metavar="NAME=PATH")
    ap.add_argument("--pp", type=int, default=1, help="pipeline-parallel size (one engine)")
    ap.add_argument("--tp", type=int, default=1, help="tensor-parallel size (one engine)")
    ap.add_argument("--stream", action="store_true",
                    help="one engine queue for all seeds/policies (per-request steering); "
                         "--checkpoint-every then only sets the export/log cadence")
    args = ap.parse_args()
    for spec in args.dir:
        name, _, path = spec.partition("=")
        if not name.isalnum() or not Path(path).exists():
            raise ValueError(f"Bad --dir {spec!r}")
        DIRECTIONS[name] = path

    from transformers import AutoTokenizer
    from vllm import SamplingParams

    from reasoning_attention.config import MODEL_ID, NLAConfig, SamplingDefaults
    from reasoning_attention.data.math_datasets import build_messages
    from reasoning_attention.grading import grade
    from reasoning_attention.serving.vllm_steering import (
        build_steering_llm, pop_steering_stats, steering_stats,
    )
    from suppress_answer import NEWLINE_CHAR, doubt_stats

    if args.batch < 1 or args.checkpoint_every < 1 or args.seeds < 1:
        raise ValueError("batch, checkpoint-every and seeds must be positive")
    policies = [parse_policy(p) for p in args.policies]
    if len({p["name"] for p in policies}) != len(policies):
        raise ValueError("Duplicate policy names")

    model_id = args.model or MODEL_ID
    layer = NLAConfig().extraction_layer if args.layer is None else args.layer
    tok = AutoTokenizer.from_pretrained(model_id)
    if model_id == MODEL_ID:
        build_random_directions(Path("data/dirs_random"), 2048, (101, 202, 303))

    rows = [json.loads(x) for x in args.cohort.read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Need a nonempty cohort with unique question IDs")

    sampling = SamplingDefaults()
    boundaries = [
        i for i in range(len(tok)) if tok.convert_ids_to_tokens(i).count(NEWLINE_CHAR) >= 2
    ]
    boundary_set = set(boundaries)
    close_id = int(tok.convert_tokens_to_ids("</think>"))
    stop_ids = sorted({int(tok.eos_token_id), int(tok.convert_tokens_to_ids("<|endoftext|>"))})

    used = sorted({p["direction"] for p in policies if p["direction"]})
    units = {
        name: torch.load(DIRECTIONS[name], map_location="cpu", weights_only=False)["unit"]
        .float()
        .tolist()
        for name in used
    }

    def prompt_ids(question: str, brevity: str | None) -> list[int]:
        """Canonical thinking template, with the shared format instruction.

        The user turn -- the task itself -- is byte-identical in every arm. The
        system turn carries FORMAT everywhere, plus the brevity text where that
        arm declares one, so the only difference between arms is the brevity
        instruction itself.
        """
        system = f"{BREVITY[brevity]} {FORMAT}" if brevity else FORMAT
        messages = [{"role": "system", "content": system}, *build_messages(question)]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
        )
        return list(tok(text, add_special_tokens=False)["input_ids"])

    prompts = {
        (r["question_id"], p["name"]): prompt_ids(r["question"], p["brevity"])
        for r in rows
        for p in policies
    }
    max_len = max(len(ids) for ids in prompts.values()) + args.max_new_tokens

    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "runner": "reasoning_policy_vllm", "protocol": "whole-question-delayed-boundary-v1",
        "model": model_id, "model_commit": tok.init_kwargs.get("_commit_hash"),
        "versions": {p: version(p) for p in ("vllm", "torch", "transformers")},
        "cohort_sha256": digest(args.cohort), "limit": args.limit,
        "question_ids": [r["question_id"] for r in rows],
        "policies": policies,
        "directions": {n: digest(Path(DIRECTIONS[n])) for n in used},
        "format_instruction": FORMAT,
        "brevity_prompts": {k: v for k, v in BREVITY.items()
                            if k in {p["brevity"] for p in policies}},
        "code": {p: digest(Path(p)) for p in (
            "scripts/reasoning_policy_vllm.py",
            "src/reasoning_attention/serving/vllm_steering.py",
            "src/reasoning_attention/grading.py")},
        "seeds": args.seeds, "seed_start": args.seed_start, "max_new_tokens": args.max_new_tokens,
        "temperature": sampling.temperature, "top_p": sampling.top_p, "top_k": sampling.top_k,
        "layer": layer, "max_model_len": max_len,
        "max_num_seqs": args.batch, "checkpoint_every": args.checkpoint_every,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True, "async_scheduling": False, "prefix_caching": False,
        "pipeline_parallel_size": args.pp, "tensor_parallel_size": args.tp,
        "stream": args.stream,
        "seed_scheme": "sha256(question_id:seed), identical across policies",
        "delay_scheme": "g = p - prompt_len + 1, one-based; prompt <think> not counted",
    }
    manifest_path = args.outdir / "run_manifest.json"
    journal = args.outdir / "rollouts.jsonl"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old != manifest:
            changed = sorted(k for k in set(old) | set(manifest)
                             if old.get(k) != manifest.get(k))
            raise ValueError(f"Manifest differs in {changed}; use a new output directory")
    else:
        if any(args.outdir.glob("rollouts*")):
            raise ValueError("Existing unmanifested outputs; use a fresh directory")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

    records = read_journal(journal)
    done = {(r["question_id"], r["policy"], r["seed"]) for r in records}
    if len(done) != len(records):
        raise ValueError("Duplicate rollout records in journal")

    def export(final: bool = False) -> None:
        csv_path = args.outdir / ("rollouts.csv" if final else "rollouts.csv.partial")
        tmp = csv_path.with_name(csv_path.name + ".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({k: row[k] for k in FIELDS} for row in records)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, csv_path)

    total = len(rows) * len(policies) * (args.seeds - args.seed_start)
    if len(done) == total:
        export(final=True)
        print("All rollouts already complete")
        return
    print(f"{len(rows)} questions x {len(policies)} policies x {args.seeds} seeds "
          f"= {total}; {len(done)} resumed; max context {max_len}", flush=True)

    llm = build_steering_llm(
        model_id, layer, boundaries, close_id, max_len,
        args.batch, args.gpu_memory_utilization, args.max_num_batched_tokens, args.pp,
        args.tp,
    )
    started = time.monotonic()
    completed = 0

    def finish(out: Any, row: dict[str, Any], policy: dict[str, Any], seed: int, result: Any,
               actual: dict[str, Any], elapsed: float, unit: Any) -> None:
        """Audit one finished rollout against an independent site re-derivation, grade it,
        and journal it. Shared by the chunked and the streaming paths."""
        name = policy["name"]
        gen = result.outputs[0]
        ids = list(gen.token_ids)
        prompt_len = len(result.prompt_token_ids)
        close = ids.index(close_id) if close_id in ids else None
        think_end = close if close is not None else len(ids)
        sites = expected_sites(
            ids, prompt_len, boundary_set, close_id, policy["delay"],
            policy["guard"],
        )
        steering = unit is not None and policy["alpha"] != 0
        if actual["sites"] != sites:
            raise RuntimeError(
                f"Site mismatch for {row['question_id']} under {name}: "
                f"worker {actual['sites'][:8]}... expected {sites[:8]}..."
            )
        if actual["injections"] != (sites if steering else []):
            raise RuntimeError(f"Injection mismatch for {row['question_id']}")
        text = tok.decode(ids, skip_special_tokens=False)
        g = grade(text, row["gold"])
        markers, _, doubt_blocks = doubt_stats(text)
        n_bound = sum(t in boundary_set for t in ids[:think_end])
        inj = actual["injections"]
        record = {
            "question_id": row["question_id"],
            "dataset": row.get("dataset", "gsm8k"),
            "gold": str(row["gold"]), "policy": name,
            "direction": policy["direction"] or "",
            "alpha": policy["alpha"], "delay": policy["delay"],
            "brevity": policy["brevity"] or "", "seed": seed,
            "correct": int(g.is_correct), "has_answer": int(g.has_answer),
            "status": g.status, "total_tokens": len(ids),
            "think_tokens": think_end + int(close is not None),
            "answer_tokens": len(ids) - think_end - int(close is not None),
            "capped": int(gen.finish_reason == "length"),
            "closed": int(close is not None), "boundaries": n_bound,
            "eligible_sites": len(sites), "injections": len(inj),
            # Where the policy actually began, in generated tokens.
            "first_injection_g": (inj[0] - prompt_len + 1) if inj else -1,
            "markers": markers, "doubt_blocks": doubt_blocks,
            "doubt_rate": round(doubt_blocks / n_bound, 4) if n_bound else -1.0,
            "text": text, "token_ids": ids,
            "prompt_tokens": prompt_len,
            "guard": policy["guard"],
            # Last eligible site the guard allowed (-1: none / guard off).
            "last_site_g": (sites[-1] - prompt_len + 1) if sites else -1,
            "request_seed": request_seed(row["question_id"], seed),
            "finish_reason": gen.finish_reason,
            "batch_generation_seconds": elapsed,
        }
        out.write(json.dumps(record) + "\n")
        records.append(record)
        done.add((row["question_id"], name, seed))

    def params_for(row: dict[str, Any], seed: int, extra: dict[str, Any]) -> Any:
        return SamplingParams(
            temperature=sampling.temperature, top_p=sampling.top_p,
            top_k=sampling.top_k, max_tokens=args.max_new_tokens,
            seed=request_seed(row["question_id"], seed), stop_token_ids=stop_ids,
            extra_args=extra,
        )

    with journal.open("a") as out:
        if args.stream:
            # Streaming: every (seed, policy, question) goes into ONE engine queue with its
            # own per-request policy (`extra_args["steer"]`), so the batch stays full and
            # the long-rollout tail is paid once per run instead of once per chunk. Each
            # rollout is audited and journaled as soon as it finishes.
            llm.collective_rpc(
                "register_steering_units",
                args=({d: units[d] for d in {p["direction"] for p in policies if p["direction"]}},),
            )
            llm.collective_rpc("configure_reasoning_steering", args=(None, 0.0, True, 0, 0,
                                                                     REPEAT_MINLEN))
            engine = llm.llm_engine
            pending: dict[str, tuple[dict[str, Any], dict[str, Any], int, str]] = {}
            alias: dict[str, str] = {}
            for seed in range(args.seed_start, args.seeds):
                for policy in policies:
                    name = policy["name"]
                    for r in rows:
                        if (r["question_id"], name, seed) in done:
                            continue
                        aid = f"{r['question_id']}|{name}|{seed}"
                        steer = {"dir": policy["direction"], "alpha": policy["alpha"],
                                 "delay": policy["delay"], "guard": policy["guard"]}
                        rid = engine.add_request(
                            aid, {"prompt_token_ids": prompts[(r["question_id"], name)]},
                            params_for(r, seed, {"steering_audit_id": aid, "steer": steer}),
                        )
                        # Outputs may carry the external id or vLLM's internal one.
                        pending[aid] = (r, policy, seed, aid)
                        alias[aid] = alias[rid] = aid
            n_total = len(pending)
            t_last, tok_window = time.monotonic(), 0
            while engine.has_unfinished_requests():
                finished = [o for o in engine.step() if o.finished]
                if not finished:
                    continue
                keys = [alias.get(o.request_id) for o in finished]
                if None in keys:
                    raise RuntimeError("Finished request with an unknown id")
                stats = pop_steering_stats(llm, [pending[k][3] for k in keys])
                if stats["calls"] == 0:
                    raise RuntimeError("Steering hook did not execute")
                for o, k in zip(finished, keys, strict=True):
                    row, policy, seed, aid = pending.pop(k)
                    unit = units.get(policy["direction"]) if policy["direction"] else None
                    finish(out, row, policy, seed, o, stats["requests"][aid],
                           time.monotonic() - started, unit)
                    completed += 1
                    tok_window += len(o.outputs[0].token_ids)
                out.flush()
                os.fsync(out.fileno())
                if completed % args.checkpoint_every < len(finished) or not pending:
                    export()
                    now = time.monotonic()
                    print(f"stream {completed}/{n_total} {tok_window / (now - t_last):.1f} tok/s; "
                          f"{(now - started) / completed:.2f}s/rollout", flush=True)
                    t_last, tok_window = now, 0
            if pending:
                raise RuntimeError(f"{len(pending)} requests never finished")
        else:
            for seed in range(args.seed_start, args.seeds):
                for policy in policies:
                    name = policy["name"]
                    todo = [r for r in rows if (r["question_id"], name, seed) not in done]
                    for offset in range(0, len(todo), args.checkpoint_every):
                        chunk = todo[offset : offset + args.checkpoint_every]
                        unit = units.get(policy["direction"]) if policy["direction"] else None
                        llm.collective_rpc(
                            "configure_reasoning_steering",
                            args=(unit, policy["alpha"], True, policy["delay"],
                                  policy["guard"], REPEAT_MINLEN),
                        )
                        inputs = [
                            {"prompt_token_ids": prompts[(r["question_id"], name)]}
                            for r in chunk
                        ]
                        params = [params_for(r, seed, {"steering_audit_id": r["question_id"]})
                                  for r in chunk]
                        batch_start = time.monotonic()
                        outputs = llm.generate(inputs, params, use_tqdm=False)
                        elapsed = time.monotonic() - batch_start
                        stats = steering_stats(llm)
                        if stats["calls"] == 0:
                            raise RuntimeError("Steering hook did not execute")
                        for row, result in zip(chunk, outputs, strict=True):
                            finish(out, row, policy, seed, result,
                                   stats["requests"][row["question_id"]], elapsed, unit)
                        out.flush()
                        os.fsync(out.fileno())
                        export()
                        completed += len(chunk)
                        toks = sum(len(o.outputs[0].token_ids) for o in outputs)
                        print(f"seed={seed} policy={name} {offset + len(chunk)}/{len(todo)} "
                              f"{toks / elapsed:.1f} tok/s; "
                              f"{(time.monotonic() - started) / completed:.2f}s/rollout",
                              flush=True)
    export(final=True)
    print(f"Wrote {args.outdir / 'rollouts.csv'}", flush=True)


if __name__ == "__main__":
    main()
