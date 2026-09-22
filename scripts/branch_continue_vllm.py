#!/usr/bin/env python3
"""vLLM backend for branch_continue.py; isolated, manifested, resumable results."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import time
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def read_journal(path: Path) -> list[dict[str, Any]]:
    """Recover only an interrupted final write; malformed complete rows are errors."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    records = []
    consumed = 0
    for line in raw.splitlines(keepends=True):
        try:
            records.append(json.loads(line))
        except (json.JSONDecodeError, UnicodeDecodeError):
            if consumed + len(line) != len(raw) or line.endswith(b"\n"):
                raise ValueError(f"Malformed complete journal record at byte {consumed}") from None
            print(f"Recovering incomplete final journal write at byte {consumed}", flush=True)
            with path.open("r+b") as f:
                f.truncate(consumed)
                f.flush()
                os.fsync(f.fileno())
            return records
        consumed += len(line)
    if raw and not raw.endswith(b"\n"):
        with path.open("ab") as f:
            f.write(b"\n")
    return records


def request_seed(question_id: str, seed: int) -> int:
    key = f"{question_id}:{seed}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "little") & 0x7fffffff


def run(args: Any) -> None:
    from transformers import AutoTokenizer
    from vllm import SamplingParams
    from branch_continue import DIRECTIONS, FIELDS
    from suppress_answer import NEWLINE_CHAR, doubt_stats
    from reasoning_attention.config import MODEL_ID, NLAConfig, SamplingDefaults
    from reasoning_attention.grading import grade
    from reasoning_attention.serving.vllm_steering import build_steering_llm
    from reasoning_attention.tokenview import _chat_header

    if args.batch < 1 or args.checkpoint_every < 1 or args.seeds < 1:
        raise ValueError("batch, checkpoint-every, and seeds must be positive")
    unknown = set(args.arms) - {"base", "exit", *DIRECTIONS}
    if unknown:
        raise ValueError(f"Unknown arms: {unknown}")
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    rows = [json.loads(x) for x in args.prefixes.read_text().splitlines() if x.strip()]
    if args.limit:
        rows = rows[:args.limit]
    if not rows or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Need nonempty prefixes with unique question IDs")
    sampling = SamplingDefaults()
    boundaries = [i for i in range(len(tok))
                  if tok.convert_ids_to_tokens(i).count(NEWLINE_CHAR) >= 2]
    boundary_set = set(boundaries)
    close_id = int(tok.convert_tokens_to_ids("</think>"))
    stop_ids = sorted({int(tok.eos_token_id), int(tok.convert_tokens_to_ids("<|endoftext|>"))})
    units = {arm: torch.load(DIRECTIONS[arm], map_location="cpu", weights_only=False)["unit"]
             .float().tolist() for arm in args.arms if arm in DIRECTIONS}
    prompts = {}
    for row in rows:
        for arm in args.arms:
            suffix = "\n</think>\n\n" if arm == "exit" else ""
            text = _chat_header(tok, row["question"]) + row["prefix"] + suffix
            prompts[(row["question_id"], arm)] = tok(text, add_special_tokens=True)["input_ids"]
    max_len = max(len(ids) + (args.exit_tokens if arm == "exit" else args.max_new_tokens)
                  for (_, arm), ids in prompts.items())
    args.outdir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backend": "vllm", "protocol": "residual-boundary-v1",
        "model": MODEL_ID, "model_commit": tok.init_kwargs.get("_commit_hash"),
        "versions": {p: version(p) for p in ("vllm", "torch", "transformers")},
        "prefix_sha256": digest(args.prefixes), "question_ids": [r["question_id"] for r in rows],
        "directions": {a: digest(Path(DIRECTIONS[a])) for a in units},
        "code": {p: digest(Path(p)) for p in (
            "scripts/branch_continue_vllm.py", "scripts/branch_continue.py",
            "src/reasoning_attention/serving/vllm_steering.py")},
        "arms": args.arms, "seeds": args.seeds, "alpha": args.alpha,
        "max_new_tokens": args.max_new_tokens, "exit_tokens": args.exit_tokens,
        "temperature": sampling.temperature, "top_p": sampling.top_p, "top_k": sampling.top_k,
        "layer": NLAConfig().extraction_layer, "max_model_len": max_len,
        "max_num_seqs": args.batch, "checkpoint_every": args.checkpoint_every,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "enforce_eager": True, "async_scheduling": False, "prefix_caching": False,
        "model_runner": "V1 within vLLM 0.22.0", "sampler": "native",
        "seed_scheme": "sha256(question_id:seed), same across arms",
        "metrics": "injections count consumed sites; unclosed thinking counts all output tokens",
    }
    manifest_path = args.outdir / "run_manifest.json"
    journal = args.outdir / "branches_vllm.jsonl"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Run manifest differs; use a new output directory")
    else:
        if any(args.outdir.glob("branches*")):
            raise ValueError("Existing unmanifested/HF outputs; use a fresh vLLM output directory")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    records = read_journal(journal)
    done = {(r["question_id"], r["arm"], r["seed"]) for r in records}
    if len(done) != len(records):
        raise ValueError("Duplicate branch records in journal")

    def export(final: bool = False) -> None:
        csv_path = args.outdir / ("branches.csv" if final else "branches.csv.partial")
        tmp = csv_path.with_name(csv_path.name + ".tmp")
        with tmp.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({k: row[k] for k in FIELDS} for row in records)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, csv_path)
        text_path = args.outdir / "branches_texts.jsonl"
        tmp = text_path.with_suffix(".tmp")
        with tmp.open("w") as f:
            for row in records:
                f.write(json.dumps({k: row[k] for k in (
                    "question_id", "arm", "seed", "gold", "band", "cand", "text")}) + "\n")
        os.replace(tmp, text_path)

    if len(done) == len(rows) * len(args.arms) * args.seeds:
        export(final=True)
        print("All vLLM branches already complete")
        return
    print(f"vLLM: {len(rows)} questions, {len(done)} resumed; max context {max_len}", flush=True)
    llm = build_steering_llm(
        MODEL_ID, NLAConfig().extraction_layer, boundaries, close_id, max_len,
        args.batch, args.gpu_memory_utilization, args.max_num_batched_tokens,
    )
    started = time.monotonic()
    completed = 0
    with journal.open("a") as out:
        for seed in range(args.seeds):
            for arm in args.arms:
                todo = [r for r in rows if (r["question_id"], arm, seed) not in done]
                for offset in range(0, len(todo), args.checkpoint_every):
                    chunk = todo[offset:offset + args.checkpoint_every]
                    llm.collective_rpc("configure_reasoning_steering",
                                       args=(units.get(arm), args.alpha, arm != "exit"))
                    inputs = [{"prompt_token_ids": prompts[(r["question_id"], arm)]} for r in chunk]
                    params = [SamplingParams(
                        temperature=sampling.temperature, top_p=sampling.top_p, top_k=sampling.top_k,
                        max_tokens=args.exit_tokens if arm == "exit" else args.max_new_tokens,
                        seed=request_seed(r["question_id"], seed), stop_token_ids=stop_ids,
                        extra_args={"steering_audit_id": r["question_id"]},
                    ) for r in chunk]
                    batch_start = time.monotonic()
                    outputs = llm.generate(inputs, params, use_tqdm=False)
                    generation_seconds = time.monotonic() - batch_start
                    stats = llm.collective_rpc("reasoning_steering_stats")[0]
                    if stats["calls"] == 0:
                        raise RuntimeError("Steering hook did not execute")
                    for row, result in zip(chunk, outputs, strict=True):
                        generated = result.outputs[0]
                        ids = list(generated.token_ids)
                        prompt_len = len(result.prompt_token_ids)
                        close = ids.index(close_id) if close_id in ids else None
                        thinking_end = close if close is not None else len(ids)
                        # The last output token is never consumed on a finished request.
                        sites = [prompt_len + i for i, token in enumerate(ids[:-1])
                                 if arm != "exit" and i < thinking_end and token in boundary_set]
                        expected_inj = sites if arm in units and args.alpha != 0 else []
                        actual = stats["requests"][row["question_id"]]
                        if actual["sites"] != sites or actual["injections"] != expected_inj:
                            raise RuntimeError(f"Injection-site mismatch: {actual}, expected {sites}")
                        text = tok.decode(ids, skip_special_tokens=False)
                        g = grade(text, row["gold"])
                        head = "<think>\n</think>\n" if arm == "exit" else "<think>\n"
                        markers, _, doubt_blocks = doubt_stats(head + text)
                        boundaries_n = sum(t in boundary_set for t in ids[:thinking_end]) if arm != "exit" else 0
                        cc = int(row["cand_correct"])
                        flip = ("keep" if cc and g.is_correct else "break" if cc else
                                "repair" if g.is_correct else "stuck")
                        record = {
                            "question_id": row["question_id"], "dataset": row["dataset"],
                            "gold": str(row["gold"]), "band": row["band"], "arm": arm, "seed": seed,
                            "cand": row["cand"], "cand_correct": cc, "freeze_at": row["freeze_at"],
                            "correct": int(g.is_correct), "has_answer": int(g.has_answer),
                            "status": g.status, "flip": flip, "cont_tokens": len(ids),
                            "cont_think_tokens": 0 if arm == "exit" else thinking_end + int(close is not None),
                            "capped": int(generated.finish_reason == "length"),
                            "closed": int(arm == "exit" or close is not None),
                            "boundaries": boundaries_n, "injections": len(actual["injections"]),
                            "markers": markers, "doubt_blocks": doubt_blocks,
                            "doubt_rate": round(doubt_blocks / boundaries_n, 4) if boundaries_n else -1.,
                            "text": text, "token_ids": ids, "hook": actual,
                            "request_seed": request_seed(row["question_id"], seed),
                            "finish_reason": generated.finish_reason,
                            "batch_generation_seconds": generation_seconds,
                        }
                        out.write(json.dumps(record) + "\n")
                        records.append(record)
                        done.add((row["question_id"], arm, seed))
                    out.flush()
                    os.fsync(out.fileno())
                    export()
                    completed += len(chunk)
                    tokens = sum(len(o.outputs[0].token_ids) for o in outputs)
                    print(f"seed={seed} arm={arm} {offset+len(chunk)}/{len(todo)} "
                          f"{tokens/generation_seconds:.1f} output tok/s; "
                          f"{(time.monotonic()-started)/completed:.2f}s/branch", flush=True)
    export(final=True)
    print(f"Wrote {args.outdir / 'branches.csv'}", flush=True)
