#!/usr/bin/env python3
r"""GPU job queue for the distillation pipeline (LOCKED PROTOCOL v2): one job per GPU.

Waits for the teacher generation, builds the SFT/DPO sets, then runs every training and
evaluation job with dependencies (a model's eval starts when its training is DONE).
Phase 1 runs 20-step LR proxies (x{0.25..4}) on the natural arm of each method x peft;
select_* takes the lowest held-out loss (§L10 item 8); phase 2 trains every arm at it. Trainings start
longest-first (DPO, then full SFT, then LoRA SFT); evals take a GPU as
soon as they are runnable. Finished jobs are skipped on restart (DONE / final csv).

    nohup uv run python scripts/distill_queue.py > /workspace/distill_logs/queue.log 2>&1 &
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = Path(os.environ.get("LOGDIR", "/workspace/distill_logs"))
GPUS = [int(g) for g in os.environ.get("GPUS", "0,1,2,3,4,5").split(",")]
MODELS = ROOT / "data/distill/models"
EVAL = ROOT / "data/distill/eval"
EVAL_SETS = "math500 confirm400 gsm8k_test val aime_amc"
PY = ["uv", "run", "python"]

NATURAL = {"sft": "short", "dpo": "natural"}
ARMS = {"sft": ("short", "steered", "ordinary", "steered_short"), "dpo": ("natural", "steered")}
LR = {("sft", "lora"): 1e-4, ("sft", "full"): 1e-5, ("dpo", "lora"): 5e-6, ("dpo", "full"): 5e-7}
MULT = (0.25, 0.5, 1.0, 2.0, 4.0)  # §L10 item 8
SWEEP_STEPS = 20
HELDOUT = {"sft": 256, "dpo": 128}
GROUPS = [("dpo", "full"), ("dpo", "lora"), ("sft", "full"), ("sft", "lora")]
SWEEP = ROOT / "data/distill/sweep"


def sweep_name(m: str, p: str, mult: float) -> str:
    return f"sweep_{m}_{NATURAL[m]}_{p}_lr{LR[(m, p)] * mult:g}"


def eval_job(name: str, model: Path | str, deps: list[str]) -> dict:
    return {"name": f"eval_{name}", "deps": deps, "done": EVAL / name / "rollouts.csv",
            "cmd": PY + ["scripts/distill_eval.py", "--model", str(model), "--name", name,
                         "--sets", *EVAL_SETS.split()]}


def select(m: str, p: str) -> None:
    """§L10 item 8: the LR with the lowest held-out loss after SWEEP_STEPS; ties -> lower LR."""
    rows = []
    for mult in MULT:
        r = json.loads((MODELS / sweep_name(m, p, mult) / "sweep_result.json").read_text())
        rows.append({"lr": r["lr"], "heldout": r["heldout"],
                     "final_heldout_loss": r["final_heldout_loss"]})
    win = min(rows, key=lambda r: (r["final_heldout_loss"], r["lr"]))
    edge = win["lr"] in (rows[0]["lr"], rows[-1]["lr"])
    SWEEP.mkdir(parents=True, exist_ok=True)
    (SWEEP / f"{m}_{p}.json").write_text(json.dumps(
        {"lr": win["lr"], "rule": f"lowest held-out loss after {SWEEP_STEPS} steps",
         "at_grid_edge": edge, "candidates": rows}, indent=1) + "\n")
    print(f"select {m}/{p}: lr {win['lr']:g}{' (GRID EDGE)' if edge else ''}; " + "; ".join(
        f"{r['lr']:g}: {r['final_heldout_loss']:.5f}" for r in rows), flush=True)


def jobs() -> list[dict]:
    out = [eval_job("base", "Qwen/Qwen3-1.7B", [])]
    for m, p in GROUPS:  # phase 1: 20-step proxies on the natural arm
        for mult in MULT:
            n = sweep_name(m, p, mult)
            out.append({"name": f"train_{n}", "deps": [], "done": MODELS / n / "DONE",
                        "cmd": PY + ["scripts/distill_train.py", "--method", m, "--arm",
                                     NATURAL[m], "--peft", p, "--name", n,
                                     "--lr", f"{LR[(m, p)] * mult:g}",
                                     "--max-steps", str(SWEEP_STEPS),
                                     "--heldout", str(HELDOUT[m])]})
    for m, p in GROUPS:  # selection, then every arm at the selected LR
        sel = f"select_{m}_{p}"
        out.append({"name": sel, "fn": (m, p), "done": SWEEP / f"{m}_{p}.json",
                    "deps": [f"train_{sweep_name(m, p, x)}" for x in MULT]})
        for a in ARMS[m]:
            n = f"{m}_{a}_{p}"
            out.append({"name": f"train_{n}", "deps": [sel], "done": MODELS / n / "DONE",
                        "cmd": PY + ["scripts/distill_train.py", "--method", m, "--arm", a,
                                     "--peft", p, "--lr-from", str(SWEEP / f"{m}_{p}.json")]})
            out.append(eval_job(n, MODELS / n, [f"train_{n}"]))
    return out


def wait_generation() -> None:
    shards = sorted((ROOT / "data/distill/gen").glob("shard*/run_manifest.json"))
    while True:
        done = [p.parent for p in shards if (p.parent / "rollouts.csv").exists()]
        if len(shards) == 6 and len(done) == 6 and not (LOG / "HOLD").exists():
            return
        print(f"generation: {len(done)}/6 shards done", flush=True)
        time.sleep(300)


def main() -> None:
    wait_generation()
    sets = ROOT / "data/distill/sets/sets_stats.json"
    if not sets.exists():
        r = subprocess.run(PY + ["scripts/distill_build_sets.py"], cwd=ROOT,
                           stdout=open(LOG / "build_sets.log", "w"), stderr=subprocess.STDOUT)
        if r.returncode:
            sys.exit("build_sets failed")
    js = jobs()
    state = {j["name"]: ("done" if j["done"].exists() else "todo") for j in js}
    running: dict[int, tuple[str, subprocess.Popen]] = {}
    while any(v in ("todo", "run") for v in state.values()):
        for g, (name, proc) in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            job = next(j for j in js if j["name"] == name)
            state[name] = "done" if rc == 0 and job["done"].exists() else "failed"
            print(f"{time.strftime('%H:%M')} {name} on GPU {g}: {state[name]} (rc {rc})",
                  flush=True)
            del running[g]
        free = [g for g in GPUS if g not in running]
        # runnable: deps done; evals first (they unblock the readout), then trainings in order
        ready = [j for j in js if state[j["name"]] == "todo"
                 and all(state[d] == "done" for d in j["deps"])]
        ready.sort(key=lambda j: (not j["name"].startswith("eval_") or j["name"] == "eval_base",
                                  js.index(j)))
        for job in [j for j in ready if "fn" in j]:
            try:
                select(*job["fn"])
                state[job["name"]] = "done"
            except Exception as e:  # noqa: BLE001
                print(f"{job['name']} failed: {e!r}", flush=True)
                state[job["name"]] = "failed"
        ready = [j for j in ready if "fn" not in j]
        for g, job in zip(free, ready):
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(g)}
            proc = subprocess.Popen(job["cmd"], cwd=ROOT, env=env,
                                    stdout=open(LOG / f"{job['name']}.log", "a"),
                                    stderr=subprocess.STDOUT)
            running[g] = (job["name"], proc)
            state[job["name"]] = "run"
            print(f"{time.strftime('%H:%M')} start {job['name']} on GPU {g} pid {proc.pid}",
                  flush=True)
        if any(state[d] == "failed" for j in js for d in j["deps"]
               if state[j["name"]] == "todo"):
            for j in js:
                if state[j["name"]] == "todo" and any(state[d] == "failed" for d in j["deps"]):
                    state[j["name"]] = "blocked"
        time.sleep(30)
    print("queue finished:", {k: v for k, v in state.items() if v != "done"} or "all done",
          flush=True)
    subprocess.run(PY + ["scripts/distill_report.py"], cwd=ROOT,
                   stdout=open(LOG / "report.log", "w"), stderr=subprocess.STDOUT)


if __name__ == "__main__":
    main()
