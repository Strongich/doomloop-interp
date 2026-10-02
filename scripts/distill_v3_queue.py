#!/usr/bin/env python3
r"""PROTOCOL v3 pipeline queue: waits for the top-up generation, builds the §V4 sets, then
  SFT  LR proxy (20 steps, x{0.25..4}, natural twin, LoRA) -> select -> sft_natural, sft_nla
  DPO  LR proxy on the natural twin, init = merged sft_natural -> select ->
       dpo_natural (init sft_natural), dpo_nla (init sft_nla); reference = the init model
  eval base + the 4 models on mathfresh (primary), math500, gsm8k_test
Jobs start only on GPUs with < 1 GB in use (v2 evals / top-up may still hold some).

    nohup uv run python scripts/distill_v3_queue.py > /workspace/distill_logs/v3queue.log 2>&1 &
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from distill_queue import LOG, PY  # noqa: E402

V3 = ROOT / "data/distill_v3"
MODELS, EVAL, SWEEP, SETS = V3 / "models", V3 / "eval", V3 / "sweep", V3 / "sets"
LR = {"sft": 5e-5, "dpo": 2.5e-6}  # the v2 LoRA selections are the grid centres
MULT = (0.25, 0.5, 1.0, 2.0, 4.0)
HELD = {"sft": 256, "dpo": 128}
EVAL_SETS = ["mathfresh", "math500", "gsm8k_test"]


def sname(m: str, x: float) -> str:
    return f"sweep_{m}_natural_lora_lr{LR[m] * x:g}"


def train(m: str, arm: str, name: str, init: str | None, extra: list[str]) -> list[str]:
    c = PY + ["scripts/distill_train.py", "--method", m, "--arm", arm, "--peft", "lora",
              "--sets", str(SETS), "--out", str(MODELS), "--name", name, *extra]
    return c + (["--init", init] if init else [])


def jobs() -> list[dict]:
    J = [{"name": "eval_base", "deps": [], "done": EVAL / "base/rollouts.csv",
          "cmd": PY + ["scripts/distill_eval.py", "--model", "Qwen/Qwen3-1.7B", "--name", "base",
                       "--outroot", str(EVAL), "--sets", *EVAL_SETS]}]
    for m in ("sft", "dpo"):
        init = str(MODELS / "sft_natural") if m == "dpo" else None
        dep = ["train_sft_natural"] if m == "dpo" else []
        for x in MULT:
            n = sname(m, x)
            J.append({"name": f"train_{n}", "deps": dep, "done": MODELS / n / "DONE",
                      "cmd": train(m, "natural", n, init, ["--lr", f"{LR[m] * x:g}",
                                   "--max-steps", "20", "--heldout", str(HELD[m])])})
        J.append({"name": f"select_{m}", "fn": m, "done": SWEEP / f"{m}.json",
                  "deps": [f"train_{sname(m, x)}" for x in MULT]})
        for twin in ("natural", "nla"):
            n = f"{m}_{twin}"
            ini = str(MODELS / f"sft_{twin}") if m == "dpo" else None
            deps = [f"select_{m}"] + ([f"train_sft_{twin}"] if m == "dpo" else [])
            J.append({"name": f"train_{n}", "deps": deps, "done": MODELS / n / "DONE",
                      "cmd": train(m, twin, n, ini, ["--lr-from", str(SWEEP / f"{m}.json")])})
            J.append({"name": f"eval_{n}", "deps": [f"train_{n}"], "done": EVAL / n / "rollouts.csv",
                      "cmd": PY + ["scripts/distill_eval.py", "--model", str(MODELS / n), "--name", n,
                                   "--outroot", str(EVAL), "--sets", *EVAL_SETS]})
    return J


def select(m: str) -> None:
    rows = [json.loads((MODELS / sname(m, x) / "sweep_result.json").read_text()) for x in MULT]
    rows = [{"lr": r["lr"], "final_heldout_loss": r["final_heldout_loss"], "heldout": r["heldout"]}
            for r in rows]
    win = min(rows, key=lambda r: (r["final_heldout_loss"], r["lr"]))
    edge = win["lr"] in (rows[0]["lr"], rows[-1]["lr"])
    SWEEP.mkdir(parents=True, exist_ok=True)
    (SWEEP / f"{m}.json").write_text(json.dumps({"lr": win["lr"], "at_grid_edge": edge,
                                                 "candidates": rows}, indent=1) + "\n")
    print(f"select {m}: lr {win['lr']:g}{' (GRID EDGE)' if edge else ''}; "
          + "; ".join(f"{r['lr']:g}: {r['final_heldout_loss']:.5f}" for r in rows), flush=True)


def free_gpus(busy: set[int]) -> list[int]:
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used",
                                   "--format=csv,noheader,nounits"], text=True)
    return [int(i) for i, mem in (l.split(", ") for l in out.strip().splitlines())
            if int(mem) < 1000 and int(i) not in busy]


def build(kind: str) -> None:
    """kind 'sft': easy-question SFT sets (v2 rollouts only), runnable at once.
    kind 'dpo': the full build, once the top-up is complete. Each waits on its HOLD file
    (V3HOLD_SFT / V3HOLD) so the statistics are reviewed before training."""
    if kind == "dpo" and len(list((V3 / "gen").glob("shard*/rollouts.csv"))) < 6:
        raise RuntimeError("top-up not complete")
    target = SETS / ("sft_stats.json" if kind == "sft" else "sets_stats.json")
    if not target.exists():
        r = subprocess.run(PY + ["scripts/distill_v3_build_sets.py"]
                           + (["--sft-only"] if kind == "sft" else []), cwd=ROOT,
                           stdout=open(LOG / f"v3_build_{kind}.log", "w"), stderr=subprocess.STDOUT)
        if r.returncode:
            raise RuntimeError(f"v3 build {kind} failed")
    if (LOG / ("V3HOLD_SFT" if kind == "sft" else "V3HOLD")).exists():
        raise RuntimeError("hold")


def main() -> None:
    js = jobs()
    js.insert(0, {"name": "build_sft", "build": "sft", "deps": [], "done": LOG / "V3_SFT_RELEASED"})
    js.insert(1, {"name": "build_dpo", "build": "dpo", "deps": [], "done": LOG / "V3_DPO_RELEASED"})
    for j in js:
        if j["name"].startswith(("train_sweep_sft", "train_sft_")):
            j["deps"] = j["deps"] + ["build_sft"]
        if j["name"].startswith(("train_sweep_dpo", "train_dpo_")):
            j["deps"] = j["deps"] + ["build_dpo"]
    state = {j["name"]: "done" if j["done"].exists() else "todo" for j in js}
    running: dict[int, tuple[str, subprocess.Popen]] = {}
    while any(v in ("todo", "run") for v in state.values()):
        for g, (name, proc) in list(running.items()):
            if proc.poll() is None:
                continue
            job = next(j for j in js if j["name"] == name)
            state[name] = "done" if proc.returncode == 0 and job["done"].exists() else "failed"
            print(f"{time.strftime('%H:%M')} {name} on GPU {g}: {state[name]}", flush=True)
            del running[g]
        ready = [j for j in js if state[j["name"]] == "todo"
                 and all(state[d] == "done" for d in j["deps"])]
        for j in [j for j in js if "build" in j and state[j["name"]] == "todo"]:
            try:
                build(j["build"])
                (j["done"]).write_text("released\n")
                state[j["name"]] = "done"
                print(f"{time.strftime('%H:%M')} {j['name']} released", flush=True)
            except RuntimeError as e:
                if str(e) not in ("hold", "top-up not complete"):
                    print(f"{j['name']} failed: {e}", flush=True)
                    state[j["name"]] = "failed"
        ready = [j for j in ready if "build" not in j]
        for j in [j for j in ready if "fn" in j]:
            try:
                select(j["fn"])
                state[j["name"]] = "done"
            except Exception as e:  # noqa: BLE001
                print(f"{j['name']} failed: {e!r}", flush=True)
                state[j["name"]] = "failed"
        for j in js:
            if state[j["name"]] == "todo" and any(state[d] in ("failed", "blocked") for d in j["deps"]):
                state[j["name"]] = "blocked"
        ready = [j for j in ready if "fn" not in j and state[j["name"]] == "todo"]
        ready.sort(key=lambda j: (j["name"] == "eval_base", js.index(j)))  # base eval last-ish
        for g, j in zip(free_gpus(set(running)), ready):
            proc = subprocess.Popen(j["cmd"], cwd=ROOT, env={**__import__("os").environ,
                                    "CUDA_VISIBLE_DEVICES": str(g)},
                                    stdout=open(LOG / f"v3_{j['name']}.log", "a"),
                                    stderr=subprocess.STDOUT)
            running[g] = (j["name"], proc)
            state[j["name"]] = "run"
            print(f"{time.strftime('%H:%M')} start {j['name']} on GPU {g}", flush=True)
            time.sleep(60)  # let the job allocate before the next free-GPU check
        time.sleep(30)
    print("v3 queue finished:", {k: v for k, v in state.items() if v != "done"} or "all done",
          flush=True)


if __name__ == "__main__":
    main()
