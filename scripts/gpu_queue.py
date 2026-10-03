#!/usr/bin/env python3
r"""Shared one-job-per-GPU queue for concurrent experiments (selective-doubt, pipeline-sft).

Reads a jobs file (jsonl) every poll, so jobs can be appended while it runs. A job:
  {"name": str, "cmd": str (bash), "done": path, "deps": [names], "prio": int (lower first),
   "gpu": bool (default true; false = runs without a GPU, any number at once)}
A job is done when its `done` path exists; a nonzero exit or a missing marker writes
<LOG>/<name>.failed and the job is not retried (delete the file to retry). Jobs whose deps
failed are skipped. The queue exits when every job is done/failed/blocked and the file has
not changed for one poll, unless --forever.

    nohup uv run python scripts/gpu_queue.py --jobs /workspace/distill_logs/jobs.jsonl &
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    ap.add_argument("--gpus", default="0,1,2,3,4,5")
    ap.add_argument("--log", type=Path, default=Path("/workspace/distill_logs"))
    ap.add_argument("--forever", action="store_true")
    args = ap.parse_args()
    gpus = [int(g) for g in args.gpus.split(",")]
    running: dict[str, tuple[int | None, subprocess.Popen]] = {}
    idle_polls = 0
    while True:
        js = [json.loads(x) for x in args.jobs.read_text().splitlines() if x.strip()]
        byname = {j["name"]: j for j in js}

        def state(n: str) -> str:
            if n in running:
                return "run"
            j = byname.get(n)
            if j is None:
                return "missing"
            if (ROOT / j["done"]).exists():
                return "done"
            if (args.log / f"{n}.failed").exists():
                return "failed"
            return "todo"

        for n, (g, p) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            del running[n]
            ok = rc == 0 and (ROOT / byname[n]["done"]).exists()
            if not ok:
                (args.log / f"{n}.failed").write_text(f"rc {rc}\n")
            print(f"{time.strftime('%m-%d %H:%M')} {n} (GPU {g}): {'done' if ok else f'FAILED rc {rc}'}",
                  flush=True)
        # GPUs pinned by a hand-started process can be reserved with <LOG>/reserve_gpu<N>
        busy = {g for g, _ in running.values()} | {
            g for g in gpus if (args.log / f"reserve_gpu{g}").exists()}
        free = [g for g in gpus if g not in busy]
        ready = []
        for j in js:
            if state(j["name"]) != "todo":
                continue
            ds = [state(d) for d in j.get("deps", [])]
            if all(d == "done" for d in ds):
                ready.append(j)
        ready.sort(key=lambda j: (j.get("prio", 50), js.index(j)))
        for j in ready:
            use_gpu = j.get("gpu", True)
            if use_gpu and not free:
                continue
            g = free.pop(0) if use_gpu else None
            env = {**os.environ}
            if g is not None:
                env["CUDA_VISIBLE_DEVICES"] = str(g)
            p = subprocess.Popen(["bash", "-c", j["cmd"]], cwd=ROOT, env=env,
                                 stdout=open(args.log / f"{j['name']}.log", "a"),
                                 stderr=subprocess.STDOUT)
            running[j["name"]] = (g, p)
            print(f"{time.strftime('%m-%d %H:%M')} start {j['name']} on GPU {g} pid {p.pid}",
                  flush=True)
        pending = [j for j in js if state(j["name"]) == "todo"
                   and not any(state(d) in ("failed", "missing") for d in j.get("deps", []))]
        if not running and not pending and not args.forever:
            idle_polls += 1
            if idle_polls >= 2:
                print("queue idle; exiting", flush=True)
                return
        else:
            idle_polls = 0
        time.sleep(20)


if __name__ == "__main__":
    main()
