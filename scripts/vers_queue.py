#!/usr/bin/env python3
"""EXPERIMENT-vers-doubt.md: gpu_queue jobs file. Order (prio): dev -> select -> P on hard128
x 8 -> P on gate400 x 4 -> -N on hard128 x 8 (last; the only arm that may be cut).
Test jobs read data/vers/selected.txt; NONE skips them (reading M0).

    python scripts/vers_queue.py && nohup python scripts/gpu_queue.py \
        --jobs /workspace/vers_logs/jobs.jsonl --gpus 0 --log /workspace/vers_logs &
"""
import json
from pathlib import Path

OUT = Path("/workspace/vers_logs/jobs.jsonl")
V = "data/vers"
RUN = ("source /workspace/env.sh && python scripts/reasoning_policy_vllm.py --stream "
       "--max-num-batched-tokens 8192 --batch 256 --gpu-memory-utilization 0.9 "
       f"--dir P={V}/dir_P_doubt.pt --dir negN={V}/dir_negN.pt")
jobs = []


def job(name, cmd, done, deps=(), prio=50, gpu=True):
    jobs.append({"name": name, "cmd": cmd, "done": done, "deps": list(deps), "prio": prio,
                 "gpu": gpu})


job("dev", f"{RUN} --cohort data/policy/dev400.jsonl --outdir {V}/dev --max-new-tokens 16384 "
    "--seeds 2 --policies base P@a0.25d256 P@a0.5d256 P@a1.0d256",
    f"{V}/dev/rollouts.csv", prio=10)
job("select", f"source /workspace/env.sh && python scripts/vers_select.py --dev {V}/dev "
    f"| tee /workspace/vers_logs/select.log", f"{V}/selected.txt", ["dev"], prio=10, gpu=False)


def test(name, cohort, s0, prio, negn=False):
    out = f"{V}/{name}"
    pol = "${P/#P@/negN@}" if negn else "$P"
    job(name, f"source /workspace/env.sh && P=$(cat {V}/selected.txt); mkdir -p {out}; "
        f"if [ \"$P\" = NONE ]; then echo 'no test run: selected NONE' > {out}/SKIPPED.txt; "
        f"else {RUN} --cohort {cohort} --outdir {out} --max-new-tokens 32768 --seeds 2 "
        f"--seed-start {s0} --policies {pol}; fi && touch {out}/done.marker",
        f"{out}/done.marker", ["select"], prio)


for s in (0, 2, 4, 6):
    test(f"h128_s{s}{s + 1}", "data/baselines/hard128.jsonl", s, 20)
for s in (0, 2):
    test(f"gate_s{s}{s + 1}", "data/gated/mathtest_gate400.jsonl", s, 30)
for s in (0, 2, 4, 6):
    test(f"negN_s{s}{s + 1}", "data/baselines/hard128.jsonl", s, 40, negn=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text("".join(json.dumps(j) + "\n" for j in jobs))
print(f"wrote {OUT}: {len(jobs)} jobs")
