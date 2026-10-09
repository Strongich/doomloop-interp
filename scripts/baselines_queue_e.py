#!/usr/bin/env python3
"""Jobs for Amendment E of EXPERIMENT-reviewer-baselines.md (SEAL fairness fix).

    python scripts/baselines_queue_e.py > /workspace/baselines_logs/jobs_e.jsonl
"""
import json

B = "data/baselines"
CONF, GATE, DEV = ("data/xfer8b/confirm_mathtest400.jsonl", "data/gated/mathtest_gate400.jsonl",
                   "data/policy/dev400.jsonl")
PRE = "source /workspace/env.sh && "
RUN = "python scripts/reasoning_policy_vllm.py --stream --max-num-batched-tokens 8192"
S17, S8 = f"{B}/seal/seal_1p7b_L20.pt", f"{B}/seal/seal_8b_L26.pt"
M = {"17": f"--batch 256 --gpu-memory-utilization 0.9 --dir SEAL={S17} --dir SEALu={S17}",
     "8": f"--model Qwen/Qwen3-8B --layer 26 --batch 256 --gpu-memory-utilization 0.9 --dir SEAL={S8}"}
ALPHAS = ("0.5", "1.0", "1.5", "2.0")
jobs = []


def job(name, cmd, done, deps=(), prio=50, gpu=True):
    jobs.append({"name": name, "cmd": PRE + cmd, "done": done, "deps": list(deps), "prio": prio,
                 "gpu": gpu})


# E1: SEAL's unit vector on N's schedule at SEAL-like strengths (1.7B)
E1 = "SEALu@a0.25d256 SEALu@a0.34d256 SEALu@a0.5d256"
for tag, coh in (("confirm", CONF), ("gate", GATE)):
    for a, b in ((0, 2), (2, 4)):
        out = f"{B}/e1_{tag}_s{a}{b - 1}"
        job(f"e1_{tag}_s{a}{b - 1}", f"{RUN} {M['17']} --cohort {coh} --outdir {out} "
            f"--max-new-tokens 32768 --seeds {b} --seed-start {a} --policies {E1}",
            f"{out}/rollouts.csv", prio=20)

# E2 dev: SEAL@a{0.5..2.0} on dev400 x 2 seeds, 16k (1.7B with its own base; 8B vs regraded gdev)
bases = {"17": f"{B}/e2_dev17", "8": "data/regraded/data/xfer8b/r8b_gdev_g*"}
for m in ("17", "8"):
    out = f"{B}/e2_dev{m}"
    pols = ("base " if m == "17" else "") + " ".join(f"SEAL@a{a}" for a in ALPHAS)
    job(f"e2_dev{m}", f"{RUN} {M[m]} --cohort {DEV} --outdir {out} --max-new-tokens 16384 "
        f"--seeds 2 --policies {pols}", f"{out}/rollouts.csv", prio=10)
    sel = f"{B}/e2_{m}_selected.txt"
    job(f"e2_select{m}", f"bash scripts/baselines_select.sh e2_{m} '{bases[m]}' {out}", sel,
        deps=[f"e2_dev{m}"], prio=10, gpu=False)
    # E2 test: the selected alpha on confirm400 and gate400 x 4, unless NONE or SEAL@a1.0
    for tag, coh in (("confirm", CONF), ("gate", GATE)):
        for a, b in ((0, 2), (2, 4)):
            out = f"{B}/e2_{m}_{tag}_s{a}{b - 1}"
            job(f"e2_{m}_{tag}_s{a}{b - 1}",
                f"P=$(cat {sel}); mkdir -p {out}; if [ \"$P\" = NONE ] || [ \"$P\" = SEAL@a1.0 ]; "
                f"then echo \"no test run: selected $P\" > {out}/SKIPPED.txt; else {RUN} {M[m]} "
                f"--cohort {coh} --outdir {out} --max-new-tokens 32768 --seeds {b} --seed-start {a} "
                f"--policies $P; fi && touch {out}/done.marker", f"{out}/done.marker",
                deps=[f"e2_select{m}"], prio=15)
for j in jobs:
    print(json.dumps(j))
