#!/usr/bin/env python3
"""Jobs for EXPERIMENT-reviewer-baselines.md (Parts A-D), run by scripts/gpu_queue.py.

    python scripts/baselines_queue.py > /workspace/baselines_logs/jobs.jsonl
    nohup python scripts/gpu_queue.py --jobs /workspace/baselines_logs/jobs.jsonl \
        --gpus 0,1,2,3 --log /workspace/baselines_logs &
"""
import json

B = "data/baselines"
CONF, GATE, DEV, HARD = ("data/xfer8b/confirm_mathtest400.jsonl", "data/gated/mathtest_gate400.jsonl",
                         "data/policy/dev400.jsonl", f"{B}/hard128.jsonl")
PRE = "source /workspace/env.sh && "
RUN = "python scripts/reasoning_policy_vllm.py --stream --max-num-batched-tokens 8192"
M17 = "--batch 256 --gpu-memory-utilization 0.9"
S17 = f"{B}/seal/seal_1p7b_L20.pt"
S8 = f"{B}/seal/seal_8b_L26.pt"
M8 = ("--model Qwen/Qwen3-8B --layer 26 --batch 256 --gpu-memory-utilization 0.9 "
      "--dir NU26=data/xfer8b/dirs_clean/L26_NU.pt --dir D26=data/xfer8b/dirs_clean/L26_D8.pt "
      f"--dir DU26=data/xfer8b/dirs_clean/L26_DU.pt --dir SEAL={S8}")
M30 = ("--model Qwen/Qwen3-30B-A3B --layer 34 --batch 128 --gpu-memory-utilization 0.93 "
       "--dir NU34=data/xfer30b/dirs_L34/L34_NU.pt --dir D34=data/xfer30b/dirs_L34/L34_D8.pt "
       "--dir DU34=data/xfer30b/dirs_L34/L34_DU.pt")
DEER = "python scripts/baselines_deer.py --max-new-tokens 32768 --batch 256"
jobs = []


def job(name, cmd, out, deps=(), prio=50, gpu=True):
    jobs.append({"name": name, "cmd": PRE + cmd, "done": out, "deps": list(deps), "prio": prio,
                 "gpu": gpu})


def halves(seeds):
    h = seeds // 2
    return [(0, h), (h, seeds)] if seeds == 4 else [(2 * i, 2 * i + 2) for i in range(seeds // 2)]


# --- SEAL vectors (1.7B was extracted by hand, same script, before the queue) --------------
job("seal_8b", "python scripts/baselines_seal_vector.py --model Qwen/Qwen3-8B --layer 26 "
    "--journals 'data/distill_8b/gen/shard*/rollouts.jsonl' "
    "--questions 'data/distill_8b/gen/shard*.jsonl' "
    f"--out {S8} --compare NU26=data/xfer8b/dirs_clean/L26_NU.pt D26=data/xfer8b/dirs_clean/L26_D8.pt",
    S8, prio=1)

# --- Part A + Part D, 1.7B: confirm400 and gate400, 400 x 4 ---------------------------------
A17 = "base N@a1.0d256 N@a0.5d256 ban_all ban_para_d256 SEAL SEALu@a1.0d256"
for tag, coh in (("confirm", CONF), ("gate", GATE)):
    for a, b in halves(4):
        out = f"{B}/a17_{tag}_s{a}{b - 1}"
        job(f"a17_{tag}_s{a}{b - 1}", f"{RUN} {M17} --cohort {coh} --outdir {out} --max-new-tokens "
            f"32768 --seeds {b} --seed-start {a} --policies {A17} --dir SEAL={S17} --dir SEALu={S17}",
            f"{out}/rollouts.csv", prio=10)
    out = f"{B}/d17_{tag}_deer"
    job(f"d17_{tag}_deer", f"{DEER} --model Qwen/Qwen3-1.7B --cohort {coh} --outdir {out} --seeds 4",
        f"{out}/rollouts.csv", prio=11)

# --- Part A + D, 8B: confirm400 (bans + SEAL), gate400 (SEAL); DEER on both ------------------
for a, b in halves(4):
    out = f"{B}/a8_confirm_s{a}{b - 1}"
    job(f"a8_confirm_s{a}{b - 1}", f"{RUN} {M8} --cohort {CONF} --outdir {out} --max-new-tokens 32768 "
        f"--seeds {b} --seed-start {a} --policies ban_all ban_para_d256 SEAL", f"{out}/rollouts.csv",
        deps=["seal_8b"], prio=20)
out = f"{B}/d8_gate_seal"
job("d8_gate_seal", f"{RUN} {M8} --cohort {GATE} --outdir {out} --max-new-tokens 32768 --seeds 4 "
    "--policies SEAL", f"{out}/rollouts.csv", deps=["seal_8b"], prio=21)
for tag, coh in (("confirm", CONF), ("gate", GATE)):
    out = f"{B}/d8_{tag}_deer"
    job(f"d8_{tag}_deer", f"{DEER} --model Qwen/Qwen3-8B --cohort {coh} --outdir {out} --seeds 4",
        f"{out}/rollouts.csv", prio=21)

# --- Part B: DU dev sweep (dev400 x 2, 16k as native D), selection, confirmation ------------
for m, M, d, base in (("8", M8, "DU26", "data/regraded/data/xfer8b/r8b_gdev_g*"),
                      ("30", M30, "DU34", "data/regraded/data/xfer30b/r30_gdev_base")):
    out = f"{B}/b{m}_dev"
    pols = " ".join(f"{d}@a{a}d256" for a in ("0.5", "0.75", "1.0"))
    job(f"b{m}_dev", f"{RUN} {M} --cohort {DEV} --outdir {out} --max-new-tokens 16384 --seeds 2 "
        f"--policies {pols}", f"{out}/rollouts.csv", deps=["seal_8b"] if m == "8" else [], prio=30)
    sel = f"{B}/b{m}_selected.txt"
    job(f"b{m}_select", f"bash scripts/baselines_select.sh b{m} '{base}' {out}", sel,
        deps=[f"b{m}_dev"], prio=30, gpu=False)
    out = f"{B}/b{m}_confirm"
    job(f"b{m}_confirm", f"P=$(cat {sel}); if [ \"$P\" = NONE ]; then mkdir -p {out} && "
        f"echo 'DU failed dev: no confirmation' > {out}/rollouts.csv; else {RUN} {M} --cohort {CONF} "
        f"--outdir {out} --max-new-tokens 32768 --seeds 4 --policies $P; fi", f"{out}/rollouts.csv",
        deps=[f"b{m}_select"], prio=31)

# --- Part C (+ Part D on hard128): hard128 x 8, after the lock ------------------------------
job("lock", "python scripts/baselines_lock.py", f"{B}/protocol_locked.json", deps=["seal_8b"],
    prio=5, gpu=False)
DUSEL = "DU=$(cat {sel}); [ \"$DU\" = NONE ] && DU=''; "
for a, b in halves(8):
    out = f"{B}/c17_s{a}{b - 1}"
    job(f"c17_s{a}{b - 1}", f"{RUN} {M17} --cohort {HARD} --outdir {out} --max-new-tokens 32768 "
        f"--seeds {b} --seed-start {a} --policies base N@a1.0d256 D@a0.5d512 ban_para_d256 SEAL "
        f"--dir SEAL={S17}", f"{out}/rollouts.csv", deps=["lock"], prio=40)
    out = f"{B}/c8_s{a}{b - 1}"
    job(f"c8_s{a}{b - 1}", DUSEL.format(sel=f"{B}/b8_selected.txt") + f"{RUN} {M8} --cohort {HARD} "
        f"--outdir {out} --max-new-tokens 32768 --seeds {b} --seed-start {a} --policies base "
        f"NU26@a0.75d256 D26@a0.5d256 $DU ban_para_d256 SEAL", f"{out}/rollouts.csv",
        deps=["lock", "b8_select"], prio=42)
    out = f"{B}/c30_s{a}{b - 1}"
    job(f"c30_s{a}{b - 1}", DUSEL.format(sel=f"{B}/b30_selected.txt") + f"{RUN} {M30} --cohort {HARD} "
        f"--outdir {out} --max-new-tokens 32768 --seeds {b} --seed-start {a} --policies base "
        f"NU34@a0.75d256 D34@a0.5d256 $DU", f"{out}/rollouts.csv", deps=["lock", "b30_select"], prio=44)
for m, model in (("17", "Qwen/Qwen3-1.7B"), ("8", "Qwen/Qwen3-8B")):
    for a, b in ((0, 4), (4, 8)):
        out = f"{B}/c{m}_deer_s{a}{b - 1}"
        job(f"c{m}_deer_s{a}{b - 1}", f"{DEER} --model {model} --cohort {HARD} --outdir {out} "
            f"--seeds {b} --seed-start {a}", f"{out}/rollouts.csv", deps=["lock"], prio=41 if m == "17" else 43)

for j in jobs:
    print(json.dumps(j))
