#!/usr/bin/env python3
r"""EXPERIMENT-probe-gated.md §G5-§G8: write the gpu_queue jobs (run after gate_build --lock).

  phase 1  gate400 (6 shards): base, N, N@a0.5, G5, G10, G20 x 4 seeds
           MATH-500 L1-3 (2 shards, 4 seeds) and GSM8K test (2 shards, 1 seed): base, N, G5/10/20
           AIME2025 + AMC23 (2 shards, 8 seeds): base, N, G10
  calib    targets from the G arms -> 3 x 3 grid on calib50 (3 shards, 2 seeds) -> p_q
  phase 2  gate400 (6 shards): R5, R10, R20 x 4 seeds; AIME/AMC: R10 x 8 seeds
  mech     200 random rollouts per arm (base, N, G10, R10) of gate400, §S3 probe + labels
  report   scripts/gate_report.py

    uv run python scripts/gate_queue.py --jobs /workspace/distill_logs/gate_jobs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
G = "data/gated"
N = "N@a1.0d256"
P1 = f"base {N} N@a0.5d256 {N}G5 {N}G10 {N}G20"
EASY = f"base {N} {N}G5 {N}G10 {N}G20"
RUN = ("uv run python scripts/reasoning_policy_vllm.py --cohort {c} --outdir {o} --policies {p} "
       "--seeds {s} --max-new-tokens 32768 --batch 512 --max-num-batched-tokens 8192 "
       "--gpu-memory-utilization 0.90 --checkpoint-every 500 --stream --gate-probe data/gated/probe.pt")
RPOL = "$(uv run python scripts/gate_calibrate.py --policies {q})"


def shard(src: str, name: str, n: int, keep=lambda r: True) -> list[str]:
    rows = [json.loads(x) for x in open(ROOT / src) if keep(json.loads(x))]
    out = []
    for k in range(n):
        c = f"{G}/cohorts/{name}_s{k}.jsonl"
        (ROOT / c).write_text("".join(json.dumps(r) + "\n" for r in rows[k::n]))
        out.append(c)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    args = ap.parse_args()
    assert (ROOT / G / "protocol_locked.json").exists(), "lock first (gate_build.py --lock)"
    (ROOT / G / "cohorts").mkdir(parents=True, exist_ok=True)
    jobs = []

    def ev(name, c, o, p, s, prio, deps=()):
        jobs.append({"name": name, "prio": prio, "deps": list(deps), "done": f"{o}/rollouts.csv",
                     "cmd": RUN.format(c=c, o=o, p=p, s=s)})

    main_c = shard(f"{G}/mathtest_gate400.jsonl", "gate400", 6)
    for k, c in enumerate(main_c):
        ev(f"gate_eval_s{k}", c, f"{G}/eval/gate400_s{k}", P1, 4, 10)
    for k, c in enumerate(shard("data/policy/math500.jsonl", "math500L13", 2, lambda r: r["level"] <= 3)):
        ev(f"gate_m500_s{k}", c, f"{G}/eval/math500L13_s{k}", EASY, 4, 12)
    for k, c in enumerate(shard("data/distill/eval_gsm8k_test.jsonl", "gsm8k", 2)):
        ev(f"gate_gsm_s{k}", c, f"{G}/eval/gsm8k_s{k}", EASY, 1, 12)
    aime = shard("data/distill/eval_aime_amc.jsonl", "aime", 2)
    for k, c in enumerate(aime):
        ev(f"gate_aime_s{k}", c, f"{G}/eval/aime_s{k}", f"base {N} {N}G10", 8, 11)

    p1 = [f"gate_eval_s{k}" for k in range(6)]
    jobs.append({"name": "gate_calib_targets", "gpu": False, "prio": 5, "deps": p1,
                 "done": f"{G}/calib_targets.json",
                 "cmd": "uv run python scripts/gate_calibrate.py --targets"})
    for k in range(3):
        o = f"{G}/calib/s{k}"
        jobs.append({"name": f"gate_calib_s{k}", "prio": 5, "deps": ["gate_calib_targets"],
                     "done": f"{o}/rollouts.csv",
                     "cmd": "mkdir -p data/gated/calib && uv run python -c \"import json;"
                            "rs=[json.loads(x) for x in open('data/gated/calib50.jsonl')];"
                            f"open('data/gated/calib/c{k}.jsonl','w').write(''.join(json.dumps(r)+chr(10) for r in rs[{k}::3]))\" && "
                            + RUN.format(c=f"{G}/calib/c{k}.jsonl", o=o,
                                         p="$(uv run python scripts/gate_calibrate.py --grid-policies)", s=2)})
    jobs.append({"name": "gate_calib_choose", "gpu": False, "prio": 5,
                 "deps": [f"gate_calib_s{k}" for k in range(3)], "done": f"{G}/calibration.json",
                 "cmd": "uv run python scripts/gate_calibrate.py --choose"})
    for k, c in enumerate(main_c):
        ev(f"gate_evalR_s{k}", c, f"{G}/eval/gate400R_s{k}", RPOL.format(q=""), 4, 6,
           ["gate_calib_choose"])
    for k, c in enumerate(aime):
        ev(f"gate_aimeR_s{k}", c, f"{G}/eval/aimeR_s{k}", RPOL.format(q="10"), 8, 7,
           ["gate_calib_choose"])

    pick = ("import csv,glob,json,random\n"
            "rs=[r for f in sorted(glob.glob('data/gated/eval/gate400*_s*/rollouts.csv')) for r in csv.DictReader(open(f))]\n"
            "c=json.load(open('data/gated/calibration.json'))\n"
            f"arms=['base','{N}','{N}G10',c['10']['policy']]\n"
            "out=[]\n"
            "for p in arms:\n"
            "    xs=sorted([r for r in rs if r['policy']==p],key=lambda r:(r['question_id'],int(r['seed'])))\n"
            "    out+=[{'question_id':r['question_id'],'seed':int(r['seed']),'policy':p} for r in random.Random(f'20261003:mech:{p}').sample(xs,200)]\n"
            "open('data/gated/mech_select.jsonl','w').write(''.join(json.dumps(r)+'\\n' for r in out))\n")
    jobs.append({"name": "gate_mech_select", "gpu": False, "prio": 4,
                 "deps": [f"gate_evalR_s{k}" for k in range(6)], "done": f"{G}/mech_select.jsonl",
                 "cmd": f"uv run python -c \"{pick}\""})
    for k in range(3):
        o = f"{G}/probes/mech_s{k}.jsonl"
        jobs.append({"name": f"gate_mech_probe_s{k}", "prio": 4, "deps": ["gate_mech_select"],
                     "done": f"{o}.DONE",
                     "cmd": f"uv run python scripts/sel_probe.py --journals '{G}/eval/gate400*_s*/rollouts.jsonl' "
                            f"--select {G}/mech_select.jsonl --questions {G}/mathtest_gate400.jsonl "
                            f"--shard {k}/3 --out {o} && touch {o}.DONE"})
    jobs.append({"name": "gate_mech_label", "gpu": False, "prio": 4,
                 "deps": [f"gate_mech_probe_s{k}" for k in range(3)], "done": f"{G}/episodes_mech.jsonl",
                 "cmd": f"uv run python scripts/sel_label.py --probes '{G}/probes/mech_s*.jsonl' "
                        f"--out {G}/episodes_mech.jsonl"})
    jobs.append({"name": "gate_report", "gpu": False, "prio": 4,
                 "deps": ["gate_mech_label", "gate_aimeR_s0", "gate_aimeR_s1", "gate_aime_s0",
                          "gate_aime_s1", "gate_m500_s0", "gate_m500_s1", "gate_gsm_s0", "gate_gsm_s1"],
                 "done": f"{G}/report.json", "cmd": "uv run python scripts/gate_report.py"})
    with args.jobs.open("a") as f:
        f.write("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"appended {len(jobs)} jobs")


if __name__ == "__main__":
    main()
