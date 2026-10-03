#!/usr/bin/env python3
r"""Append the EXPERIMENT-selective-doubt.md §S6 evaluation + §S7 mechanism-check jobs to the
shared gpu_queue jobs file. Run only after the gate (§S4) passed and the directions exist.

  sel400  8 arms x 4 seeds, 6 question shards (one engine config, same request seeds)
  aime    base, N@a1.0, N_all, N_red x 8 seeds, 2 shards
  mech    200 random rollouts per arm (base, N@a1.0, N_all, N_red; seed 20261002), probed
          and labeled exactly as §S3

    uv run python scripts/sel_queue_eval.py --jobs /workspace/distill_logs/jobs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SEL = "data/selective"
DIRS = "--dir Nall=data/selective/dir_N_all.pt --dir Nred=data/selective/dir_N_red.pt " \
       "--dir Nprod=data/selective/dir_N_prod.pt --dir Nsel=data/selective/dir_N_sel.pt"
MAIN = "base N@a0.5d256 N@a0.75d256 N@a1.0d256 Nall@a1.0d256 Nred@a1.0d256 Nprod@a1.0d256 Nsel@a1.0d256"
AIME = "base N@a1.0d256 Nall@a1.0d256 Nred@a1.0d256"
RUN = ("uv run python scripts/reasoning_policy_vllm.py --cohort {c} --outdir {o} --policies {p} "
       "--seeds {s} --max-new-tokens 32768 --batch 512 --max-num-batched-tokens 8192 "
       "--gpu-memory-utilization 0.90 --checkpoint-every 500 --stream " + DIRS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=Path, required=True)
    args = ap.parse_args()
    (ROOT / SEL / "eval").mkdir(parents=True, exist_ok=True)
    jobs = []
    rows = [json.loads(x) for x in open(ROOT / SEL / "mathtest_sel400.jsonl")]
    for k in range(6):
        c = f"{SEL}/eval/sel400_s{k}.jsonl"
        (ROOT / c).write_text("".join(json.dumps(r) + "\n" for r in rows[k::6]))
        o = f"{SEL}/eval/sel400_s{k}"
        jobs.append({"name": f"sel_eval_s{k}", "prio": 15, "done": f"{o}/rollouts.csv",
                     "cmd": RUN.format(c=c, o=o, p=MAIN, s=4)})
    aime = [json.loads(x) for x in open(ROOT / "data/distill/eval_aime_amc.jsonl")]
    for k in range(2):
        c = f"{SEL}/eval/aime_s{k}.jsonl"
        (ROOT / c).write_text("".join(json.dumps(r) + "\n" for r in aime[k::2]))
        o = f"{SEL}/eval/aime_s{k}"
        jobs.append({"name": f"sel_eval_aime_s{k}", "prio": 16, "done": f"{o}/rollouts.csv",
                     "cmd": RUN.format(c=c, o=o, p=AIME, s=8)})
    pick = ("import csv,glob,json,random\n"
            "rs=[r for f in sorted(glob.glob('data/selective/eval/sel400_s*/rollouts.csv')) for r in csv.DictReader(open(f))]\n"
            "out=[]\n"
            "for p in ['base','N@a1.0d256','Nall@a1.0d256','Nred@a1.0d256']:\n"
            "    xs=sorted([r for r in rs if r['policy']==p],key=lambda r:(r['question_id'],int(r['seed'])))\n"
            "    out+=[{'question_id':r['question_id'],'seed':int(r['seed']),'policy':p} for r in random.Random(f'20261002:mech:{p}').sample(xs,200)]\n"
            "open('data/selective/mech_select.jsonl','w').write(''.join(json.dumps(r)+'\\n' for r in out))\n")
    jobs.append({"name": "sel_mech_select", "gpu": False, "prio": 14,
                 "deps": [f"sel_eval_s{k}" for k in range(6)], "done": f"{SEL}/mech_select.jsonl",
                 "cmd": f"uv run python -c \"{pick}\""})
    for k in range(3):
        o = f"{SEL}/probes/mech_s{k}.jsonl"
        jobs.append({"name": f"sel_mech_probe_s{k}", "prio": 14, "deps": ["sel_mech_select"],
                     "done": f"{o}.DONE",
                     "cmd": f"uv run python scripts/sel_probe.py --journals '{SEL}/eval/sel400_s*/rollouts.jsonl' "
                            f"--select {SEL}/mech_select.jsonl --questions {SEL}/mathtest_sel400.jsonl "
                            f"--shard {k}/3 --out {o} && touch {o}.DONE"})
    jobs.append({"name": "sel_mech_label", "gpu": False, "prio": 14,
                 "deps": [f"sel_mech_probe_s{k}" for k in range(3)], "done": f"{SEL}/episodes_mech.jsonl",
                 "cmd": f"uv run python scripts/sel_label.py --probes '{SEL}/probes/mech_s*.jsonl' "
                        f"--out {SEL}/episodes_mech.jsonl"})
    jobs.append({"name": "sel_report", "gpu": False, "prio": 14,
                 "deps": ["sel_mech_label", "sel_eval_aime_s0", "sel_eval_aime_s1"],
                 "done": f"{SEL}/report.json", "cmd": "uv run python scripts/sel_report.py"})
    with args.jobs.open("a") as f:
        f.write("".join(json.dumps(j) + "\n" for j in jobs))
    print(f"appended {len(jobs)} jobs")


if __name__ == "__main__":
    main()
