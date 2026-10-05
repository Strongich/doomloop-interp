#!/usr/bin/env python3
r"""PROTOCOL v4 (8B) jobs for scripts/gpu_queue.py (appended to the jobs file).

  --phase A  (after generation) teacher report; Base eval; online teacher reference
             (base + NU26@a0.75d256 through the steering runner on mathfresh + gate400)
  --phase B  (after the teacher report is reviewed and any §M11 entry written)
             sets -> trainer smoke -> 10 LR proxies -> select (flat-curve rule) ->
             Short-SFT, Steered-SFT, DPO-natural, DPO-steered (LoRA) -> evals -> report

Every model's eval is split into two jobs over disjoint sets (same engine and seeds):
  a = mathfresh, aime_amc, val_math      b = mathgate, math500, gsm8k_test

    uv run python scripts/distill8b_queue.py --phase A --jobs /workspace/distill_logs/d8b_jobs.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = "data/distill_8b"
M8 = "Qwen/Qwen3-8B"
GEN = [f"d8b_gen_s{k}" for k in range(6)]
SPLIT = {"a": "mathfresh aime_amc val_math", "b": "mathgate math500 gsm8k_test"}
CENTRE = {"sft": 1e-4, "dpo": 5e-6}
MULT = (0.25, 0.5, 1.0, 2.0, 4.0)
NATURAL = {"sft": "short", "dpo": "natural"}
HELD = {"sft": 256, "dpo": 128}
# model name -> (method, set arm)
MODELS = {"sft_short_lora": ("sft", "short"), "sft_steered_lora": ("sft", "steered_short"),
          "dpo_natural_lora": ("dpo", "natural"), "dpo_steered_lora": ("dpo", "steered")}


def evals(name: str, model: str, deps: list[str], prio: int) -> list[dict]:
    return [{"name": f"d8b_eval_{name}_{p}", "prio": prio, "deps": deps,
             "done": f"{D}/eval/{name}__{p}/rollouts.csv",
             "cmd": f"uv run python scripts/distill_eval.py --model {model} --name {name}__{p} "
                    f"--outroot {D}/eval --sets {sets}"} for p, sets in SPLIT.items()]


def train(name: str, method: str, arm: str, extra: str) -> str:
    return (f"uv run python scripts/distill_train.py --model {M8} --method {method} --arm {arm} "
            f"--peft lora --sets {D}/sets --out {D}/models --name {name} {extra}")


def phase_a() -> list[dict]:
    J = [{"name": "d8b_teacher_report", "gpu": False, "prio": 2, "deps": GEN,
          "done": f"{D}/teacher_report.json",
          "cmd": "uv run python scripts/distill8b_teacher_report.py"}]
    J += evals("base", M8, GEN, 30)
    rows = [json.loads(x) for f in ("data/distill_v3/mathtest_fresh1000.jsonl",
                                    "data/gated/mathtest_gate400.jsonl") for x in open(ROOT / f)]
    (ROOT / D / "online").mkdir(parents=True, exist_ok=True)
    for k in range(4):
        c = f"{D}/online/cohort_s{k}.jsonl"
        (ROOT / c).write_text("".join(json.dumps(r) + "\n" for r in rows[k::4]))
        o = f"{D}/online/s{k}"
        J.append({"name": f"d8b_online_s{k}", "prio": 31, "deps": GEN, "done": f"{o}/rollouts.csv",
                  "cmd": f"uv run python scripts/reasoning_policy_vllm.py --model {M8} --layer 26 "
                         f"--dir NU26=data/xfer8b/dirs_clean/L26_NU.pt --cohort {c} --outdir {o} "
                         f"--policies base NU26@a0.75d256 --seeds 4 --max-new-tokens 32768 "
                         f"--batch 512 --max-num-batched-tokens 8192 --gpu-memory-utilization 0.90 "
                         f"--checkpoint-every 500 --stream"})
    return J


def phase_b() -> list[dict]:
    J = [{"name": "d8b_sets", "gpu": False, "prio": 2, "deps": ["d8b_teacher_report"],
          "done": f"{D}/sets/sets_stats.json",
          "cmd": f"uv run python scripts/distill_build_sets.py --runs '{D}/gen/shard*' "
                 f"--pool data/distill/math_train.jsonl --out {D}/sets --model {M8} "
                 f"--steer NU26@a0.75d256"}]
    J.append({"name": "d8b_smoke", "prio": 2, "deps": ["d8b_sets"], "done": f"{D}/models/smoke_dpo/DONE",
              "cmd": train("smoke_sft", "sft", "short", "--lr 1e-4 --limit 96 --max-steps 1 --heldout 16 "
                           "--no-wandb") + " && " +
                     train("smoke_dpo", "dpo", "natural", "--lr 5e-6 --limit 48 --max-steps 1 --heldout 8 "
                           "--no-wandb")})
    for m in ("sft", "dpo"):
        names = []
        for x in MULT:
            n = f"sweep_{m}_{NATURAL[m]}_lora_lr{CENTRE[m] * x:g}"
            names.append(f"d8b_{n}")
            J.append({"name": f"d8b_{n}", "prio": 3, "deps": ["d8b_smoke"],
                      "done": f"{D}/models/{n}/DONE",
                      "cmd": train(n, m, NATURAL[m], f"--lr {CENTRE[m] * x:g} --max-steps 20 "
                                   f"--heldout {HELD[m]}")})
        J.append({"name": f"d8b_select_{m}", "gpu": False, "prio": 3, "deps": names,
                  "done": f"{D}/sweep/{m}.json",
                  "cmd": f"uv run python scripts/distill8b_select.py --method {m}"})
    for name, (m, arm) in MODELS.items():
        J.append({"name": f"d8b_train_{name}", "prio": 4, "deps": [f"d8b_select_{m}"],
                  "done": f"{D}/models/{name}/DONE",
                  "cmd": train(name, m, arm, f"--lr-from {D}/sweep/{m}.json")})
        J += evals(name, f"{D}/models/{name}", [f"d8b_train_{name}"], 5)
    J.append({"name": "d8b_report", "gpu": False, "prio": 6,
              "deps": [f"d8b_eval_{n}_{p}" for n in ["base", *MODELS] for p in SPLIT]
              + [f"d8b_online_s{k}" for k in range(4)],
              "done": f"{D}/report.json", "cmd": "uv run python scripts/distill8b_report.py"})
    return J


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B"], required=True)
    ap.add_argument("--jobs", type=Path, required=True)
    args = ap.parse_args()
    J = phase_a() if args.phase == "A" else phase_b()
    have = {json.loads(x)["name"] for x in args.jobs.read_text().splitlines() if x.strip()}
    J = [j for j in J if j["name"] not in have]
    with args.jobs.open("a") as f:
        f.write("".join(json.dumps(j) + "\n" for j in J))
    print(f"appended {len(J)} jobs")


if __name__ == "__main__":
    main()
