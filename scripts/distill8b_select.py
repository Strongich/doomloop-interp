#!/usr/bin/env python3
r"""PROTOCOL v4 §M5 LR selection from the 20-step proxies (natural arm of each method).

  rule  lowest held-out loss at step 20 (ties -> lower LR); a winner at the grid edge is
        flagged. NEW flat-curve rule: if the five step-20 losses span < 1% of their mean,
        the proxy is uninformative and the §L6 centre LR is used. Recorded either way.

    uv run python scripts/distill8b_select.py --method sft
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "data/distill_8b"
CENTRE = {"sft": 1e-4, "dpo": 5e-6}
MULT = (0.25, 0.5, 1.0, 2.0, 4.0)
ARM = {"sft": "short", "dpo": "natural"}


def sweep_name(m: str, x: float) -> str:
    return f"sweep_{m}_{ARM[m]}_lora_lr{CENTRE[m] * x:g}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", choices=["sft", "dpo"], required=True)
    m = ap.parse_args().method
    rows = []
    for x in MULT:
        r = json.loads((D / "models" / sweep_name(m, x) / "sweep_result.json").read_text())
        rows.append({"lr": r["lr"], "final_heldout_loss": r["final_heldout_loss"], "heldout": r["heldout"]})
    losses = [r["final_heldout_loss"] for r in rows]
    mean = sum(losses) / len(losses)
    span = (max(losses) - min(losses)) / mean
    win = min(rows, key=lambda r: (r["final_heldout_loss"], r["lr"]))
    flat = span < 0.01
    lr = CENTRE[m] if flat else win["lr"]
    res = {"method": m, "lr": lr, "proxy_winner": win["lr"], "span_rel": round(span, 5),
           "flat_rule_applied": flat,
           "at_grid_edge": (not flat) and win["lr"] in (rows[0]["lr"], rows[-1]["lr"]),
           "candidates": rows}
    (D / "sweep").mkdir(parents=True, exist_ok=True)
    (D / "sweep" / f"{m}.json").write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: v for k, v in res.items() if k != "candidates"}),
          "; ".join(f"{r['lr']:g}: {r['final_heldout_loss']:.5f}" for r in rows))


if __name__ == "__main__":
    main()
