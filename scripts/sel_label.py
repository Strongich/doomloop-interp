#!/usr/bin/env python3
r"""EXPERIMENT-selective-doubt.md §S3: doubt episodes and their labels, from sel_probe output.

Boundaries b_0..b_{n-1}; doubt[k] = paragraph after b_k is a doubt paragraph.
  episode at k     starts at the doubt paragraph after b_k
  before           probe at b_k (the site N is built / injected at)
  end              b_e with e = min(next doubt index m > k, k + 4); if e >= n -> end of thinking
                   (the paragraphs of the episode are k+1..e)
  sensitivity      after = probe at b_{k+1} (end of the doubt paragraph itself), or end of
                   thinking if k+1 >= n
  labels           before empty -> excluded. An empty `after` is not correct (counted, and
                   reported as after_empty).
      productive wrong->right | redundant right->right same | other right->right different form
      harmful right->wrong    | failed wrong->wrong
  "same" = probe_candidates.same_answer (string, else symbolic).
Covariates (§S4): level, token index b_k, paragraph index k, doubt paragraphs before k,
stability = same_answer(probe b_k, probe b_{k-1}) (0 for k = 0).

    uv run python scripts/sel_label.py --probes 'data/selective/probes/source_s*.jsonl' \
        --out data/selective/episodes_source.jsonl
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import sys
from multiprocessing import Pool
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

LABELS = ("productive", "redundant", "other", "harmful", "failed")


def label(before: dict, after: dict, same_answer) -> str:
    if before["correct"]:
        if not after["correct"]:
            return "harmful"
        return "redundant" if same_answer(before["cand"], after["cand"]) else "other"
    return "productive" if after["correct"] else "failed"


def episodes(line: str) -> list[dict]:
    from probe_candidates import same_answer

    r = json.loads(line)
    bs, doubt, pr = r["boundaries"], r["doubt"], r["probes"]
    end = pr[str(r["end_cut"])]
    at = lambda i: pr[str(bs[i])] if i < len(bs) else end  # noqa: E731
    out = []
    for k, d in enumerate(doubt):
        if not d:
            continue
        before = pr[str(bs[k])]
        nxt = next((m for m in range(k + 1, len(bs)) if doubt[m]), len(bs))
        e = min(nxt, k + 4)
        after, sens = at(e), at(k + 1)
        ep = {"key": r["key"], "question_id": r["question_id"], "seed": r["seed"],
              "policy": r["policy"], "level": r["level"], "k": k, "cut": bs[k],
              "end_index": e if e < len(bs) else -1, "n_paragraphs": e - k,
              "opens": r["opens"][k], "before": before["cand"], "after": after["cand"],
              "before_correct": before["correct"], "after_correct": after["correct"],
              "after_empty": int(not after["cand"]),
              "cov_level": r["level"], "cov_token": bs[k], "cov_paragraph": k,
              "cov_doubt_so_far": sum(doubt[:k]),
              "cov_stable": int(k > 0 and same_answer(before["cand"], pr[str(bs[k - 1])]["cand"]))}
        if not before["cand"]:
            ep["label"] = ep["sens_label"] = "excluded"
        else:
            ep["label"] = label(before, after, same_answer)
            ep["sens_label"] = label(before, sens, same_answer)
        out.append(ep)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probes", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()
    lines = [x for f in sorted(glob.glob(str(ROOT / args.probes))) for x in open(f) if x.strip()]
    with Pool(args.workers) as p:
        eps = [e for es in p.map(episodes, lines, chunksize=4) for e in es]
    args.out.write_text("".join(json.dumps(e) + "\n" for e in eps))

    rollouts = [json.loads(x) for x in lines]
    n_roll = len(rollouts)
    n_doubt = sum(sum(r["doubt"]) for r in rollouts)
    by_lv: dict = collections.defaultdict(collections.Counter)
    for e in eps:
        by_lv[e["level"]][e["label"]] += 1
        by_lv["all"][e["label"]] += 1
    prod = [e for e in eps if e["label"] == "productive"]
    flips = sum(e["label"] != e["sens_label"] for e in eps if e["label"] != "excluded")
    sens = collections.Counter(e["sens_label"] for e in eps)
    per_roll = collections.Counter(e["key"] for e in prod)
    rep = {
        "rollouts": n_roll, "questions": len({r["question_id"] for r in rollouts}),
        "doubt_paragraphs": n_doubt, "episodes": len(eps),
        "doubt_col_agreement": sum(sum(r["doubt"]) == r["doubt_blocks_col"] for r in rollouts),
        "labels_by_level": {str(k): dict(v) for k, v in sorted(by_lv.items(), key=lambda x: str(x[0]))},
        "productive_questions": len({e["question_id"] for e in prod}),
        "productive_per_rollout": round(len(prod) / n_roll, 3),
        "rollouts_with_productive": len(per_roll),
        "productive_share_of_doubt_paragraphs": round(len(prod) / n_doubt, 4),
        "productive_share_of_labeled": round(len(prod) / max(1, sum(
            e["label"] != "excluded" for e in eps)), 4),
        "after_empty": sum(e["after_empty"] for e in eps if e["label"] != "excluded"),
        "sensitivity": {"labels": dict(sens), "flips": flips},
        "opens_variant": dict(collections.Counter(e["label"] for e in eps if e["opens"])),
    }
    (args.out.with_suffix(".report.json")).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
