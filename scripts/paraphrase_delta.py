#!/usr/bin/env python3
"""Does Δ encode the MEANING of the edit, or the literal words it inserted?

The confound, raised by arXiv:2605.03907's decomposition argument: Δ is the
difference of two AR forward passes over texts that differ in exactly the tokens
"Wait, let me double-check that." versus "Therefore, the total is calculated as
follows." Part of Δ may therefore encode *those strings* rather than the
model-internal doubt configuration.

The `continue` control does not rule this out — both edits differ precisely in
those tokens, so it shows that an edit of that shape is not sufficient, not that
Δ's lexical component is inert.

The test: rewrite CONTINUE_PARAGRAPH several times with disjoint wording and the
same meaning. Compute Δ for each over the same explanations.

  * Δ's cluster tightly  -> Δ tracks meaning; the confound is answered.
  * Δ's scatter          -> a large share of Δ is lexical, and that belongs in
                            the caveats.

The paraphrase spread is also the noise floor the "per-probe variation is just AR
noise" claim has never had: variation between two ways of saying the same thing
is, by construction, not signal.

No generation — AR forward passes over explanations already on disk.
"""

from __future__ import annotations

import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from steer_demo import (  # noqa: E402
    CONTINUE_PARAGRAPH,
    DOUBT_PARAGRAPH,
    edit_explanation,
    load_ar,
    reconstruct,
)

# Same meaning as CONTINUE_PARAGRAPH — "the next thing is a continuation, not a
# doubt" — with deliberately different vocabulary and sentence shape. None reuses
# "Therefore", "total" or "derivation" from the original.
PARAPHRASES = {
    "original": CONTINUE_PARAGRAPH,
    "p1": (
        "This closes the sentence, so what follows is most likely a fresh clause "
        'moving the work forward, something like "Adding these together gives the '
        'result." to carry on.'
    ),
    "p2": (
        "The thought is complete here, and the next tokens probably begin the "
        'following step of the argument, for instance "Substituting that value '
        'back in yields the answer." as the work proceeds.'
    ),
    "p3": (
        "Having finished this point, the model is most likely about to open a new "
        'statement that advances the solution, e.g. "The remaining quantity can '
        'now be computed directly." and keep going.'
    ),
    "p4": (
        "This wraps up the current clause; what comes next is very likely a "
        'continuation of the calculation, such as "Multiplying both sides now '
        'gives the figure we need." rather than a pause.'
    ),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--explanations", type=Path, default=Path("data/block_explanations_3way.csv"))
    p.add_argument("--ar", type=Path, default=Path("checkpoints/ar_rl_ep1"))
    p.add_argument("--kind", default="doubt_wait")
    p.add_argument("--limit", type=int, default=297)
    p.add_argument("--out", type=Path, default=Path("data/paraphrase_delta.pt"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    probes = [
        r
        for r in csv.DictReader(args.explanations.open())
        if r["kind"] == args.kind and r["explanation"].count("\n\n") >= 1
    ][: args.limit]
    print(f"{len(probes)} probes, {len(PARAPHRASES)} paraphrases + 1 doubt control\n")

    ar_tok, ar_backbone, affine = load_ar(args.ar)
    variants = {**PARAPHRASES, "DOUBT(opposite)": DOUBT_PARAGRAPH}
    acc: dict[str, list[torch.Tensor]] = {k: [] for k in variants}

    with torch.no_grad():
        for i, r in enumerate(probes, 1):
            e0 = r["explanation"]
            a = reconstruct(ar_tok, ar_backbone, affine, e0)
            for name, para in variants.items():
                try:
                    ed = edit_explanation(e0, para)
                except SystemExit:
                    continue
                acc[name].append((reconstruct(ar_tok, ar_backbone, affine, ed) - a).float().cpu())
            if i % 50 == 0:
                print(f"  [{i}/{len(probes)}]", flush=True)

    units: dict[str, torch.Tensor] = {}
    print("\nper-paraphrase mean direction:")
    for name, vs in acc.items():
        if not vs:
            continue
        stack = torch.stack(vs)
        mean = stack.mean(0)
        units[name] = mean / mean.norm()
        print(f"  {name:<16} n={stack.shape[0]:>4}  mean||Δ||={stack.norm(dim=1).mean():>7.1f}")

    names = [n for n in variants if n in units]
    print("\npairwise cosine between paraphrase directions:")
    print("                " + "".join(f"{n[:9]:>11}" for n in names))
    for a_ in names:
        row = "".join(
            f"{torch.nn.functional.cosine_similarity(units[a_][None], units[b][None]).item():>11.3f}"
            for b in names
        )
        print(f"{a_[:15]:<16}" + row)

    cont = [n for n in names if n != "DOUBT(opposite)"]
    within = [
        torch.nn.functional.cosine_similarity(units[a_][None], units[b][None]).item()
        for a_, b in combinations(cont, 2)
    ]
    across = [
        torch.nn.functional.cosine_similarity(units[c][None], units["DOUBT(opposite)"][None]).item()
        for c in cont
    ]
    print(f"\nwithin paraphrases : mean {sum(within)/len(within):+.3f} "
          f"(min {min(within):+.3f}, max {max(within):+.3f})")
    print(f"continue vs doubt  : mean {sum(across)/len(across):+.3f}")
    print(
        "\nTight within-cluster and a clearly separated doubt direction means Δ "
        "tracks meaning,\nnot the words the template happened to use."
    )
    torch.save({"units": units}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
