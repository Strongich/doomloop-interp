r"""Sample size for a non-inferiority claim, from observed paired variation.

The cost claim and the accuracy claim are not equally expensive. Demonstrating a
DIFFERENCE in tokens is cheap: the effect is huge relative to its spread.
Demonstrating that accuracy did NOT fall is a null claim, and a null claim is
bought with sample size alone.

The criterion: the lower bound of the two-sided 95% interval on the paired
difference must sit above `-margin`. With per-question paired sd `s`, that needs

    observed_mean > -margin + 1.96 * s / sqrt(n)

so power depends on where the TRUE effect sits, not only on n. This is the trap
worth naming: a policy that genuinely costs 1 point is *inside* a 2-point margin
and still nearly impossible to certify, because the interval has to clear the
margin from wherever the truth actually is. Reporting power only under "true
effect = 0" therefore flatters the design, which is why `--effects` sweeps a
range and the zero column is never quoted alone.

Variance must come from the SAME generation regime the claim will be made in.
Prefix-continuation and whole-question generation are different estimands with
different spreads, so `--run` should point at whole-question development output
once it exists.

    uv run python scripts/power_analysis.py --run data/reasoning_policy_v1/stage2 --arm N@a1.0d512
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import statistics as st
from pathlib import Path


def phi(z: float) -> float:
    """Standard normal CDF."""
    return 0.5 * math.erfc(-z / math.sqrt(2))


def paired_sd(run: Path, arm: str, base: str, field: str) -> tuple[float, float, int]:
    """Per-question paired sd, mean, n -- seeds averaged within question first."""
    vals: dict[str, dict[str, list[float]]] = collections.defaultdict(
        lambda: collections.defaultdict(list)
    )
    # The whole-question runner writes rollouts.csv keyed by `policy`; the older
    # prefix runner writes branches.csv keyed by `arm`. Read either, so a power
    # estimate can be sanity-checked against historical output -- while noting
    # that the two are different estimands and only the former is quotable here.
    path = run / "rollouts.csv"
    key = "policy"
    if not path.exists():
        path, key = run / "branches.csv", "arm"
    with path.open() as f:
        for r in csv.DictReader(f):
            vals[r["question_id"]][r[key]].append(float(r[field]))
    diffs = [
        st.mean(v[arm]) - st.mean(v[base]) for v in vals.values() if arm in v and base in v
    ]
    if len(diffs) < 3:
        raise ValueError(f"Only {len(diffs)} paired questions for {arm} vs {base}")
    return st.stdev(diffs), st.mean(diffs), len(diffs)


def power_at(n: int, sd: float, margin: float, true_effect: float) -> float:
    """P(lower 95% bound > -margin) given the true effect, in proportion units."""
    se = sd / math.sqrt(n)
    threshold = -margin + 1.96 * se
    return 1.0 - phi((threshold - true_effect) / se)


def n_for(power: float, sd: float, margin: float, true_effect: float) -> float:
    """Smallest n reaching `power`; inf when the true effect is at/past the margin."""
    slack = margin + true_effect  # distance from the truth to the margin
    if slack <= 0:
        return float("inf")
    return ((1.96 + _z(power)) * sd / slack) ** 2


def _z(p: float) -> float:
    """Inverse normal via bisection -- avoids a scipy dependency for one number."""
    lo, hi = -10.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if phi(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--arm", required=True, help="the selected policy")
    ap.add_argument("--base", default="base")
    ap.add_argument("--field", default="correct")
    ap.add_argument("--margin", type=float, default=2.0, help="percentage points")
    ap.add_argument("--pool", type=int, default=997, help="available fresh questions")
    ap.add_argument("--effects", type=float, nargs="+", default=[0.0, -0.25, -0.5, -1.0, -1.5])
    args = ap.parse_args()

    sd, obs, n = paired_sd(args.run, args.arm, args.base, args.field)
    se = sd / math.sqrt(n)
    print(f"{args.run}  {args.arm} vs {args.base}  field={args.field}")
    print(f"  observed: {100 * obs:+.2f} pp over n={n} questions")
    print(f"  paired sd = {sd:.4f}, SE = {100 * se:.2f} pp, "
          f"95% CI [{100 * (obs - 1.96 * se):+.2f}, {100 * (obs + 1.96 * se):+.2f}]")
    print(f"\nNon-inferiority margin: -{args.margin:.1f} pp. "
          f"Available fresh pool: {args.pool}\n")

    m = args.margin / 100
    print(f"{'true effect':>12s}{'power @ pool':>14s}{'n for 80%':>12s}{'n for 90%':>12s}")
    for eff in args.effects:
        e = eff / 100
        p = power_at(args.pool, sd, m, e)
        n80 = n_for(0.80, sd, m, e)
        n90 = n_for(0.90, sd, m, e)
        f80 = "infeasible" if n80 == float("inf") else f"{n80:,.0f}"
        f90 = "infeasible" if n90 == float("inf") else f"{n90:,.0f}"
        print(f"{eff:>+11.2f} pp{100 * p:>13.0f}%{f80:>12s}{f90:>12s}")
    print("\nPower at the true effect of 0 alone overstates the design; read the row")
    print("matching the effect you actually expect. 'infeasible' means the true")
    print("effect is at or beyond the margin, so no sample size can certify it.")


if __name__ == "__main__":
    main()
