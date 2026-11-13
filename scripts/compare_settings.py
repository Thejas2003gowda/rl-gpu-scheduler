"""Compare agents trained with one information level against the same agents
trained with another (same algorithm and cluster size), on the same test episodes.

Reads the per_episode.csv files written by scripts/evaluate.py. For each setting
of --a whose name contains _<a-info> (for example dqn_c256_signals), finds the
setting of --b with _<b-info> instead (dqn_c256_estimate) and reports
``gpusched.stats.difference_test``: the difference of the two IQM % changes in
total wait vs first-fit, with a 95% CI that resamples each agent's seeds
separately and the episodes jointly. Negative = the --a agents wait less.

    python scripts/compare_settings.py --a results/rl/C_test --b results/rl/B_test \\
        --a-info signals --b-info estimate --out results/rl/C_signals_vs_estimate
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched.stats import difference_test, success_test  # noqa: E402


def waits(df, label):
    d = df[df["setting"] == label]
    table = d.pivot(index="seed", columns="episode", values="avg_wait_h").sort_index(axis=1)
    starts = d.groupby("episode")["start"].first().sort_index().to_numpy()
    return table.to_numpy(), starts, int(d["capacity"].iloc[0]) if "capacity" in d else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--a", required=True, help="evaluate.py output folder of the new agents")
    ap.add_argument("--b", required=True, help="evaluate.py output folder of the comparison agents")
    ap.add_argument("--a-info", required=True)
    ap.add_argument("--b-info", required=True)
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    a = pd.read_csv(Path(args.a) / "per_episode.csv")
    b = pd.read_csv(Path(args.b) / "per_episode.csv")
    if "capacity" not in a:
        raise SystemExit(f"{args.a}/per_episode.csv has no capacity column (re-run scripts/evaluate.py)")
    base = a[a["setting"] == "baseline"]

    report, lines = {}, []
    for label in sorted(set(a["setting"]) - {"baseline"}):
        if f"_{args.a_info}" not in label:
            continue
        other = label.replace(f"_{args.a_info}", f"_{args.b_info}")
        if other not in set(b["setting"]):
            print(f"skip {label}: no {other} in {args.b}")
            continue
        wa, starts_a, cap = waits(a, label)
        wb, starts_b, _ = waits(b, other)
        if not np.array_equal(starts_a, starts_b):
            raise SystemExit(f"{label} and {other} were not evaluated on the same episodes")
        ff = base[(base["policy"] == "fifo_ff") & (base["capacity"] == cap)].sort_values("episode")
        if not np.array_equal(ff["start"].to_numpy(), starts_a):
            raise SystemExit(f"first-fit episodes do not match {label}")
        ref = ff["avg_wait_h"].to_numpy()
        diff = difference_test(wa, wb, ref, n_boot=args.n_boot)
        sa, sb = success_test(wa, ref, n_boot=args.n_boot), success_test(wb, ref, n_boot=args.n_boot)
        report[label] = {"compared_with": other, "capacity": cap, "difference": diff,
                         args.a_info: sa, args.b_info: sb}
        lines.append(f"| {label} vs {other} | {sa['iqm_change_pct']:+.1f}% [{sa['ci_lo']:+.1f}, {sa['ci_hi']:+.1f}] "
                     f"| {sb['iqm_change_pct']:+.1f}% [{sb['ci_lo']:+.1f}, {sb['ci_hi']:+.1f}] "
                     f"| {diff['difference_pct_points']:+.1f} [{diff['ci_lo']:+.1f}, {diff['ci_hi']:+.1f}] "
                     f"| {'yes' if diff['a_better'] else 'no'} |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    md = [f"{args.a_info} vs {args.b_info}: change in total wait vs first-fit on the same test episodes "
          "(IQM over seeds, 95% CI). Difference = first minus second, in percentage points "
          "(negative = less waiting); CI resamples each agent's seeds separately and the episodes jointly.",
          "", f"| Agents | {args.a_info} | {args.b_info} | Difference | {args.a_info} better (CI < 0) |",
          "|---|---|---|---|---|", *lines]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
