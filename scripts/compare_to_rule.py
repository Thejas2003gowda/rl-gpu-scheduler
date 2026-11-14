"""Compare each agent setting of an evaluation with one rule-based baseline (for
example SJF with true run times) on the same test episodes.

Reads the per_episode.csv written by scripts/evaluate.py and applies the main
test's statistics with the rule as the reference: IQM over seeds of the % change
in total wait vs the rule, with a 95% CI that resamples seeds and episodes.
Negative = the agents wait less than the rule; "beats the rule" = CI below 0.

    python scripts/compare_to_rule.py --report results/rl/D_test --rule sjf_oracle --out results/rl/D_vs_sjf_true
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched.stats import success_test  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", required=True, help="scripts/evaluate.py output folder")
    ap.add_argument("--rule", default="sjf_oracle", help="baseline policy name, e.g. sjf_oracle, sjf_pred, sgf")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_csv(Path(args.report) / "per_episode.csv")
    if "capacity" not in df:
        raise SystemExit(f"{args.report}/per_episode.csv has no capacity column (re-run scripts/evaluate.py)")
    base = df[df["setting"] == "baseline"]
    report, lines = {}, []
    for label in sorted(set(df["setting"]) - {"baseline"}):
        d = df[df["setting"] == label]
        cap = int(d["capacity"].iloc[0])
        w = d.pivot(index="seed", columns="episode", values="avg_wait_h").sort_index(axis=1)
        starts = d.groupby("episode")["start"].first().sort_index().to_numpy()
        rule = base[(base["policy"] == args.rule) & (base["capacity"] == cap)].sort_values("episode")
        if rule.empty:
            raise SystemExit(f"no {args.rule} baseline for {cap} GPUs in {args.report}")
        if not np.array_equal(rule["start"].to_numpy(), starts):
            raise SystemExit(f"{label}: episodes differ from the {args.rule} baseline")
        r = success_test(w.to_numpy(), rule["avg_wait_h"].to_numpy(), n_boot=args.n_boot)
        r["beats_rule"] = r.pop("success")
        report[label] = {"capacity": cap, "rule": args.rule, **r}
        per_seed = np.array(r["per_seed_change_pct"])
        lines.append(f"| {label} | {len(per_seed)} | {r['iqm_change_pct']:+.1f}% [{r['ci_lo']:+.1f}, {r['ci_hi']:+.1f}] "
                     f"| {int((per_seed < 0).sum())} of {len(per_seed)} | {'yes' if r['beats_rule'] else 'no'} |")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    md = [f"Agents vs `{args.rule}` on the same test episodes: IQM over seeds of the % change in total "
          "wait (negative = agents wait less), 95% CI resampling seeds and episodes.", "",
          f"| Setting | Seeds | Change vs {args.rule} [95% CI] | Seeds below 0 | Beats the rule (CI < 0) |",
          "|---|---|---|---|---|", *lines]
    (out / "summary.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
