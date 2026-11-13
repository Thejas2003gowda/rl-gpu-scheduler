"""Secondary measures on the main evaluation's test episodes: average, median and
99th-percentile job completion time (JCT), average bounded slowdown, and GPU
utilization, for the RL agents and the rule-based baselines.

Agents: from the per_episode.csv written by scripts/evaluate.py (each seed's mean
over the episodes, then the IQM over seeds). Baselines: re-run on the same
episodes (the simulator is deterministic). Changes are vs first-fit, in %;
negative = better, except utilization (absolute, higher = GPUs busier).
Medians and 99th percentiles are per episode, then averaged over episodes.

    python scripts/summarize_secondary.py --report results/rl/B_test --out results/analysis/secondary
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched.baselines import get_policies, run_episode  # noqa: E402
from gpusched.experiment import load_vc_jobs, make_env, nonoverlap_starts  # noqa: E402
from gpusched.metrics import iqm  # noqa: E402

METRICS = ["avg_jct_h", "median_jct_h", "p99_jct_h", "avg_bsld", "utilization"]
RULES = ["fifo_ff", "random", "fifo", "sgf", "sjf", "sjf_pred", "sjf_oracle"]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", default="results/rl/B_test")
    ap.add_argument("--out", default="results/analysis/secondary")
    args = ap.parse_args()

    report = json.loads((Path(args.report) / "report.json").read_text())
    per_ep = pd.read_csv(Path(args.report) / "per_episode.csv")
    by_env = {}
    for label, r in report.items():
        s = r["settings"]
        by_env.setdefault((s["jobs_csv"], s["vc"], s["capacity"]), []).append(label)

    rows, lines = [], []
    for (jobs_csv, vc, cap), labels in sorted(by_env.items(), key=lambda kv: kv[0][2]):
        jobs = load_vc_jobs(jobs_csv, vc)
        env = make_env(jobs, "test", cap, "estimate", 1024)
        starts = nonoverlap_starts(env)
        pol = get_policies(seed=0)
        values = {}
        for r in RULES:
            if r == "sjf_pred" and "pred_duration" not in jobs.columns:
                continue
            m = pd.DataFrame([run_episode(env, pol[r], start=int(s))[0] for s in starts])
            values[r] = {k: float(m[k].mean()) for k in METRICS}
        for label in sorted(labels):
            d = per_ep[per_ep["setting"] == label]
            if sorted(d["start"].unique()) != sorted(int(s) for s in starts):
                raise SystemExit(f"{label}: episodes differ from the non-overlapping test episodes")
            per_seed = d.groupby("seed")[METRICS].mean()
            values[label] = {k: float(iqm(per_seed[k].to_numpy())) for k in METRICS}
        ff = values["fifo_ff"]
        lines += [f"### {cap} GPUs", "",
                  f"First-fit: average JCT {ff['avg_jct_h']:.2f} h, median {ff['median_jct_h']:.2f} h, "
                  f"P99 {ff['p99_jct_h']:.1f} h, bounded slowdown {ff['avg_bsld']:.1f}, "
                  f"utilization {100 * ff['utilization']:.1f}%.", "",
                  "| Policy | Average JCT | Median JCT | P99 JCT | Bounded slowdown | Utilization |",
                  "|---|---|---|---|---|---|"]
        for name, v in values.items():
            ch = {k: 100 * (v[k] / ff[k] - 1) for k in METRICS[:4]}
            rows.append({"capacity": cap, "policy": name, **v, **{f"{k}_change_pct": c for k, c in ch.items()}})
            if name == "fifo_ff":
                continue
            lines.append(f"| {name} | {ch['avg_jct_h']:+.1f}% | {ch['median_jct_h']:+.1f}% | {ch['p99_jct_h']:+.1f}% "
                         f"| {ch['avg_bsld']:+.1f}% | {100 * v['utilization']:.1f}% |")
        lines.append("")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "secondary_metrics.csv", index=False)
    md = ["# Secondary measures (test period, non-overlapping episodes)", "",
          "Change vs first-fit (negative = better); utilization is absolute. Agents: IQM over seeds "
          "of each seed's mean over the episodes. Medians and P99 are per episode, then averaged. "
          "Descriptive only: no criterion is attached to these.", "", *lines]
    (out / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
