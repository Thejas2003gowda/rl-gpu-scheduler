"""Evaluate the rule-based baselines on test-period episodes.

Every policy runs on the SAME job sequences (paired comparison), so differences
come from the policy, not from luck in which jobs were sampled.

Main metric: average job completion time (JCT). In this simulator run times
never change, so average JCT = average wait + a constant, and both rank
policies identically. Bounded slowdown and utilization are also reported.
Test episodes default to 1,024 jobs (RLScheduler evaluated on 1,024-job
sequences; RL training in Phase 2 uses 256-job episodes).

Examples:
    python scripts/run_baselines.py --data synthetic
    python scripts/run_baselines.py --data philly --jobs-csv data/philly_jobs.csv --vc largest --load 0.9
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched import (GPUSchedEnv, add_duration_estimates, bootstrap_ci, capacity_for_load,  # noqa: E402
                      iqm, make_synthetic_jobs, offered_load, paired_bootstrap, time_split)
from gpusched.baselines import get_policies, run_episode  # noqa: E402

POLICY_ORDER = ["random", "fifo", "fifo_ff", "sgf", "sjf", "sjf_oracle"]
LABELS = {"random": "Random", "fifo": "FIFO (strict)", "fifo_ff": "FIFO first-fit",
          "sgf": "Smallest-GPU-first", "sjf": "SJF (user avg)", "sjf_oracle": "SJF (oracle)"}
MAIN = "avg_jct_h"
METRICS = ["avg_jct_h", "avg_wait_h", "median_jct_h", "p99_jct_h", "avg_bsld", "utilization"]


def load_jobs(args):
    if args.data == "synthetic":
        df = make_synthetic_jobs(n_jobs=args.synthetic_jobs, capacity=args.synthetic_capacity,
                                 target_load=args.synthetic_load, seed=args.seed)
        return add_duration_estimates(df), "synthetic"
    df = pd.read_csv(args.jobs_csv)
    if args.vc == "largest":
        vc = df["vc"].value_counts().index[0]
    else:
        vc = args.vc
    if vc != "all":
        df = df[df["vc"] == vc]
    return df.reset_index(drop=True), f"philly vc={vc}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", choices=["synthetic", "philly"], default="synthetic")
    ap.add_argument("--jobs-csv", default="data/philly_jobs.csv")
    ap.add_argument("--vc", default="largest", help="virtual cluster id, 'largest', or 'all'")
    ap.add_argument("--capacity", type=int, default=None, help="cluster GPUs (overrides --load)")
    ap.add_argument("--load", type=float, default=0.9, help="target offered load used to size the cluster")
    ap.add_argument("--reference", default="fifo_ff", choices=POLICY_ORDER,
                    help="policy the paired comparison is measured against")
    ap.add_argument("--non-overlapping", action="store_true",
                    help="use back-to-back, non-overlapping test episodes (ignores --episodes)")
    ap.add_argument("--warm-start", action="store_true",
                    help="start each episode with the jobs still running at that time (per the trace)")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--episode-len", type=int, default=1024, help="jobs per TEST episode")
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--synthetic-jobs", type=int, default=20_000)
    ap.add_argument("--synthetic-capacity", type=int, default=64)
    ap.add_argument("--synthetic-load", type=float, default=0.9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    jobs, source = load_jobs(args)
    _, test = time_split(jobs, args.train_frac)
    if args.capacity is not None:
        capacity = args.capacity
    elif args.data == "synthetic":
        capacity = args.synthetic_capacity
    else:
        capacity = capacity_for_load(jobs, args.load)
    env = GPUSchedEnv(test, capacity=capacity, window=args.window, episode_len=args.episode_len,
                      warm_start=args.warm_start, history=jobs)

    out = Path(args.out or f"results/baselines_{args.data}")
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    if args.non_overlapping:
        starts = np.arange(0, env.n_jobs_total - args.episode_len + 1, args.episode_len)
        args.episodes = len(starts)
    else:
        starts = rng.integers(0, env.n_jobs_total - args.episode_len + 1, size=args.episodes)

    policies = get_policies(args.seed)
    rows, jct = [], {p: [] for p in POLICY_ORDER}
    for ep, start in enumerate(starts):
        for name in POLICY_ORDER:
            m, per_job = run_episode(env, policies[name], start=int(start))
            rows.append({"policy": name, "episode": ep, "start": int(start), **m,
                         "bg_gpus_at_start": env.bg_gpus_at_start,
                         "bg_overflow_jobs": env.bg_overflow_jobs})
            jct[name].append(per_job["jct_h"])
    per_ep = pd.DataFrame(rows)
    per_ep.to_csv(out / "per_episode.csv", index=False)

    # Summary: mean with 95% bootstrap CI, IQM, and paired ratio vs SJF (per episode).
    sjf_main = per_ep[per_ep.policy == "sjf"].set_index("episode")[MAIN]
    summary = []
    for name in POLICY_ORDER:
        d = per_ep[per_ep.policy == name]
        row = {"policy": name}
        for metric in METRICS:
            vals = d[metric].to_numpy()
            lo, hi = bootstrap_ci(vals, np.mean, seed=args.seed)
            row[f"{metric}_mean"], row[f"{metric}_ci_lo"], row[f"{metric}_ci_hi"] = vals.mean(), lo, hi
            row[f"{metric}_iqm"] = iqm(vals)
        ratio = d.set_index("episode")[MAIN] / sjf_main
        row["jct_ratio_vs_sjf_median"] = float(ratio.median())
        row["episodes_better_than_sjf"] = float((ratio < 0.999).mean())
        summary.append(row)
    summary = pd.DataFrame(summary)
    summary.to_csv(out / "summary.csv", index=False)

    config = {
        "source": source, "capacity_gpus": capacity, "offered_load_test": offered_load(test, capacity),
        "jobs_total": len(jobs), "jobs_test": len(test), "jobs_dropped_too_big": env.n_dropped_too_big,
        "episodes": args.episodes, "episode_len": args.episode_len, "window": args.window,
        "train_frac": args.train_frac, "seed": args.seed, "warm_start": args.warm_start,
        "non_overlapping": args.non_overlapping, "reference": args.reference,
        "bg_gpus_at_start_mean": float(per_ep["bg_gpus_at_start"].mean()),
        "bg_overflow_jobs_mean": float(per_ep["bg_overflow_jobs"].mean()),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))

    # Markdown table for the README / report.
    warm = (f"warm start ON (avg {config['bg_gpus_at_start_mean']:.0f} GPUs busy at episode start)"
            if args.warm_start else "warm start OFF")
    lines = [f"Source: {source} | capacity {capacity} GPUs | test offered load "
             f"{config['offered_load_test']:.2f} | {warm} | {args.episodes} episodes x {args.episode_len} jobs",
             "",
             "| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) "
             "| Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        lines.append(
            f"| {LABELS[r.policy]} | {r.avg_jct_h_mean:.2f} [{r.avg_jct_h_ci_lo:.2f}, {r.avg_jct_h_ci_hi:.2f}] "
            f"| {r.avg_jct_h_iqm:.2f} | {r.avg_wait_h_mean:.2f} | {r.median_jct_h_mean:.2f} "
            f"| {r.p99_jct_h_mean:.1f} | {r.avg_bsld_mean:.2f} | {r.utilization_mean:.2f} "
            f"| {r.jct_ratio_vs_sjf_median:.3f} |")
    table = "\n".join(lines)
    (out / "summary.md").write_text(table + "\n")
    print(table)

    # Paired comparison: each policy vs the reference on the SAME episodes (average wait).
    ref = per_ep[per_ep.policy == args.reference].sort_values("episode")["avg_wait_h"].to_numpy()
    paired_rows = []
    for name in POLICY_ORDER:
        if name == args.reference:
            continue
        pol = per_ep[per_ep.policy == name].sort_values("episode")["avg_wait_h"].to_numpy()
        r = paired_bootstrap(pol, ref, seed=args.seed)
        r["policy"] = name
        r["significant"] = bool(r["diff_ci_hi"] < 0 or r["diff_ci_lo"] > 0)
        paired_rows.append(r)
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(out / "paired.csv", index=False)
    plines = [f"Paired comparison vs {LABELS[args.reference]} on the same {args.episodes} episodes "
              f"(average wait; negative = less waiting)", "",
              "| Policy | Wait difference (h): mean [95% CI] | Change in total wait: % [95% CI] "
              "| Episodes with less wait | Clear difference? |",
              "|---|---|---|---|---|"]
    for _, r in paired.iterrows():
        plines.append(
            f"| {LABELS[r['policy']]} | {r['mean_diff']:+.2f} [{r['diff_ci_lo']:+.2f}, {r['diff_ci_hi']:+.2f}] "
            f"| {r['pct_change']:+.1f}% [{r['pct_ci_lo']:+.1f}, {r['pct_ci_hi']:+.1f}] "
            f"| {100 * r['frac_better']:.0f}% | {'yes' if r['significant'] else 'no (CI includes 0)'} |")
    ptable = "\n".join(plines)
    (out / "paired.md").write_text(ptable + "\n")
    print("\n" + ptable)

    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    ys = np.arange(len(paired))
    ax.errorbar(paired["pct_change"], ys,
                xerr=[paired["pct_change"] - paired["pct_ci_lo"], paired["pct_ci_hi"] - paired["pct_change"]],
                fmt="o", capsize=4, color="#4C72B0")
    ax.axvline(0, color="grey", lw=1, ls="--")
    ax.set_yticks(ys, [LABELS[p] for p in paired["policy"]])
    ax.set_xlabel(f"Change in total wait vs {LABELS[args.reference]} (%, 95% CI)")
    ax.set_title("Paired comparison (left of 0 = less waiting)", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / "paired_vs_reference.png", dpi=150)
    plt.close(fig)

    # Figure 1: IQM of average JCT with CI, and spread across episodes.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    names = [LABELS[p] for p in POLICY_ORDER]
    vals = {p: per_ep[per_ep.policy == p][MAIN].to_numpy() for p in POLICY_ORDER}
    iqms = [iqm(vals[p]) for p in POLICY_ORDER]
    cis = [bootstrap_ci(vals[p], iqm, seed=args.seed) for p in POLICY_ORDER]
    err = np.array([[m - lo for m, (lo, _) in zip(iqms, cis)], [hi - m for m, (_, hi) in zip(iqms, cis)]])
    axes[0].bar(names, iqms, yerr=err, capsize=4, color="#4C72B0")
    axes[0].set_ylabel("Average JCT, hours (IQM across episodes)")
    axes[0].set_title("Lower is better (95% bootstrap CI)")
    axes[1].boxplot([vals[p] for p in POLICY_ORDER], showfliers=True)
    axes[1].set_xticks(range(1, len(names) + 1), names)
    axes[1].set_ylabel("Average JCT per episode, hours")
    axes[1].set_title("Spread across episodes")
    for ax in axes:
        ax.tick_params(axis="x", rotation=30)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
    fig.suptitle(f"Baselines on {source} (test period, {args.episode_len}-job episodes)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "baselines_jct.png", dpi=150)
    plt.close(fig)

    # Figure 2: distribution of job completion times.
    fig, ax = plt.subplots(figsize=(6, 4))
    for p in POLICY_ORDER:
        x = np.sort(np.concatenate(jct[p]))
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), label=LABELS[p])
    ax.set_xscale("log")
    ax.set_xlabel("Job completion time (hours, log scale)")
    ax.set_ylabel("Fraction of jobs")
    ax.set_title("JCT distribution (all test episodes)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "jct_cdf.png", dpi=150)
    plt.close(fig)
    print(f"\nSaved results to {out}/")


if __name__ == "__main__":
    main()
