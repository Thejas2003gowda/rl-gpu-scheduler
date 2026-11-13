"""Evaluate trained agents on the TEST period and apply the success criterion.

Each --runs folder holds one setting (algorithm, cluster size, information
level) with one sub-folder per seed (seed0/, seed1/, ...), as written by
train.py. All seeds and all rule-based baselines run on the same test episodes.

Reported per setting:
  * success criterion (pre-registered): IQM over seeds of the % change in total
    wait vs first-fit, 95% CI resampling seeds and episodes; success = CI < 0
  * share of the first-fit -> SJF(true run times) gap closed
  * change vs the best realistic rule on the same episodes
  * how often the agent chose WAIT

Example:
    python scripts/evaluate.py --runs runs/ppo_c256_estimate runs/dqn_c256_estimate --out results/rl/c256_estimate

E3 (agents trained on one virtual cluster, tested on another, no retraining):
    python scripts/evaluate.py --runs runs/B/dqn_c256_estimate runs/B/ppo_c256_estimate \
        --jobs-csv data/philly_jobs.csv --vc 11cb48 --capacity 72 --out results/rl/E3_11cb48
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched.baselines import get_policies, run_episode  # noqa: E402
from gpusched.experiment import (CountingPolicy, load_agent, load_vc_jobs, make_env,  # noqa: E402
                                 nonoverlap_starts, overlapping_starts)
from gpusched.stats import change_per_seed, gap_recovered, success_test  # noqa: E402

REALISTIC = ["sgf", "sjf", "sjf_pred", "random"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--which", default="best", choices=["best", "last"])
    ap.add_argument("--overlapping", action="store_true", help="30 overlapping episodes instead (robustness)")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--jobs-csv", default=None, help="test on this job table instead of the training one")
    ap.add_argument("--vc", default=None, help="test on this virtual cluster instead of the training one")
    ap.add_argument("--capacity", type=int, default=None, help="test at this cluster size instead")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    settings = {}
    for run in args.runs:
        seeds = sorted(Path(run).glob("seed*/config.json"))
        if not seeds:
            raise SystemExit(f"no seed*/config.json under {run}")
        cfgs = [json.loads(p.read_text()) for p in seeds]
        key = {k: cfgs[0][k] for k in ("jobs_csv", "vc", "capacity", "duration_info", "no_wait")}
        for c in cfgs:
            if any(c[k] != v for k, v in key.items()):
                raise SystemExit(f"seeds in {run} do not share the same settings")
        for name in ("jobs_csv", "vc", "capacity"):  # E3: evaluate elsewhere than trained
            if getattr(args, name) is not None:
                key[name] = getattr(args, name)
        settings[Path(run).name] = (key, [p.parent for p in seeds])

    # all settings evaluated together must share the test environment's data and cluster
    envs = {}
    rows, report = [], {}
    for label, (key, seed_dirs) in settings.items():
        env_key = (key["jobs_csv"], key["vc"], key["capacity"])
        if env_key not in envs:
            jobs = load_vc_jobs(key["jobs_csv"], key["vc"])
            env = make_env(jobs, "test", key["capacity"], "estimate", 1024)
            starts = overlapping_starts(env) if args.overlapping else nonoverlap_starts(env)
            base = {}
            for name, pol in get_policies(seed=0).items():
                if name == "sjf_pred" and "pred_duration" not in jobs.columns:
                    continue
                base[name] = np.array([run_episode(env, pol, start=int(s))[0]["avg_wait_h"] for s in starts])
                rows += [{"setting": "baseline", "policy": name, "capacity": key["capacity"], "seed": -1,
                          "episode": e, "start": int(s), "avg_wait_h": w}
                         for e, (s, w) in enumerate(zip(starts, base[name]))]
            envs[env_key] = (jobs, starts, base)
        jobs, starts, base = envs[env_key]
        env = make_env(jobs, "test", key["capacity"], key["duration_info"], 1024, no_wait=key["no_wait"])

        waits, wait_share = [], []
        for d in seed_dirs:
            ckpt = next(iter(sorted(d.glob(f"{args.which}.*"))), None)
            if ckpt is None:
                raise SystemExit(f"missing {args.which} checkpoint in {d}")
            pol = CountingPolicy(load_agent(ckpt))
            w = []
            for e, s in enumerate(starts):
                m, _ = run_episode(env, pol, start=int(s))
                w.append(m["avg_wait_h"])
                rows.append({"setting": label, "policy": label, "capacity": key["capacity"], "seed": d.name,
                             "episode": e, "start": int(s), **m})
            waits.append(w)
            wait_share.append(pol.waits / max(pol.steps, 1))
        waits = np.array(waits)
        ref, oracle = base["fifo_ff"], base["sjf_oracle"]
        best_rule = min((r for r in REALISTIC if r in base), key=lambda r: base[r].sum())
        report[label] = {
            "settings": key, "checkpoint": args.which, "episodes": len(starts),
            "success_test_vs_first_fit": success_test(waits, ref, n_boot=args.n_boot),
            "gap_closed_to_sjf_true": gap_recovered(waits, ref, oracle, n_boot=args.n_boot),
            "best_realistic_rule": best_rule,
            "change_vs_best_rule_pct_per_seed": change_per_seed(waits, base[best_rule]).tolist(),
            "baseline_change_vs_first_fit_pct": {r: float(change_per_seed(base[r], ref)[0]) for r in base},
            "wait_action_share_per_seed": wait_share,
        }

    pd.DataFrame(rows).to_csv(out / "per_episode.csv", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=2))
    lines = [f"Test period, {'30 overlapping' if args.overlapping else 'non-overlapping'} 1,024-job episodes, "
             f"{args.which} checkpoints. Negative = less waiting than first-fit.", "",
             "| Setting | Seeds | IQM change vs first-fit [95% CI] | Success? | Gap to SJF(true) closed "
             "| vs best realistic rule (median seed) | WAIT share |", "|---|---|---|---|---|---|---|"]
    for label, r in report.items():
        s, g = r["success_test_vs_first_fit"], r["gap_closed_to_sjf_true"]
        lines.append(f"| {label} | {s['n_seeds']} | {s['iqm_change_pct']:+.1f}% [{s['ci_lo']:+.1f}, {s['ci_hi']:+.1f}] "
                     f"| {'yes' if s['success'] else 'no'} | {100 * g['share']:.0f}% [{100 * g['ci_lo']:.0f}, "
                     f"{100 * g['ci_hi']:.0f}] | {np.median(r['change_vs_best_rule_pct_per_seed']):+.1f}% "
                     f"vs {r['best_realistic_rule']} | {np.mean(r['wait_action_share_per_seed']):.3f} |")
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
