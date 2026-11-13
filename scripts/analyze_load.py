"""Exploratory: is the agents' advantage larger when the cluster is busier?

No new test episodes are chosen. This uses the 15 non-overlapping test episodes
of the main evaluation and measures how busy each one is from the trace alone:

    offered load = GPU-time requested by the episode's jobs
                   / (GPUs x time from its first to its last submission)

plus the GPUs held by background jobs when it starts (warm start). It then
relates each policy's per-episode change in waiting vs first-fit to that load
(Spearman correlation) and compares the busier half of the episodes (load above
the median) with the lighter half, using the main test's statistics (IQM over
seeds, 95% CI resampling seeds and the episodes of that half).

    python scripts/analyze_load.py --report results/rl/B_test --out results/analysis/load
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
from gpusched.stats import change_per_seed, success_test  # noqa: E402

RULES = ["sgf", "sjf", "sjf_pred", "sjf_oracle"]
NAMES = {"sgf": "Smallest-GPU-first", "sjf": "SJF (user average)", "sjf_pred": "SJF (predicted)",
         "sjf_oracle": "SJF (true run times)"}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--report", default="results/rl/B_test", help="scripts/evaluate.py output folder")
    ap.add_argument("--settings", nargs="*", default=None,
                    help="agent settings to include (default: all *_estimate settings)")
    ap.add_argument("--n-boot", type=int, default=10_000)
    ap.add_argument("--out", default="results/analysis/load")
    args = ap.parse_args()

    report = json.loads((Path(args.report) / "report.json").read_text())
    per_ep = pd.read_csv(Path(args.report) / "per_episode.csv")
    labels = args.settings or sorted(k for k in report if k.endswith("_estimate"))
    by_env = {}
    for label in labels:
        s = report[label]["settings"]
        by_env.setdefault((s["jobs_csv"], s["vc"], s["capacity"]), []).append(label)

    rows, lines, summary = [], [], {}
    for (jobs_csv, vc, cap), group in sorted(by_env.items(), key=lambda kv: kv[0][2]):
        jobs = load_vc_jobs(jobs_csv, vc)
        env = make_env(jobs, "test", cap, "estimate", 1024)
        starts = nonoverlap_starts(env)
        load, bg = [], []
        for s in starts:
            sl = slice(int(s), int(s) + env.episode_len)  # the jobs the simulator replays
            sub, gpus, dur = env._all_submit[sl], env._all_gpus[sl], env._all_dur[sl]
            load.append(float((gpus * dur).sum() / (cap * (sub[-1] - sub[0]))))
            env.reset(options={"start": int(s)})
            bg.append(int(env.bg_gpus_at_start))
        load = np.array(load)
        pol = get_policies(seed=0)
        base = {r: np.array([run_episode(env, pol[r], start=int(s))[0]["avg_wait_h"] for s in starts])
                for r in ["fifo_ff", *RULES]}
        ref = base["fifo_ff"]
        waits = {}
        for label in group:
            d = per_ep[per_ep["setting"] == label]
            w = d.pivot(index="seed", columns="episode", values="avg_wait_h").sort_index(axis=1).to_numpy()
            if not np.array_equal(d.groupby("episode")["start"].first().sort_index().to_numpy(), starts):
                raise SystemExit(f"{label}: episodes differ from the non-overlapping test episodes")
            # the agents' report must have used the same first-fit waits
            check = report[label]["success_test_vs_first_fit"]["per_seed_change_pct"]
            if not np.allclose(change_per_seed(w, ref), check, atol=1e-6):
                raise SystemExit(f"{label}: first-fit waits do not match the report")
            waits[label] = w
        busy = load > np.median(load)
        halves = {"lighter": ~busy, "busier": busy}

        out = {"capacity": cap, "episodes": len(starts), "offered_load": load.tolist(),
               "bg_gpus_at_start": bg, "median_load": float(np.median(load)), "policies": {}}
        for e, s in enumerate(starts):
            row = {"capacity": cap, "episode": e, "start": int(s), "offered_load": load[e],
                   "bg_gpus_at_start": bg[e], "first_fit_avg_wait_h": ref[e], "half": "busier" if busy[e] else "lighter"}
            for r in RULES:
                row[f"{r}_change_pct"] = 100 * (base[r][e] / ref[e] - 1)
            for label, w in waits.items():
                row[f"{label}_change_pct_mean_over_seeds"] = float(np.mean(100 * (w[:, e] / ref[e] - 1)))
            rows.append(row)

        def rho(per_episode_change):
            return float(pd.Series(load).corr(pd.Series(per_episode_change), method="spearman"))

        table = []
        for label, w in waits.items():
            res = {h: success_test(w[:, m], ref[m], n_boot=args.n_boot) for h, m in halves.items()}
            r = rho(np.mean(100 * (w / ref - 1), axis=0))
            out["policies"][label] = {"spearman_load_vs_change": r,
                                      **{h: {k: res[h][k] for k in ("iqm_change_pct", "ci_lo", "ci_hi")}
                                         for h in halves}}
            table.append(f"| {label} | {res['lighter']['iqm_change_pct']:+.1f}% "
                         f"[{res['lighter']['ci_lo']:+.1f}, {res['lighter']['ci_hi']:+.1f}] "
                         f"| {res['busier']['iqm_change_pct']:+.1f}% "
                         f"[{res['busier']['ci_lo']:+.1f}, {res['busier']['ci_hi']:+.1f}] | {r:+.2f} |")
        for r in RULES:
            ch = {h: float(change_per_seed(base[r][m], ref[m])[0]) for h, m in halves.items()}
            rr = rho(100 * (base[r] / ref - 1))
            out["policies"][r] = {"spearman_load_vs_change": rr, **{h: {"change_pct": ch[h]} for h in halves}}
            table.append(f"| {NAMES[r]} | {ch['lighter']:+.1f}% | {ch['busier']:+.1f}% | {rr:+.2f} |")
        summary[str(cap)] = out
        lines += [f"### {cap} GPUs", "",
                  f"Offered load of the {len(starts)} episodes: {load.min():.2f} to {load.max():.2f} "
                  f"(median {np.median(load):.2f}); lighter half {int((~busy).sum())} episodes, "
                  f"busier half {int(busy.sum())}.", "",
                  "| Policy | Lighter half | Busier half | Spearman: load vs change |",
                  "|---|---|---|---|", *table, ""]

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir / "per_episode_load.csv", index=False)
    (outdir / "report.json").write_text(json.dumps(summary, indent=2))
    md = ["# Exploratory: advantage vs how busy the test episode is", "",
          "Same 15 non-overlapping test episodes as the main test (no new episodes chosen). "
          "Change in total wait vs first-fit within each half (agents: IQM over seeds, 95% CI "
          "resampling seeds and that half's episodes). Negative = less waiting. Spearman: rank "
          "correlation between an episode's offered load and the per-episode change (agents: "
          "averaged over seeds); negative = the policy gains more in busier episodes. "
          "Not part of any pre-registered criterion.", "", *lines]
    (outdir / "summary.md").write_text("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    main()
