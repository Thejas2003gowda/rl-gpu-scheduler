"""Reality check: compare simulated waits with the waits recorded in the trace.

For every job kept by ``parse_philly_job_log``, the real wait is
    first attempt's start time - submit time
(the queueing delay before the job first got its GPUs). The script also
measures how many GPUs one virtual cluster really had in use over time, from
the real start and end time of every attempt. That is a lower bound on the
cluster's real size.

It then runs first-fit in the simulator on the SAME test jobs (non-overlapping
1,024-job episodes, warm start) at several cluster sizes and puts the real and
simulated wait distributions side by side. Finally it reports the offered load
of the training, validation and test periods at each size.

Usage:
    python scripts/check_trace_waits.py --job-log philly-traces/trace-data/cluster_job_log
"""
import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched import GPUSchedEnv, offered_load, read_jobs_csv, time_split  # noqa: E402
from gpusched.baselines import fifo_ff, run_episode  # noqa: E402

FMT = "%Y-%m-%d %H:%M:%S"


def _t(value):
    if value in (None, "None"):
        return None
    try:
        return datetime.strptime(value, FMT)
    except (TypeError, ValueError):
        return None


def real_waits_and_attempts(job_log, vc):
    """Per-job real wait (s) for all jobs, and (start, end, gpus) for every attempt in ``vc``."""
    with Path(job_log).open() as f:
        raw = json.load(f)
    waits, submits, intervals = {}, {}, []
    for job in raw:
        submit = _t(job.get("submitted_time"))
        attempts = job.get("attempts") or []
        if submit is None or not attempts:
            continue
        submits[job.get("jobid")] = submit
        first_start = _t(attempts[0].get("start_time"))
        if first_start is not None:
            waits[job.get("jobid")] = (first_start - submit).total_seconds()
        if job.get("vc") != vc:
            continue
        for att in attempts:
            s, e = _t(att.get("start_time")), _t(att.get("end_time"))
            g = sum(len(d.get("gpus") or []) for d in (att.get("detail") or []))
            if s is not None and e is not None and e > s and g > 0:
                intervals.append((s, e, g))
    return waits, submits, intervals


def gpus_in_use(intervals, t_from, t_to):
    """Time-weighted distribution of GPUs in use between t_from and t_to (datetimes)."""
    events = []
    for s, e, g in intervals:
        events.append((s, g))
        events.append((e, -g))
    events.sort()
    level, last, weights, levels = 0, None, [], []
    for t, dg in events:
        if last is not None and t > last:
            a, b = max(last, t_from), min(t, t_to)
            if b > a:
                weights.append((b - a).total_seconds())
                levels.append(level)
        level += dg
        last = t
    levels, weights = np.array(levels, float), np.array(weights, float)
    order = np.argsort(levels)
    cum = np.cumsum(weights[order]) / weights.sum()

    def pct(q):
        return float(levels[order][np.searchsorted(cum, q)])
    return {"peak": float(levels.max()), "p50": pct(0.5), "p90": pct(0.9),
            "time_avg": float((levels * weights).sum() / weights.sum())}


def describe(w_hours):
    w = np.asarray(w_hours, float)
    return {"jobs": len(w), "mean_h": w.mean(), "median_h": np.median(w),
            "p90_h": np.percentile(w, 90), "p99_h": np.percentile(w, 99),
            "waited_over_1min_%": 100 * (w > 1 / 60).mean(),
            "waited_over_1h_%": 100 * (w > 1).mean()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job-log", required=True)
    ap.add_argument("--jobs-csv", default="data/philly_jobs.csv")
    ap.add_argument("--vc", default="6214e9")
    ap.add_argument("--capacities", default="128,192,256")
    ap.add_argument("--episode-len", type=int, default=1024)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--val-frac", type=float, default=0.15, help="last part of the training period")
    ap.add_argument("--out", default="results/trace_check")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    all_jobs = read_jobs_csv(args.jobs_csv)
    jobs = all_jobs[all_jobs["vc"] == args.vc].reset_index(drop=True)
    train, test = time_split(jobs, args.train_frac)
    n_val = int(len(train) * args.val_frac)
    fit, val = train.iloc[:-n_val], train.iloc[-n_val:]

    waits, submits, intervals = real_waits_and_attempts(args.job_log, args.vc)
    test = test.copy()
    test["real_wait_h"] = test["job_id"].map(waits) / 3600
    missing = int(test["real_wait_h"].isna().sum())

    # philly_jobs.csv times are seconds from the first submission among ALL kept jobs.
    t0 = min(submits[j] for j in all_jobs["job_id"])
    # sanity check: the mapping reproduces the CSV's submit times
    probe = test.iloc[:: max(len(test) // 50, 1)]
    err = max(abs((submits[j] - t0).total_seconds() - s) for j, s in zip(probe["job_id"], probe["submit_time"]))
    if err > 1:
        raise RuntimeError(f"time origin mismatch ({err:.0f} s)")

    # GPUs really in use by this virtual cluster during the test period.
    t_from = t0 + timedelta(seconds=float(test["submit_time"].min()))
    t_to = t0 + timedelta(seconds=float(test["submit_time"].max()))
    usage_test = gpus_in_use(intervals, t_from, t_to)
    t_all_from = t0 + timedelta(seconds=float(jobs["submit_time"].min()))
    usage_all = gpus_in_use(intervals, t_all_from, t_to)

    rows = []
    starts = np.arange(0, len(test) - args.episode_len + 1, args.episode_len)
    covered = np.concatenate([np.arange(s, s + args.episode_len) for s in starts])
    rows.append({"source": "trace (recorded)", **describe(test["real_wait_h"].to_numpy()[covered])})
    load_rows = []
    for cap in [int(c) for c in args.capacities.split(",")]:
        env = GPUSchedEnv(test, capacity=cap, episode_len=args.episode_len, warm_start=True, history=jobs)
        sim = []
        for s in starts:
            _, per_job = run_episode(env, fifo_ff, start=int(s))
            sim.append(per_job["wait_h"])
        rows.append({"source": f"simulator, first-fit, {cap} GPUs", **describe(np.concatenate(sim))})
        load_rows.append({"capacity": cap, "train_load": offered_load(fit, cap),
                          "val_load": offered_load(val, cap), "test_load": offered_load(test, cap)})

    table = pd.DataFrame(rows)
    loads = pd.DataFrame(load_rows)
    table.to_csv(out / "waits_real_vs_sim.csv", index=False)
    loads.to_csv(out / "offered_load.csv", index=False)
    usage = pd.DataFrame([{"period": "test", **usage_test}, {"period": "whole trace", **usage_all}])
    usage.to_csv(out / "gpus_in_use.csv", index=False)

    lines = [f"Virtual cluster {args.vc}: real vs simulated waits on the same "
             f"{len(covered):,} test jobs ({len(starts)} non-overlapping episodes)",
             f"(jobs with no recorded first start: {missing})", "",
             "| Source | Mean wait (h) | Median (h) | 90th pct (h) | 99th pct (h) | Waited > 1 min | Waited > 1 h |",
             "|---|---|---|---|---|---|---|"]
    for _, r in table.iterrows():
        lines.append(f"| {r.source} | {r.mean_h:.2f} | {r.median_h:.3f} | {r.p90_h:.2f} | {r.p99_h:.1f} "
                     f"| {r['waited_over_1min_%']:.0f}% | {r['waited_over_1h_%']:.0f}% |")
    lines += ["", f"GPUs really in use by {args.vc} (from recorded attempt times, time-weighted)", "",
              "| Period | Peak | Median | 90th pct | Average |", "|---|---|---|---|---|"]
    for _, r in usage.iterrows():
        lines.append(f"| {r.period} | {r.peak:.0f} | {r.p50:.0f} | {r.p90:.0f} | {r.time_avg:.0f} |")
    lines += ["", "Offered load (GPU-time requested / (GPUs x time span))", "",
              "| GPUs | Training (fit) | Validation | Test |", "|---|---|---|---|"]
    for _, r in loads.iterrows():
        lines.append(f"| {r.capacity:.0f} | {r.train_load:.2f} | {r.val_load:.2f} | {r.test_load:.2f} |")
    text = "\n".join(lines)
    (out / "summary.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
