"""Job data for the GPU scheduling environment.

Every job table used by the project has one row per job with these columns:

    job_id, user, vc, status, submit_time, trace_end_time, gpus, duration,
    est_duration, est_source

Times are in seconds. ``submit_time`` and ``trace_end_time`` are measured from
the first submission in the table. ``duration`` is the time the job holds its
GPUs once started. ``est_duration`` is what a scheduler could know at
submission time (see ``add_duration_estimates``).
"""
from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PHILLY_TIME_FMT = "%Y-%m-%d %H:%M:%S"
JOB_COLUMNS = [
    "job_id", "user", "vc", "status",
    "submit_time", "trace_end_time", "gpus", "duration",
]
MIN_DURATION = 1.0  # seconds; shorter jobs are treated as logging noise


# --------------------------------------------------------------------------
# Philly trace
# --------------------------------------------------------------------------
def _parse_time(value):
    if value is None or value == "None":
        return None
    try:
        return datetime.strptime(value, PHILLY_TIME_FMT)
    except (TypeError, ValueError):
        return None


def parse_philly_job_log(path):
    """Parse Philly's ``cluster_job_log`` (a JSON list) into a job table.

    Modeling choices (state these in the paper):
      * gpus     = largest number of GPUs used by any attempt (the gang size).
      * duration = sum over attempts of (end - start), i.e. total GPU-holding
                   time. Retries are folded into one job; the time between
                   retries is not counted.
      * Dropped: jobs with no submit time, no attempts, no attempt with valid
        start and end times, zero GPUs, or a last attempt with no end time
        (still running when the trace was captured).

    Returns (jobs DataFrame, dict of drop counts).
    """
    with Path(path).open() as f:
        raw = json.load(f)

    dropped = {
        "no_submit_time": 0, "no_attempts": 0, "still_running": 0,
        "no_valid_attempt": 0, "zero_gpus": 0,
    }
    rows = []
    for job in raw:
        submit = _parse_time(job.get("submitted_time"))
        if submit is None:
            dropped["no_submit_time"] += 1
            continue
        attempts = job.get("attempts") or []
        if not attempts:
            dropped["no_attempts"] += 1
            continue
        if attempts[-1].get("end_time") in (None, "None"):
            dropped["still_running"] += 1
            continue

        duration, gpus, last_end = 0.0, 0, None
        for att in attempts:
            n_gpus = sum(len(d.get("gpus") or []) for d in (att.get("detail") or []))
            gpus = max(gpus, n_gpus)
            start, end = _parse_time(att.get("start_time")), _parse_time(att.get("end_time"))
            if start is not None and end is not None and end > start:
                duration += (end - start).total_seconds()
                last_end = end if last_end is None else max(last_end, end)

        if last_end is None or duration < MIN_DURATION:
            dropped["no_valid_attempt"] += 1
            continue
        if gpus == 0:
            dropped["zero_gpus"] += 1
            continue
        rows.append((job.get("jobid"), job.get("user"), job.get("vc"), job.get("status"),
                     submit, last_end, gpus, duration))

    df = pd.DataFrame(rows, columns=["job_id", "user", "vc", "status",
                                     "submit_dt", "end_dt", "gpus", "duration"])
    if df.empty:
        raise ValueError(f"No usable jobs found in {path}")
    t0 = df["submit_dt"].min()
    df["submit_time"] = (df["submit_dt"] - t0).dt.total_seconds()
    df["trace_end_time"] = (df["end_dt"] - t0).dt.total_seconds()
    df = df[JOB_COLUMNS].sort_values("submit_time", kind="stable").reset_index(drop=True)
    df["gpus"] = df["gpus"].astype(int)
    return df, dropped


# --------------------------------------------------------------------------
# Synthetic jobs (for testing, and as a fallback when the trace is unavailable)
# --------------------------------------------------------------------------
def make_synthetic_jobs(n_jobs=20_000, n_users=60, capacity=64, target_load=0.9, seed=0):
    """Generate jobs with Philly-like shapes. NOT real data; label results as synthetic.

    * Most jobs use 1 GPU; a few use 8-32.
    * Each user has a typical job length; individual jobs vary around it
      (log-normal), so a user's past average is informative but noisy.
    * Job lengths are heavy-tailed: 10 seconds to 10 days.
    * Arrivals are random (Poisson), with the rate set so the offered load is
      roughly ``target_load`` on a cluster of ``capacity`` GPUs.
    """
    rng = np.random.default_rng(seed)
    gpu_choices = np.array([1, 2, 4, 8, 16, 32])
    gpu_probs = np.array([0.62, 0.10, 0.12, 0.11, 0.04, 0.01])

    user_log_mean = rng.normal(math.log(1800.0), 1.5, size=n_users)
    user_weights = rng.dirichlet(np.full(n_users, 0.5))
    users = rng.choice(n_users, size=n_jobs, p=user_weights)
    duration = np.exp(rng.normal(user_log_mean[users], 1.2))
    duration = np.clip(duration, 10.0, 10 * 24 * 3600.0)
    gpus = np.minimum(rng.choice(gpu_choices, size=n_jobs, p=gpu_probs), capacity)

    rate = target_load * capacity / float(np.mean(gpus * duration))
    gaps = rng.exponential(1.0 / rate, size=n_jobs)
    submit = np.cumsum(gaps) - gaps[0]

    df = pd.DataFrame({
        "job_id": [f"syn_{i}" for i in range(n_jobs)],
        "user": [f"u{u}" for u in users],
        "vc": "synthetic",
        "status": "Pass",
        "submit_time": submit,
        "trace_end_time": submit + duration,  # history as if the job had not waited
        "gpus": gpus.astype(int),
        "duration": duration,
    })
    return df


# --------------------------------------------------------------------------
# Duration estimates (no peeking at the future)
# --------------------------------------------------------------------------
def add_duration_estimates(df, kind="mean", default_estimate=3600.0):
    """Add ``est_duration``: the submitting user's past average job length.

    Only jobs that FINISHED (per the trace) strictly before this job was
    SUBMITTED are used, so the estimate never uses future information.
    Fallbacks: average over all users' finished jobs, then ``default_estimate``.

    kind = "mean"    -> arithmetic mean (what the proposal states)
    kind = "geomean" -> geometric mean (less sensitive to a few very long jobs)
    """
    if kind not in ("mean", "geomean"):
        raise ValueError("kind must be 'mean' or 'geomean'")
    df = df.sort_values("submit_time", kind="stable").reset_index(drop=True)
    order = np.argsort(df["trace_end_time"].to_numpy(), kind="stable")
    end_sorted = df["trace_end_time"].to_numpy()[order]
    user_sorted = df["user"].to_numpy()[order]
    value_sorted = df["duration"].to_numpy()[order]
    if kind == "geomean":
        value_sorted = np.log(value_sorted)

    user_sum, user_cnt = {}, {}
    all_sum, all_cnt, ptr = 0.0, 0, 0
    est = np.empty(len(df))
    source = np.empty(len(df), dtype=object)
    for i, (t_submit, user) in enumerate(zip(df["submit_time"].to_numpy(), df["user"].to_numpy())):
        while ptr < len(end_sorted) and end_sorted[ptr] < t_submit:
            u, v = user_sorted[ptr], value_sorted[ptr]
            user_sum[u] = user_sum.get(u, 0.0) + v
            user_cnt[u] = user_cnt.get(u, 0) + 1
            all_sum += v
            all_cnt += 1
            ptr += 1
        if user_cnt.get(user, 0) > 0:
            value, source[i] = user_sum[user] / user_cnt[user], "user"
        elif all_cnt > 0:
            value, source[i] = all_sum / all_cnt, "global"
        else:
            est[i], source[i] = default_estimate, "default"
            continue
        est[i] = math.exp(value) if kind == "geomean" else value
    df["est_duration"] = est
    df["est_source"] = source
    return df


# --------------------------------------------------------------------------
# Splits and cluster sizing
# --------------------------------------------------------------------------
def time_split(df, train_frac=0.7):
    """Split by submit time: early jobs for training, later jobs for testing."""
    df = df.sort_values("submit_time", kind="stable").reset_index(drop=True)
    cut = int(len(df) * train_frac)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def capacity_for_load(df, load=0.9, multiple=8):
    """Cluster size (GPUs) at which the offered load is about ``load``.

    offered load = total GPU-seconds requested / (capacity * time span).
    Rounded up to a multiple of ``multiple`` (servers have 8 GPUs). Jobs that
    ask for more GPUs than this are dropped by the environment, which reports
    how many it dropped.
    """
    span = df["submit_time"].max() - df["submit_time"].min()
    if span <= 0:
        raise ValueError("Need jobs spread over time to compute load")
    work = float((df["gpus"] * df["duration"]).sum())
    cap = work / (span * load)
    cap = int(math.ceil(cap / multiple) * multiple)
    return max(cap, multiple)


def offered_load(df, capacity):
    span = df["submit_time"].max() - df["submit_time"].min()
    return float((df["gpus"] * df["duration"]).sum()) / (capacity * span)
