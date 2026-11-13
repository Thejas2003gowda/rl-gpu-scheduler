"""Run-time prediction from information a scheduler has at submit time.

``history_features`` builds, for every job, features that use ONLY:
  * the job's own request (GPUs, submit time), and
  * jobs that, per the trace, had FINISHED strictly before this job was
    submitted (their run times and outcomes), or had been SUBMITTED before it
    (submission times only; their run times are not known yet).

Never used (they leak the future): the job's own status, run time, end time
or number of attempts, and the run times of jobs still running.

Simple predictors (user mean / geometric mean / median / last-2) come from the
same features. A gradient-boosted model is trained on log run time with labels
that were available at the training cut-off (see ``label_available``).
"""
from __future__ import annotations

import bisect
import math
from collections import deque
from datetime import datetime, timezone

import numpy as np
import pandas as pd

FAILED = ("Failed", "Killed")
FEATURES = [
    "log_gpus", "u_n", "u_mean", "u_std", "u_median", "u_last1", "u_last2",
    "u_fail_frac", "ug_n", "ug_mean", "hour", "weekday", "log_since_prev",
    "sub_1h", "sub_24h", "u_running",
]


def history_features(df):
    """Leak-free features for every row of ``df`` (any number of virtual clusters).

    Log run times are natural logs of seconds. Missing history -> NaN.
    Returns a DataFrame aligned with ``df.index``.
    """
    order = np.argsort(df["submit_time"].to_numpy(), kind="stable")
    submit = df["submit_time"].to_numpy(float)
    end = df["trace_end_time"].to_numpy(float)
    users = df["user"].to_numpy()
    gpus = df["gpus"].to_numpy(int)
    logdur = np.log(df["duration"].to_numpy(float))
    failed = df["status"].isin(FAILED).to_numpy()
    unix = df["submit_unix"].to_numpy(float)

    fin = np.argsort(end, kind="stable")
    ptr = 0
    u_n, u_s, u_ss, u_sorted, u_last, u_fail = {}, {}, {}, {}, {}, {}
    ug_n, ug_s = {}, {}
    u_sub, u_prev, u_recent = {}, {}, {}

    out = np.full((len(df), len(FEATURES)), np.nan)
    for i in order:
        t, u, g = submit[i], users[i], gpus[i]
        # 1) everything that finished strictly before t becomes known history
        while ptr < len(fin) and end[fin[ptr]] < t:
            j = fin[ptr]
            uj, x = users[j], logdur[j]
            u_n[uj] = u_n.get(uj, 0) + 1
            u_s[uj] = u_s.get(uj, 0.0) + x
            u_ss[uj] = u_ss.get(uj, 0.0) + x * x
            bisect.insort(u_sorted.setdefault(uj, []), x)
            last = u_last.setdefault(uj, [])
            last.append(x)
            if len(last) > 2:
                last.pop(0)
            u_fail[uj] = u_fail.get(uj, 0) + int(failed[j])
            key = (uj, gpus[j])
            ug_n[key] = ug_n.get(key, 0) + 1
            ug_s[key] = ug_s.get(key, 0.0) + x
            ptr += 1

        row = out[i]
        row[0] = math.log(g)
        n = u_n.get(u, 0)
        row[1] = math.log1p(n)
        if n:
            mean = u_s[u] / n
            row[2] = mean
            row[3] = math.sqrt(max(u_ss[u] / n - mean * mean, 0.0))
            srt = u_sorted[u]
            row[4] = 0.5 * (srt[(n - 1) // 2] + srt[n // 2])
            row[5] = u_last[u][-1]
            row[6] = float(np.mean(u_last[u]))
            row[7] = u_fail[u] / n
        k = ug_n.get((u, g), 0)
        row[8] = math.log1p(k)
        if k:
            row[9] = ug_s[(u, g)] / k
        stamp = datetime.fromtimestamp(unix[i], tz=timezone.utc)
        row[10] = stamp.hour + stamp.minute / 60
        row[11] = stamp.weekday()
        if u in u_prev:
            row[12] = math.log1p(t - u_prev[u])
        recent = u_recent.setdefault(u, deque())
        while recent and recent[0] < t - 86400:
            recent.popleft()
        row[13] = math.log1p(sum(1 for s in recent if s >= t - 3600))
        row[14] = math.log1p(len(recent))
        row[15] = math.log1p(u_sub.get(u, 0) - n)  # submitted earlier, not finished yet
        # 2) now record this submission (time only; its run time stays unknown)
        u_sub[u] = u_sub.get(u, 0) + 1
        u_prev[u] = t
        recent.append(t)
    return pd.DataFrame(out, columns=FEATURES, index=df.index)


def simple_predictions(feats, user_mean_seconds, fallback_log):
    """Log run-time predictions from the simple rules. NaN history -> fallback."""
    def fill(x):
        return np.where(np.isnan(x), fallback_log, x)
    return {
        "user_mean": np.log(np.maximum(user_mean_seconds, 1.0)),  # existing estimate (arithmetic mean)
        "user_geomean": fill(feats["u_mean"].to_numpy()),
        "user_median": fill(feats["u_median"].to_numpy()),
        "user_last2": fill(feats["u_last2"].to_numpy()),
        "user_gpu_mean": fill(np.where(np.isnan(feats["ug_mean"]), feats["u_mean"], feats["ug_mean"])),
    }


def label_available(df, cutoff):
    """Jobs whose run time was known at ``cutoff``: submitted and finished before it."""
    return (df["trace_end_time"] < cutoff).to_numpy()


def fit_model(X, y, seed=0, **params):
    from sklearn.ensemble import HistGradientBoostingRegressor
    defaults = dict(loss="absolute_error", learning_rate=0.05, max_iter=400, max_leaf_nodes=31,
                    min_samples_leaf=50, l2_regularization=1.0, random_state=seed)
    defaults.update(params)
    return HistGradientBoostingRegressor(**defaults).fit(X, y)


# ------------------------------------------------------------------ scoring
def error_factor(pred_log, true_log):
    """Median multiplicative error: exp(median |log pred - log true|). 1 = perfect."""
    return float(np.exp(np.median(np.abs(np.asarray(pred_log) - np.asarray(true_log)))))


def pair_order_accuracy(pred_log, true_log, window=10, min_ratio=1.1):
    """How often the prediction orders two nearby jobs the right way round.

    Jobs must be in submit order. Pairs are jobs less than ``window`` apart
    (as they would share the agent's 10-job view). Pairs whose true run times
    differ by less than ``min_ratio`` are skipped (near ties). A tied
    prediction counts as half right.
    """
    p, t = np.asarray(pred_log, float), np.asarray(true_log, float)
    right = total = 0.0
    margin = math.log(min_ratio)
    for d in range(1, window):
        dt = t[d:] - t[:-d]
        dp = p[d:] - p[:-d]
        keep = np.abs(dt) >= margin
        dt, dp = dt[keep], dp[keep]
        right += np.sum(np.sign(dt) == np.sign(dp)) + 0.5 * np.sum(dp == 0)
        total += len(dt)
    return float(right / total) if total else float("nan")


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- agent inputs
# Run-time signals given to the agent directly (duration_info="signals"), one
# block per queued job, all scaled to [0, 1] with fixed scales and computed from
# ``history_features`` (so they never use future information). Missing history
# -> 0; the user's job count (also 0 then) tells the agent there is no history.
AGENT_SIGNALS = [
    "sig_user_jobs",      # log(1 + user's finished jobs) / 10
    "sig_user_median",    # median log run time of the user's finished jobs / log(30 days)
    "sig_user_spread",    # std of those log run times / 5
    "sig_user_last2",     # mean log run time of the user's last 2 finished jobs / log(30 days)
    "sig_user_fail",      # share of the user's finished jobs that failed or were killed
    "sig_user_gpu_mean",  # mean log run time of the user's finished jobs with the same GPU count
    "sig_hour_sin",       # time of day (UTC), cyclic: (sin + 1) / 2
    "sig_hour_cos",       #                             (cos + 1) / 2
]
_LOG_MAX_TIME = math.log1p(30 * 24 * 3600.0)


def agent_signal_features(df):
    """Per-job agent inputs (``AGENT_SIGNALS``) for every row of ``df``, aligned with its index."""
    f = history_features(df)
    angle = 2 * np.pi * f["hour"] / 24
    out = pd.DataFrame({
        "sig_user_jobs": f["u_n"] / 10,
        "sig_user_median": f["u_median"] / _LOG_MAX_TIME,
        "sig_user_spread": f["u_std"] / 5,
        "sig_user_last2": f["u_last2"] / _LOG_MAX_TIME,
        "sig_user_fail": f["u_fail_frac"],
        "sig_user_gpu_mean": f["ug_mean"] / _LOG_MAX_TIME,
        "sig_hour_sin": (np.sin(angle) + 1) / 2,
        "sig_hour_cos": (np.cos(angle) + 1) / 2,
    }, index=df.index)
    return out[AGENT_SIGNALS].fillna(0.0).clip(0.0, 1.0)


def add_agent_signals(jobs):
    """Copy of ``jobs`` with the ``AGENT_SIGNALS`` columns added (no-op if present)."""
    if all(c in jobs.columns for c in AGENT_SIGNALS):
        return jobs
    out = jobs.copy()
    out[AGENT_SIGNALS] = agent_signal_features(jobs)
    return out
