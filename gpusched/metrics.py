"""Scheduling metrics and summary statistics."""
from __future__ import annotations

import numpy as np

BSLD_THRESHOLD = 10.0  # seconds; standard floor so tiny jobs don't explode slowdown


def episode_metrics(submit, start, finish, gpus, duration, capacity,
                    bsld_threshold=BSLD_THRESHOLD, busy_gpu_seconds=None):
    """Metrics for one finished episode. All arrays are per job, in seconds.

    If ``busy_gpu_seconds`` is given (measured by the simulator, including any
    warm-start background jobs), utilization uses it; otherwise it uses the
    episode jobs' own GPU-seconds.
    """
    submit, start, finish = map(np.asarray, (submit, start, finish))
    gpus, duration = np.asarray(gpus, float), np.asarray(duration, float)
    if np.isnan(start).any() or np.isnan(finish).any():
        raise ValueError("Episode not finished: some jobs never started or finished")

    wait = start - submit
    jct = finish - submit
    bsld = np.maximum(jct / np.maximum(duration, bsld_threshold), 1.0)
    makespan = float(finish.max() - submit.min())
    busy = float((gpus * duration).sum()) if busy_gpu_seconds is None else float(busy_gpu_seconds)
    utilization = busy / (capacity * makespan) if makespan > 0 else 0.0
    return {
        "avg_wait_h": float(wait.mean() / 3600),
        "median_wait_h": float(np.median(wait) / 3600),
        "avg_jct_h": float(jct.mean() / 3600),
        "median_jct_h": float(np.median(jct) / 3600),
        "p99_jct_h": float(np.percentile(jct, 99) / 3600),
        "avg_bsld": float(bsld.mean()),
        "utilization": utilization,
        "makespan_h": makespan / 3600,
    }


def iqm(values):
    """Interquartile mean: average of the middle 50% (Agarwal et al., 2021)."""
    x = np.sort(np.asarray(values, float))
    n = len(x)
    if n == 0:
        return float("nan")
    lo = int(np.floor(0.25 * n))
    hi = n - lo
    return float(x[lo:hi].mean()) if hi > lo else float(x.mean())


def bootstrap_ci(values, stat=np.mean, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap confidence interval for ``stat``."""
    x = np.asarray(values, float)
    if len(x) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boots = np.array([stat(rng.choice(x, size=len(x), replace=True)) for _ in range(n_boot)])
    return float(np.percentile(boots, 100 * alpha / 2)), float(np.percentile(boots, 100 * (1 - alpha / 2)))


def paired_bootstrap(policy_vals, ref_vals, n_boot=5000, alpha=0.05, seed=0):
    """Compare a policy with a reference on the SAME episodes.

    Inputs are per-episode values (e.g. average wait), in the same episode order.
    Returns the mean per-episode difference (policy - reference) with a 95%
    bootstrap CI, the % change in the total (sum policy / sum reference - 1)
    with a 95% CI, and the share of episodes where the policy was lower.
    Episodes are resampled together, so both policies always see the same
    resampled episodes.
    """
    a = np.asarray(policy_vals, float)
    b = np.asarray(ref_vals, float)
    if a.shape != b.shape or len(a) < 2:
        raise ValueError("need two equal-length arrays with at least 2 episodes")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = (a[idx] - b[idx]).mean(axis=1)
    ratios = a[idx].sum(axis=1) / b[idx].sum(axis=1) - 1.0
    lo, hi = 100 * alpha / 2, 100 * (1 - alpha / 2)
    return {
        "mean_diff": float((a - b).mean()),
        "diff_ci_lo": float(np.percentile(diffs, lo)),
        "diff_ci_hi": float(np.percentile(diffs, hi)),
        "pct_change": float(100 * (a.sum() / b.sum() - 1.0)),
        "pct_ci_lo": float(100 * np.percentile(ratios, lo)),
        "pct_ci_hi": float(100 * np.percentile(ratios, hi)),
        "frac_better": float((a < b - 1e-12).mean()),
    }
