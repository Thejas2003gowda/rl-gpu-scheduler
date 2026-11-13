"""Statistics for RL results across seeds and test episodes.

The success criterion: for each seed, compute the change in total
waiting time vs first-fit over the test episodes; the agent succeeds if the
interquartile mean (IQM) of this change across seeds has a 95% bootstrap CI
(resampling seeds AND episodes) entirely below zero.

All episodes have the same number of jobs, so the sum of per-episode average
waits is proportional to the total wait.
"""
from __future__ import annotations

import numpy as np

from .metrics import iqm


def change_per_seed(pol, ref):
    """pol: (seeds, episodes) average waits; ref: (episodes,). Returns % change per seed."""
    pol, ref = np.atleast_2d(np.asarray(pol, float)), np.asarray(ref, float)
    return 100.0 * (pol.sum(axis=1) / ref.sum() - 1.0)


def success_test(pol, ref, n_boot=10_000, alpha=0.05, seed=0):
    """IQM over seeds of the % change in total wait vs the reference, with a CI
    from resampling seeds and episodes together (the same resampled episodes for
    every seed and for the reference, so the comparison stays paired)."""
    pol, ref = np.atleast_2d(np.asarray(pol, float)), np.asarray(ref, float)
    n_seeds, n_eps = pol.shape
    if ref.shape != (n_eps,):
        raise ValueError("ref must have one value per episode")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.integers(0, n_seeds, n_seeds)
        e = rng.integers(0, n_eps, n_eps)
        boots[b] = iqm(change_per_seed(pol[np.ix_(s, e)], ref[e]))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = iqm(change_per_seed(pol, ref))
    return {"iqm_change_pct": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "success": bool(hi < 0), "n_seeds": int(n_seeds), "n_episodes": int(n_eps),
            "per_seed_change_pct": change_per_seed(pol, ref).tolist()}


def gap_recovered(pol, ref, oracle, n_boot=10_000, alpha=0.05, seed=0):
    """Share of the reference-to-oracle gap closed by the policy, on totals:
    (ref - pol) / (ref - oracle). pol may be (seeds, episodes); seeds are
    averaged. Computed on totals because per-episode ratios explode when an
    episode's gap is near zero."""
    pol = np.atleast_2d(np.asarray(pol, float)).mean(axis=0)
    ref, oracle = np.asarray(ref, float), np.asarray(oracle, float)
    rng = np.random.default_rng(seed)

    def share(idx):
        gap = ref[idx].sum() - oracle[idx].sum()
        return (ref[idx].sum() - pol[idx].sum()) / gap if gap > 0 else np.nan

    n = len(ref)
    boots = np.array([share(rng.integers(0, n, n)) for _ in range(n_boot)])
    lo, hi = np.nanpercentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"share": float(share(np.arange(n))), "ci_lo": float(lo), "ci_hi": float(hi)}


def difference_test(pol_a, pol_b, ref, n_boot=10_000, alpha=0.05, seed=0):
    """Is agent A better than agent B? Difference of their IQM % changes vs the
    reference (A minus B; negative = A waits less), with a CI that resamples A's
    seeds and B's seeds separately (they are different training runs) and the
    episodes jointly (the same resampled episodes for A, B and the reference)."""
    pol_a, pol_b = np.atleast_2d(np.asarray(pol_a, float)), np.atleast_2d(np.asarray(pol_b, float))
    ref = np.asarray(ref, float)
    n_eps = len(ref)
    if pol_a.shape[1] != n_eps or pol_b.shape[1] != n_eps:
        raise ValueError("both agents need one value per reference episode")
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for b in range(n_boot):
        sa = rng.integers(0, len(pol_a), len(pol_a))
        sb = rng.integers(0, len(pol_b), len(pol_b))
        e = rng.integers(0, n_eps, n_eps)
        boots[b] = (iqm(change_per_seed(pol_a[np.ix_(sa, e)], ref[e]))
                    - iqm(change_per_seed(pol_b[np.ix_(sb, e)], ref[e])))
    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = iqm(change_per_seed(pol_a, ref)) - iqm(change_per_seed(pol_b, ref))
    return {"difference_pct_points": float(point), "ci_lo": float(lo), "ci_hi": float(hi),
            "a_better": bool(hi < 0), "n_seeds_a": int(len(pol_a)), "n_seeds_b": int(len(pol_b)),
            "n_episodes": int(n_eps)}
