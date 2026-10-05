"""Rule-based baseline schedulers.

Every policy sees only the visible window (the first ``window`` queued jobs),
exactly like the RL agent, so comparisons are fair.

    random      : random job among those that fit
    fifo        : strict first-in-first-out. If the oldest job does not fit,
                  WAIT (nothing jumps ahead of it).
    fifo_ff     : first-fit. The oldest job that fits right now.
    sjf         : shortest job first, using the user's past average
                  (realistic: true durations are unknown at submission).
    sjf_oracle  : shortest job first using the TRUE duration. Impossible in
                  practice; shows how much perfect duration knowledge is worth.
    sgf         : smallest GPU request first.

The greedy rules (all except fifo) never wait while some visible job fits.
"""
from __future__ import annotations

import numpy as np


def _fitting(env):
    return [j for j in env.visible_jobs() if j["fits"]]


def _first_fitting_slot(env):
    fits = _fitting(env)
    return fits[0]["slot"] if fits else env.WAIT


def make_random(seed=0):
    rng = np.random.default_rng(seed)

    def policy(env):
        fits = _fitting(env)
        return int(rng.choice([j["slot"] for j in fits])) if fits else env.WAIT
    return policy


def fifo(env):
    visible = env.visible_jobs()
    if visible and visible[0]["fits"]:
        return 0
    mask = env.action_masks()
    return env.WAIT if mask[env.WAIT] else _first_fitting_slot(env)


def fifo_ff(env):
    return _first_fitting_slot(env)


def _argmin_slot(env, key):
    fits = _fitting(env)
    if not fits:
        return env.WAIT
    return min(fits, key=lambda j: (j[key], j["slot"]))["slot"]


def sjf(env):
    return _argmin_slot(env, "est_duration")


def sjf_oracle(env):
    return _argmin_slot(env, "true_duration")


def sgf(env):
    return _argmin_slot(env, "gpus")


def get_policies(seed=0):
    return {
        "random": make_random(seed),
        "fifo": fifo,
        "fifo_ff": fifo_ff,
        "sjf": sjf,
        "sjf_oracle": sjf_oracle,
        "sgf": sgf,
    }


def run_episode(env, policy, start=None, seed=None):
    """Run one full episode. Returns (metrics dict, per-job record dict)."""
    env.reset(seed=seed, options={"start": start} if start is not None else None)
    total_reward, terminated, truncated, info = 0.0, False, False, {}
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(policy(env))
        total_reward += reward
    if truncated:
        raise RuntimeError("Episode was truncated before all jobs finished")
    metrics = dict(info["metrics"])
    metrics["episode_reward"] = total_reward
    metrics["steps"] = env.steps
    metrics["invalid_actions"] = env.invalid_actions
    jobs = {
        "jct_h": (env.finish_time - env.submit) / 3600,
        "wait_h": (env.start_time - env.submit) / 3600,
        "duration_h": env.dur / 3600,
        "gpus": env.gpus.copy(),
    }
    return metrics, jobs
