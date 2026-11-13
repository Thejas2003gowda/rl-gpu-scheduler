"""Shared setup for training and evaluation: data splits, environments, policies.

Periods of one virtual cluster (by submit time):
    fit         first 70% minus its last 15%   -> RL training episodes (256 jobs)
    validation  last 15% of the first 70%      -> checkpoint selection, tuning
    test        last 30%                       -> final results only
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .data import read_jobs_csv, time_split
from .env import GPUSchedEnv

TRAIN_FRAC, VAL_FRAC = 0.7, 0.15


def load_vc_jobs(jobs_csv, vc):
    jobs = read_jobs_csv(jobs_csv)
    jobs = jobs[jobs["vc"] == str(vc)].sort_values("submit_time", kind="stable").reset_index(drop=True)
    if jobs.empty:
        raise ValueError(f"no jobs for virtual cluster {vc} in {jobs_csv}")
    return jobs


def splits(jobs):
    train, test = time_split(jobs, TRAIN_FRAC)
    n_val = int(len(train) * VAL_FRAC)
    return {"fit": train.iloc[:-n_val].reset_index(drop=True),
            "validation": train.iloc[-n_val:].reset_index(drop=True),
            "test": test}


def make_env(jobs, period, capacity, duration_info, episode_len, window=10, no_wait=False):
    part = splits(jobs)[period]
    return GPUSchedEnv(part, capacity=capacity, window=window, episode_len=episode_len,
                       duration_info=duration_info, warm_start=True, history=jobs, no_wait=no_wait)


def nonoverlap_starts(env):
    return np.arange(0, env.n_jobs_total - env.episode_len + 1, env.episode_len)


def overlapping_starts(env, n=30, seed=0):
    """Same placement rule as run_baselines.py (seed 0), so episodes match."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, env.n_jobs_total - env.episode_len + 1, size=n)


class CountingPolicy:
    """Wraps an agent as a policy(env) -> action and counts deliberate WAITs.
    The simulator only asks for an action when some visible job fits, so every
    WAIT chosen is a deliberate decision to keep GPUs free."""

    def __init__(self, act):
        self.act = act
        self.steps = self.waits = 0

    def __call__(self, env):
        a = int(self.act(env._obs(), env.action_masks()))
        self.steps += 1
        self.waits += int(a == env.WAIT)
        return a


def load_agent(path):
    """Returns act(obs, mask) for a saved DQN (.pt) or MaskablePPO (.zip) checkpoint."""
    path = Path(path)
    if path.suffix == ".pt":
        from .dqn import MaskedDQN
        agent = MaskedDQN.load(path)
        return agent.predict
    from sb3_contrib import MaskablePPO
    model = MaskablePPO.load(path, device="cpu")

    def act(obs, mask):
        a, _ = model.predict(obs, action_masks=mask, deterministic=True)
        return int(a)
    return act
