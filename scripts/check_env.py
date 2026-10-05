"""Check the environment follows the Gymnasium API and measure its speed.

Usage:
    python scripts/check_env.py                      # synthetic jobs
    python scripts/check_env.py --jobs-csv data/philly_jobs.csv --capacity 64
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from gymnasium.utils.env_checker import check_env

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched import GPUSchedEnv, add_duration_estimates, make_synthetic_jobs  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-csv", default=None)
    ap.add_argument("--capacity", type=int, default=64)
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    if args.jobs_csv:
        jobs = pd.read_csv(args.jobs_csv)
    else:
        jobs = add_duration_estimates(make_synthetic_jobs(n_jobs=5000, capacity=args.capacity))

    env = GPUSchedEnv(jobs, capacity=args.capacity)
    check_env(env, skip_render_check=True)
    print("Gymnasium check_env: passed")
    print(f"Observation size: {env.observation_space.shape[0]}, actions: {env.action_space.n}")

    rng = np.random.default_rng(0)
    steps, t0 = 0, time.time()
    for ep in range(args.episodes):
        env.reset(seed=ep)
        done = False
        while not done:
            valid = np.flatnonzero(env.action_masks())
            _, _, term, trunc, _ = env.step(int(rng.choice(valid)))
            done = term or trunc
            steps += 1
    dt = time.time() - t0
    print(f"Random masked rollouts: {args.episodes} episodes, {steps} steps, "
          f"{steps / dt:,.0f} steps/s ({steps / args.episodes:.0f} steps per episode)")


if __name__ == "__main__":
    main()
