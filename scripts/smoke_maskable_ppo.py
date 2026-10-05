"""Quick check that GPUSchedEnv trains with sb3-contrib's MaskablePPO.

This is NOT a result: 20,000 steps is far too little training. It only checks
that (1) training runs, (2) the action masks are respected (0 invalid actions),
and (3) how fast training is on this machine, to plan Phase 2.

Usage:
    pip install -r requirements-rl.txt
    python scripts/smoke_maskable_ppo.py
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched import GPUSchedEnv, add_duration_estimates, make_synthetic_jobs, time_split  # noqa: E402
from gpusched.baselines import fifo_ff, run_episode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-csv", default="data/philly_jobs.csv")
    ap.add_argument("--vc", default="6214e9")
    ap.add_argument("--capacity", type=int, default=256)
    ap.add_argument("--timesteps", type=int, default=20_000)
    args = ap.parse_args()

    if Path(args.jobs_csv).exists():
        jobs = pd.read_csv(args.jobs_csv)
        jobs = jobs[jobs["vc"] == args.vc].reset_index(drop=True)
        source = f"philly vc={args.vc}"
    else:
        jobs = add_duration_estimates(make_synthetic_jobs(capacity=args.capacity))
        source = "synthetic"
    train, test = time_split(jobs)

    # Training setup: 256-job episodes from the training period, warm start.
    train_env = GPUSchedEnv(train, capacity=args.capacity, episode_len=256,
                            warm_start=True, history=jobs)
    masked_env = ActionMasker(train_env, lambda env: env.action_masks())
    model = MaskablePPO("MlpPolicy", masked_env, seed=0, device="cpu", verbose=0)

    t0 = time.time()
    model.learn(total_timesteps=args.timesteps)
    seconds = time.time() - t0
    steps_per_s = args.timesteps / seconds
    print(f"Data: {source}, {args.capacity} GPUs, warm start")
    print(f"MaskablePPO trained {args.timesteps:,} steps in {seconds:.0f} s ({steps_per_s:,.0f} steps/s)")
    print(f"Estimated time for 1,000,000 steps: {1_000_000 / steps_per_s / 60:.0f} min")

    # Test setup: 1,024-job episodes from the test period, warm start.
    test_env = GPUSchedEnv(test, capacity=args.capacity, episode_len=1024,
                           warm_start=True, history=jobs)

    def ppo_policy(env):
        action, _ = model.predict(env._obs(), action_masks=env.action_masks(), deterministic=True)
        return int(action)

    total_invalid = 0
    for start in (0, 1024, 2048):
        m_ff, _ = run_episode(test_env, fifo_ff, start=start)
        m_rl, _ = run_episode(test_env, ppo_policy, start=start)
        total_invalid += m_rl["invalid_actions"]
        print(f"Test episode starting at job {start}: first-fit avg wait {m_ff['avg_wait_h']:.2f} h | "
              f"barely-trained PPO avg wait {m_rl['avg_wait_h']:.2f} h | "
              f"invalid actions {m_rl['invalid_actions']}")

    if total_invalid == 0:
        print("CHECK PASSED: MaskablePPO trains on GPUSchedEnv and never chose an invalid action.")
    else:
        print(f"CHECK FAILED: {total_invalid} invalid actions; the masks are not being used.")


if __name__ == "__main__":
    main()
