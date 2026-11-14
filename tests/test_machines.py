"""Tests for the machine-level cluster model (machine_gpus=m)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched import GPUSchedEnv  # noqa: E402
from gpusched.baselines import get_policies, run_episode  # noqa: E402


def jobs_table(submit, gpus, dur):
    n = len(submit)
    return pd.DataFrame({"job_id": [f"j{i}" for i in range(n)], "user": "u", "submit_time": np.asarray(submit, float),
                         "gpus": np.asarray(gpus, int), "duration": np.asarray(dur, float),
                         "est_duration": np.asarray(dur, float)})


def random_jobs(n=400, seed=0, sizes=(1, 1, 1, 1, 2, 4, 8)):
    rng = np.random.default_rng(seed)
    return jobs_table(np.sort(rng.uniform(0, 40_000, n)), rng.choice(sizes, n), np.exp(rng.normal(7, 1.5, n)))


def test_one_machine_is_the_pool_model():
    """With one machine as big as the cluster, every policy gives exactly the same schedule."""
    jobs = random_jobs()
    pool = GPUSchedEnv(jobs, capacity=16, episode_len=128, warm_start=True, history=jobs)
    one = GPUSchedEnv(jobs, capacity=16, episode_len=128, warm_start=True, history=jobs, machine_gpus=16)
    for name in get_policies(seed=0):
        if name == "sjf_pred":
            continue
        for start in (0, 100, 250):
            a, _ = run_episode(pool, get_policies(seed=start)[name], start=start)   # fresh random state
            b, _ = run_episode(one, get_policies(seed=start)[name], start=start)
            assert a == b, (name, start)


def test_fragmentation_blocks_a_whole_machine_job():
    """12 one-GPU jobs fill machine 0 and half of machine 1 (best-fit); the 4 short ones on
    machine 0 finish at t=10, leaving 4 + 4 free GPUs. An 8-GPU job arriving at t=20 starts
    at once in the pool model, but on 8-GPU machines it waits until t=1000."""
    submit = [0.0] * 12 + [20.0]
    gpus = [1] * 12 + [8]
    dur = [10.0] * 4 + [1000.0] * 8 + [100.0]
    jobs = jobs_table(submit, gpus, dur)
    starts = {}
    for name, m in (("pool", None), ("machines", 8)):
        env = GPUSchedEnv(jobs, capacity=16, window=16, episode_len=13, machine_gpus=m)
        env.reset(options={"start": 0})
        done = False
        while not done:
            mask = env.action_masks()
            if name == "machines" and 12 in env.queue and env.now >= 20 and env.now < 1000:
                slot = env.queue.index(12)
                assert env.free == 8 and not mask[slot]           # 8 GPUs free, but not on one machine
                assert env._obs()[slot * 5 + 2] == 0.0             # "fits now" says no
                assert sorted(env.mfree.tolist()) == [4, 4]
            _, _, done, trunc, _ = env.step(int(np.flatnonzero(mask)[0]))
            done = done or trunc
        starts[name] = env.start_time[12]
    assert starts["pool"] == 20.0 and starts["machines"] == 1000.0


def test_fits_and_best_fit_placement():
    env = GPUSchedEnv(jobs_table([0.0] * 4, [1] * 4, [5.0] * 4), capacity=32, episode_len=4, machine_gpus=8)
    env.reset(options={"start": 0})
    env.mfree[:] = [7, 4, 8, 2]
    env.free = int(env.mfree.sum())
    env._refresh_machines()
    assert env._fits(8) and env._fits(7) and not env._fits(16)
    assert env._place(3) == [(1, 3)]          # fewest free GPUs that still fit
    assert env._place(2) == [(3, 2)]
    assert env._place(8) == [(2, 8)]
    assert not env._fits(8) and env._fits(7)
    env.mfree[:] = [8, 0, 8, 8]
    env._refresh_machines()
    assert env._fits(16) and env._place(16) == [(0, 8), (2, 8)]   # two empty machines
    with pytest.raises(ValueError):
        GPUSchedEnv(random_jobs(), capacity=20, episode_len=128, machine_gpus=8)


@pytest.mark.parametrize("seed", [0, 1])
def test_machine_bookkeeping_stays_consistent(seed):
    """Free GPUs per machine always add up to the free total, with warm start and random choices."""
    jobs = random_jobs(500, seed, sizes=(1, 1, 1, 2, 4, 8, 16))
    env = GPUSchedEnv(jobs, capacity=32, episode_len=200, warm_start=True, history=jobs, machine_gpus=8)
    rng = np.random.default_rng(seed)
    for start in (0, 150, 290):
        env.reset(options={"start": start})
        done = False
        while not done:
            assert env.mfree.sum() == env.free and env.mfree.min() >= 0 and env.mfree.max() <= 8
            mask = env.action_masks()
            for slot, job in enumerate(env.queue[: env.window]):  # mask agrees with a direct check
                g = env.gpus[job]
                ok = (env.mfree >= g).any() if g <= 8 else (env.mfree == 8).sum() >= -(-g // 8)
                assert mask[slot] == ok
            _, _, done, trunc, _ = env.step(int(rng.choice(np.flatnonzero(mask))))
            done = done or trunc
        assert env.n_finished == env.episode_len
