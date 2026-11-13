"""Tests for the run-time signals given to the agent directly (duration_info="signals")."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched import GPUSchedEnv  # noqa: E402
from gpusched.env import GLOBAL_FEATURES, SLOT_FEATURES  # noqa: E402
from gpusched.predict import AGENT_SIGNALS, add_agent_signals, agent_signal_features  # noqa: E402
from gpusched.stats import difference_test  # noqa: E402

DAY0 = 1_500_000_000.0


def trace_like_jobs(n=300, seed=0):
    rng = np.random.default_rng(seed)
    submit = np.sort(rng.uniform(0, 50_000, n))
    dur = np.exp(rng.normal(6, 2, n))
    return pd.DataFrame({
        "job_id": [f"j{i}" for i in range(n)], "user": rng.choice(list("abcde"), n), "vc": "x",
        "status": rng.choice(["Pass", "Failed", "Killed"], n), "submit_time": submit,
        "trace_end_time": submit + dur, "gpus": rng.choice([1, 2, 4, 8], n), "duration": dur,
        "submit_unix": DAY0 + submit, "est_duration": np.full(n, 600.0),
    })


def test_signals_in_range_and_never_use_the_future():
    df = trace_like_jobs()
    base = agent_signal_features(df)
    assert list(base.columns) == AGENT_SIGNALS
    assert np.isfinite(base.to_numpy()).all()
    assert (base.to_numpy() >= 0).all() and (base.to_numpy() <= 1).all()
    k = 150
    changed = df.copy()
    changed.loc[k:, "duration"] *= 7.0          # job k and all later jobs run longer and fail
    changed.loc[k:, "trace_end_time"] = changed.loc[k:, "submit_time"] + changed.loc[k:, "duration"]
    changed.loc[k:, "status"] = "Killed"
    after = agent_signal_features(changed)
    np.testing.assert_array_equal(base.iloc[: k + 1].to_numpy(), after.iloc[: k + 1].to_numpy())


def test_signals_observation_extends_the_standard_one():
    """The first 52 values are exactly the 'estimate' observation; the extra block
    holds each queued job's signals in slot order, and zeros for empty slots."""
    jobs = add_agent_signals(trace_like_jobs(60, 1))
    sig_env = GPUSchedEnv(jobs, capacity=8, window=10, episode_len=40, duration_info="signals")
    est_env = GPUSchedEnv(jobs, capacity=8, window=10, episode_len=40, duration_info="estimate")
    n_std = 10 * SLOT_FEATURES + GLOBAL_FEATURES
    assert est_env.observation_space.shape == (n_std,)
    assert sig_env.observation_space.shape == (n_std + 10 * len(AGENT_SIGNALS),)
    sig_obs, _ = sig_env.reset(options={"start": 5})
    est_obs, _ = est_env.reset(options={"start": 5})
    sig_values = jobs[AGENT_SIGNALS].to_numpy(np.float32)[5:45]
    k = len(AGENT_SIGNALS)
    for _ in range(25):
        np.testing.assert_array_equal(sig_obs[:n_std], est_obs)
        extra = sig_obs[n_std:].reshape(10, k)
        for slot in range(10):
            if slot < len(sig_env.queue):
                np.testing.assert_allclose(extra[slot], sig_values[sig_env.queue[slot]], rtol=0, atol=1e-7)
            else:
                assert not extra[slot].any()
        a = int(np.flatnonzero(sig_env.action_masks())[0])
        sig_obs, _, done, trunc, _ = sig_env.step(a)
        est_obs, _, _, _, _ = est_env.step(a)
        if done or trunc:
            break


def test_signals_need_their_columns():
    with pytest.raises(ValueError):
        GPUSchedEnv(trace_like_jobs(60), capacity=8, episode_len=40, duration_info="signals")


def test_slot_scorer_with_signals_moves_score_with_the_job():
    torch = pytest.importorskip("torch")
    from gpusched.nets import SlotScorer, extra_per_slot
    k = len(AGENT_SIGNALS)
    assert extra_per_slot(52, 10) == 0 and extra_per_slot(52 + 10 * k, 10) == k
    with pytest.raises(ValueError):
        extra_per_slot(53, 10)
    torch.manual_seed(0)
    net = SlotScorer(window=10, hidden=(16, 16), extra=k)
    obs = torch.rand(1, 52 + 10 * k)
    obs[0, 0:50:5] = 1.0  # all slots filled
    swapped = obs.clone()
    swapped[0, 5:10], swapped[0, 15:20] = obs[0, 15:20], obs[0, 5:10]       # slots 1 and 3
    e = 52
    s1, s3 = slice(e + 1 * k, e + 2 * k), slice(e + 3 * k, e + 4 * k)
    swapped[0, s1], swapped[0, s3] = obs[0, s3], obs[0, s1]                  # their signals too
    a, b = net(obs)[0], net(swapped)[0]
    assert torch.allclose(a[1], b[3]) and torch.allclose(a[3], b[1])
    assert torch.allclose(a[10], b[10], atol=1e-6)
    # the signals change the scores (they are really used)
    other = obs.clone()
    other[0, e:] = torch.rand(10 * k)
    assert not torch.allclose(net(other)[0, :10], a[:10])


def test_agents_build_on_the_signals_environment():
    torch = pytest.importorskip("torch")
    from gpusched.dqn import DQNConfig, MaskedDQN
    jobs = add_agent_signals(trace_like_jobs(60, 2))
    env = GPUSchedEnv(jobs, capacity=8, episode_len=40, duration_info="signals")
    obs, _ = env.reset(options={"start": 0})
    agent = MaskedDQN(env.observation_space.shape[0], env.action_space.n, DQNConfig(arch="slot", hidden=[16]))
    assert env.action_masks()[agent.predict(obs, env.action_masks())]
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO
    from gpusched.nets import SlotPolicy
    model = MaskablePPO(SlotPolicy, env, n_steps=32, batch_size=16, seed=0, device="cpu",
                        policy_kwargs={"net_arch": {"pi": [], "vf": [16]}, "slot_hidden": [16]})
    assert model.policy.action_net.extra == len(AGENT_SIGNALS)
    model.learn(64)


def test_difference_test():
    rng = np.random.default_rng(0)
    ref = rng.uniform(5, 10, 15)
    b = ref * rng.uniform(0.98, 1.02, (10, 15))
    better = difference_test(b * 0.9, b, ref, n_boot=2000)
    assert better["a_better"] and abs(better["difference_pct_points"] + 10) < 0.5
    same = difference_test(b, b, ref, n_boot=2000)
    assert not same["a_better"] and same["ci_lo"] < 0 < same["ci_hi"]
