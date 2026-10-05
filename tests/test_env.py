"""Hand-checkable tests. Run with:  python -m pytest -q"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gpusched import GPUSchedEnv, add_duration_estimates, parse_philly_job_log  # noqa: E402
from gpusched.baselines import fifo, run_episode, sjf_oracle  # noqa: E402


def tiny_jobs():
    """4 jobs on a 4-GPU cluster. Worked out by hand below."""
    return pd.DataFrame({
        "job_id": ["J0", "J1", "J2", "J3"],
        "user": ["a", "b", "c", "d"],
        "vc": "test",
        "status": "Pass",
        "submit_time": [0.0, 10.0, 20.0, 30.0],
        "trace_end_time": [100.0, 30.0, 70.0, 40.0],
        "gpus": [4, 1, 2, 4],
        "duration": [100.0, 20.0, 50.0, 10.0],
        "est_duration": [100.0, 20.0, 50.0, 10.0],
    })


def make_env(**kw):
    return GPUSchedEnv(tiny_jobs(), capacity=4, window=10, episode_len=4, **kw)


def test_fifo_by_hand():
    # J0 runs 0-100 (all 4 GPUs). At t=100: J1 starts (1 GPU, ends 120), J2 starts
    # (2 GPUs, ends 150). J3 needs 4 GPUs: waits until J2 ends at 150, runs 150-160.
    # Waits: 0, 90, 80, 120 -> mean 72.5 s.  JCTs: 100, 110, 130, 130 -> mean 117.5 s.
    env = make_env()
    m, jobs = run_episode(env, fifo, start=0)
    np.testing.assert_allclose(env.start_time, [0, 100, 100, 150])
    np.testing.assert_allclose(env.finish_time, [100, 120, 150, 160])
    assert m["avg_wait_h"] * 3600 == pytest.approx(72.5)
    assert m["avg_jct_h"] * 3600 == pytest.approx(117.5)
    # Busy GPU-seconds = 4*100 + 1*20 + 2*50 + 4*10 = 560; makespan 160 -> 560 / 640
    assert m["utilization"] == pytest.approx(560 / 640)


def test_sjf_oracle_by_hand():
    # At t=100 all three wait. Shortest is J3 (10 s): runs 100-110 on all GPUs.
    # At t=110: J1 (20 s) then J2 (50 s) start. Waits: 0, 100, 90, 70 -> mean 65 s.
    env = make_env()
    m, _ = run_episode(env, sjf_oracle, start=0)
    np.testing.assert_allclose(env.start_time, [0, 110, 110, 100])
    assert m["avg_wait_h"] * 3600 == pytest.approx(65.0)


def test_dense_reward_sums_to_total_wait():
    scale = 7.0
    env = make_env(reward_scale=scale)
    m, _ = run_episode(env, fifo, start=0)
    total_wait = 72.5 * 4
    assert m["episode_reward"] == pytest.approx(-total_wait / scale)


def test_masks_and_invalid_action():
    env = make_env()
    env.reset(options={"start": 0})
    mask = env.action_masks()
    assert mask[0] and mask[env.WAIT]
    env.step(0)  # start J0 -> cluster full, next decision at t=100
    assert env.now == pytest.approx(100.0)
    # Slot 9 is empty, so action 9 is invalid: it is replaced and penalised.
    _, reward, *_ = env.step(9)
    assert env.invalid_actions == 1
    assert reward <= -env.invalid_penalty


def test_obs_in_bounds():
    env = make_env()
    obs, _ = env.reset(options={"start": 0})
    assert env.observation_space.contains(obs)


def write_sample_philly_log(path):
    jobs = [
        {   # retried job: 8 GPUs, attempts of 74 s and 2*86400+5*3600+39*60+42 s
            "status": "Pass", "vc": "vc1", "jobid": "app_1", "user": "u1",
            "submitted_time": "2017-10-07 01:11:39",
            "attempts": [
                {"start_time": "2017-10-07 01:12:09", "end_time": "2017-10-07 01:13:23",
                 "detail": [{"ip": "m47", "gpus": [f"gpu{i}" for i in range(8)]}]},
                {"start_time": "2017-10-07 01:13:30", "end_time": "2017-10-09 06:53:12",
                 "detail": [{"ip": "m412", "gpus": [f"gpu{i}" for i in range(8)]}]},
            ]},
        {"status": "Killed", "vc": "vc1", "jobid": "app_2", "user": "u1",
         "submitted_time": "2017-10-07 02:00:00", "attempts": []},                  # no attempts
        {"status": "Pass", "vc": "vc2", "jobid": "app_3", "user": "u2",
         "submitted_time": "2017-10-07 03:00:00",
         "attempts": [{"start_time": "2017-10-07 03:00:10", "end_time": None,
                       "detail": [{"ip": "m1", "gpus": ["gpu0"]}]}]},                # still running
        {"status": "Failed", "vc": "vc2", "jobid": "app_4", "user": "u2",
         "submitted_time": "2017-10-07 04:00:00",
         "attempts": [{"start_time": "2017-10-07 04:00:00", "end_time": "2017-10-07 04:10:00",
                       "detail": [{"ip": "m1", "gpus": ["gpu0", "gpu1"]},
                                  {"ip": "m2", "gpus": ["gpu0", "gpu1"]}]}]},        # 4 GPUs, 2 servers
    ]
    path.write_text(json.dumps(jobs))


def test_philly_parser(tmp_path):
    log = tmp_path / "cluster_job_log"
    write_sample_philly_log(log)
    df, dropped = parse_philly_job_log(log)
    assert list(df["job_id"]) == ["app_1", "app_4"]
    assert dropped["no_attempts"] == 1 and dropped["still_running"] == 1
    assert df.loc[0, "gpus"] == 8
    assert df.loc[0, "duration"] == pytest.approx(74 + 2 * 86400 + 5 * 3600 + 39 * 60 + 42)
    assert df.loc[1, "gpus"] == 4
    assert df.loc[1, "duration"] == pytest.approx(600)
    assert df.loc[0, "submit_time"] == 0.0


def test_estimates_never_use_the_future():
    # Same user. a: submit 0, runs 100 s (ends 100). b: submit 50, runs 50 s.
    # c: submit 200. Trace end times are what really happened in the cluster.
    df = pd.DataFrame({
        "job_id": ["a", "b", "c"], "user": ["u", "u", "u"], "vc": "x", "status": "Pass",
        "submit_time": [0.0, 50.0, 200.0],
        "trace_end_time": [100.0, 110.0, 300.0],
        "gpus": [1, 1, 1], "duration": [100.0, 50.0, 100.0],
    })
    out = add_duration_estimates(df, default_estimate=42.0)
    # a has no history -> default.
    assert out.loc[0, "est_source"] == "default"
    # b is submitted at t=50, before a finishes (t=100): still no history.
    assert out.loc[1, "est_source"] == "default" and out.loc[1, "est_duration"] == 42.0
    # c (t=200) sees a and b, both finished -> mean(100, 50) = 75. It never sees itself.
    assert out.loc[2, "est_source"] == "user" and out.loc[2, "est_duration"] == pytest.approx(75.0)


# ---------------------------------------------------------------- warm start
def warm_case():
    """4-GPU cluster. Episode = E0, E1. History adds two older jobs.

    H  : submit 0, 2 GPUs, runs 100 s -> still running at t=50 (ends abs 100)
    H2 : submit 1, 4 GPUs, long        -> still running, but does not fit after H
    E0 : submit 50, 4 GPUs, 10 s       (episode time 0)
    E1 : submit 60, 2 GPUs, 20 s       (episode time 10)
    """
    episode = pd.DataFrame({
        "job_id": ["E0", "E1"], "user": ["a", "b"], "vc": "t", "status": "Pass",
        "submit_time": [50.0, 60.0], "trace_end_time": [60.0, 80.0],
        "gpus": [4, 2], "duration": [10.0, 20.0], "est_duration": [10.0, 20.0],
    })
    older = pd.DataFrame({
        "job_id": ["H", "H2"], "user": ["c", "d"], "vc": "t", "status": "Pass",
        "submit_time": [0.0, 1.0], "trace_end_time": [100.0, 1001.0],
        "gpus": [2, 4], "duration": [100.0, 1000.0], "est_duration": [100.0, 1000.0],
    })
    return episode, pd.concat([older, episode], ignore_index=True)


def test_warm_start_fifo_by_hand():
    # H holds 2 GPUs until episode time 50; H2 is skipped (overflow).
    # t=0: E0 needs 4, only 2 free. t=10: E1 arrives; strict FIFO waits for E0.
    # t=50: H ends -> E0 runs 50-60. t=60: E1 runs 60-80. Waits 50, 50.
    # Busy GPU-seconds: 2*50 (H) + 4*10 (E0) + 2*20 (E1) = 180 over 80 s on 4 GPUs.
    episode, history = warm_case()
    env = GPUSchedEnv(episode, capacity=4, episode_len=2, warm_start=True, history=history)
    m, _ = run_episode(env, fifo, start=0)
    assert env.bg_jobs_at_start == 1 and env.bg_gpus_at_start == 2 and env.bg_overflow_jobs == 1
    np.testing.assert_allclose(env.start_time, [50, 60])
    assert m["avg_wait_h"] * 3600 == pytest.approx(50.0)
    assert m["utilization"] == pytest.approx(180 / (4 * 80))


def test_warm_start_first_fit_by_hand():
    # Same case with first-fit: at t=10 E1 fits (2 free) and runs 10-30.
    # E0 must wait for H to end at t=50 and runs 50-60. Waits 50, 0.
    from gpusched.baselines import fifo_ff
    episode, history = warm_case()
    env = GPUSchedEnv(episode, capacity=4, episode_len=2, warm_start=True, history=history)
    m, _ = run_episode(env, fifo_ff, start=0)
    np.testing.assert_allclose(env.start_time, [50, 10])
    assert m["avg_wait_h"] * 3600 == pytest.approx(25.0)


def test_cold_start_unchanged():
    # Without warm start the cluster starts empty: no waiting at all.
    episode, _ = warm_case()
    env = GPUSchedEnv(episode, capacity=4, episode_len=2)
    m, _ = run_episode(env, fifo, start=0)
    assert env.bg_gpus_at_start == 0
    assert m["avg_wait_h"] == pytest.approx(0.0)


def test_paired_bootstrap_by_hand():
    # Policy is exactly 1 hour worse than the reference in every episode:
    # difference 1 with a zero-width CI; total 10 vs 7 -> +42.9%; never better.
    from gpusched import paired_bootstrap
    r = paired_bootstrap([2.0, 3.0, 5.0], [1.0, 2.0, 4.0])
    assert r["mean_diff"] == pytest.approx(1.0)
    assert r["diff_ci_lo"] == pytest.approx(1.0) and r["diff_ci_hi"] == pytest.approx(1.0)
    assert r["pct_change"] == pytest.approx(100 * (10 / 7 - 1))
    assert r["frac_better"] == 0.0
