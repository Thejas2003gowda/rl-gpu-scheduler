"""Hand-checked tests for run-time features and predictor scoring."""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched.predict import FEATURES, history_features, pair_order_accuracy  # noqa: E402

DAY0 = 1_500_000_000.0  # an arbitrary wall-clock origin (seconds since 1970, UTC)


def three_jobs():
    """One user. a: submit 0, 100 s, ends 100. b: submit 50, 50 s, ends 100.5, failed.
    c: submit 200, 2 GPUs. Worked out by hand in the tests."""
    return pd.DataFrame({
        "job_id": ["a", "b", "c"], "user": ["u", "u", "u"], "vc": "x",
        "status": ["Pass", "Failed", "Pass"],
        "submit_time": [0.0, 50.0, 200.0], "trace_end_time": [100.0, 100.5, 300.0],
        "gpus": [1, 1, 2], "duration": [100.0, 50.0, 100.0],
        "submit_unix": [DAY0, DAY0 + 50, DAY0 + 200],
    })


def col(f, name):
    return f[name].to_numpy()


def test_features_by_hand():
    f = history_features(three_jobs())
    # a: nothing known yet
    assert np.isnan(f.loc[0, "u_mean"]) and f.loc[0, "u_n"] == 0
    assert np.isnan(f.loc[0, "log_since_prev"])
    # b (t=50): a was submitted but has not finished -> no run-time history,
    # one of the user's jobs still running, previous submission 50 s ago.
    assert f.loc[1, "u_n"] == 0 and np.isnan(f.loc[1, "u_mean"])
    assert f.loc[1, "u_running"] == pytest.approx(math.log1p(1))
    assert f.loc[1, "log_since_prev"] == pytest.approx(math.log1p(50))
    assert f.loc[1, "sub_1h"] == pytest.approx(math.log1p(1))
    # c (t=200): a and b finished. b finished last (100.5), so it is "last 1".
    lo_a, lo_b = math.log(100), math.log(50)
    assert f.loc[2, "u_n"] == pytest.approx(math.log1p(2))
    assert f.loc[2, "u_mean"] == pytest.approx((lo_a + lo_b) / 2)
    assert f.loc[2, "u_median"] == pytest.approx((lo_a + lo_b) / 2)
    assert f.loc[2, "u_std"] == pytest.approx(abs(lo_a - lo_b) / 2)
    assert f.loc[2, "u_last1"] == pytest.approx(lo_b)
    assert f.loc[2, "u_fail_frac"] == pytest.approx(0.5)
    assert f.loc[2, "u_running"] == pytest.approx(0.0)
    # c asks for 2 GPUs; the user never ran a 2-GPU job before.
    assert f.loc[2, "ug_n"] == 0 and np.isnan(f.loc[2, "ug_mean"])
    assert f.loc[2, "log_gpus"] == pytest.approx(math.log(2))


def test_features_never_use_the_future():
    """Changing a job's own outcome, or anything about later jobs, must not
    change the features of that job or of any earlier job."""
    rng = np.random.default_rng(0)
    n = 300
    submit = np.sort(rng.uniform(0, 50_000, n))
    dur = np.exp(rng.normal(6, 2, n))
    df = pd.DataFrame({
        "job_id": [f"j{i}" for i in range(n)], "user": rng.choice(list("abcde"), n), "vc": "x",
        "status": rng.choice(["Pass", "Failed", "Killed"], n), "submit_time": submit,
        "trace_end_time": submit + dur, "gpus": rng.choice([1, 2, 4, 8], n), "duration": dur,
        "submit_unix": DAY0 + submit,
    })
    base = history_features(df)
    k = 150
    changed = df.copy()
    changed.loc[k:, "duration"] *= 7.0          # job k and all later jobs run longer
    changed.loc[k:, "trace_end_time"] = changed.loc[k:, "submit_time"] + changed.loc[k:, "duration"]
    changed.loc[k:, "status"] = "Killed"
    after = history_features(changed)
    np.testing.assert_array_equal(base.iloc[: k + 1].to_numpy(), after.iloc[: k + 1].to_numpy())
    assert list(base.columns) == FEATURES


def test_pair_order_accuracy_by_hand():
    true = np.log([10, 100, 1000])
    assert pair_order_accuracy(true, true, window=10) == 1.0
    assert pair_order_accuracy(-true, true, window=10) == 0.0
    # constant prediction: every pair is a tie -> half right
    assert pair_order_accuracy(np.zeros(3), true, window=10) == 0.5
    # window=2 only looks at neighbours: (10,100) right, (100,1000) wrong
    assert pair_order_accuracy(np.log([10, 1000, 100]), true, window=2) == 0.5


# ---------------------------------------------------------------- RL statistics
def test_success_test_by_hand():
    from gpusched.stats import change_per_seed, gap_recovered, success_test
    ref = np.array([10.0, 10.0])
    pol = np.array([[9.0, 9.0], [8.0, 10.0], [11.0, 11.0], [9.0, 10.0]])
    # per seed: -10%, -10%, +10%, -5%  -> IQM of the middle two (-10, -5) = -7.5
    np.testing.assert_allclose(change_per_seed(pol, ref), [-10, -10, 10, -5])
    r = success_test(pol, ref, n_boot=200)
    assert r["iqm_change_pct"] == pytest.approx(-7.5)
    assert r["ci_lo"] <= r["iqm_change_pct"] <= r["ci_hi"]
    # an agent that always waits 20% less than the reference succeeds
    assert success_test(0.8 * np.tile(ref, (5, 1)), ref, n_boot=200)["success"]
    # gap: ref total 20, oracle total 10, policy total 15 -> half the gap closed
    g = gap_recovered(np.array([[7.5, 7.5]]), ref, np.array([5.0, 5.0]), n_boot=200)
    assert g["share"] == pytest.approx(0.5)
