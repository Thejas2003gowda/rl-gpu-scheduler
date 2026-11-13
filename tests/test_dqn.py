"""Tests for the own DQN implementation (gpusched/dqn.py)."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched import GPUSchedEnv  # noqa: E402
from gpusched.baselines import run_episode, sjf_oracle  # noqa: E402
from gpusched.dqn import DQNConfig, MaskedDQN, masked_argmax  # noqa: E402


class Fixed(torch.nn.Module):
    """A 'network' that always returns the same Q-values (for hand-checked targets)."""

    def __init__(self, values):
        super().__init__()
        self.values = torch.tensor(values, dtype=torch.float32)

    def forward(self, x):
        return self.values.expand(x.shape[0], -1)


def test_masked_argmax_never_invalid():
    g = torch.Generator().manual_seed(0)
    q = torch.randn(500, 11, generator=g)
    mask = torch.rand(500, 11, generator=g) < 0.3
    mask[:, 10] = True  # at least one valid action per row
    a = masked_argmax(q, mask)
    assert mask[torch.arange(500), a].all()
    # best VALID action, not the best overall
    q1 = torch.tensor([[9.0, 1.0, 5.0]])
    assert masked_argmax(q1, torch.tensor([[False, True, True]])).item() == 2


def test_double_dqn_target_by_hand():
    # Next state: online Q = [4, 5, 3], target Q = [10, 20, 30], action 1 invalid.
    # Double DQN: online picks best valid = action 0 (4 > 3) -> target Q 10.
    # Plain DQN: best valid target Q = 30. Terminal transition: no bootstrap.
    agent = MaskedDQN(3, 3, DQNConfig(gamma=0.5, double=True))
    agent.q, agent.q_target = Fixed([[4.0, 5.0, 3.0]]), Fixed([[10.0, 20.0, 30.0]])
    r = torch.tensor([1.0, 1.0])
    nxt = torch.zeros(2, 3)
    done = torch.tensor([0.0, 1.0])
    mask = torch.tensor([[True, False, True]] * 2)
    np.testing.assert_allclose(agent.targets(r, nxt, done, mask).numpy(), [1 + 0.5 * 10, 1.0])
    agent.cfg.double = False
    np.testing.assert_allclose(agent.targets(r, nxt, done, mask).numpy(), [1 + 0.5 * 30, 1.0])


def test_epsilon_greedy_only_valid_actions():
    agent = MaskedDQN(4, 5, seed=1)
    mask = np.array([False, True, False, True, False])
    acts = {agent.act(np.zeros(4, np.float32), mask, epsilon=1.0) for _ in range(200)}
    assert acts == {1, 3}
    assert agent.act(np.zeros(4, np.float32), mask, epsilon=0.0) in (1, 3)


def toy_jobs(n_groups, seed):
    """1-GPU cluster. Each group: a blocker B (10 s) arrives at t=0; a long job L
    (100 s) and a short job S (1 s) arrive at t=1 and t=2, in random order.
    Starting B at once and then S gives SJF's average wait of 6 s. Keeping the GPU
    free until S arrives and running S first is better (4 s if S comes first,
    5 s otherwise): the deliberate WAIT that greedy rules never make."""
    rng = np.random.default_rng(seed)
    rows = []
    for k in range(n_groups):
        base = 1000.0 * k
        order = [("L", 100.0), ("S", 1.0)] if rng.random() < 0.5 else [("S", 1.0), ("L", 100.0)]
        for name, t, dur in [("B", 0.0, 10.0), (order[0][0], 1.0, order[0][1]), (order[1][0], 2.0, order[1][1])]:
            rows.append({"job_id": f"{name}{k}", "user": "u", "vc": "toy", "status": "Pass",
                         "submit_time": base + t, "trace_end_time": base + t + dur, "gpus": 1,
                         "duration": dur, "est_duration": dur})
    return pd.DataFrame(rows)


def best_possible_wait(env):
    """Exhaustive search over every action sequence from the env's current state."""
    best = np.inf
    for a in np.flatnonzero(env.action_masks()):
        e = copy.deepcopy(env)
        _, _, term, trunc, info = e.step(int(a))
        best = min(best, info["metrics"]["avg_wait_h"] if term else best_possible_wait(e))
    return best


def test_toy_optimum_beats_sjf_by_waiting():
    env = GPUSchedEnv(toy_jobs(4, 1), capacity=1, episode_len=3, duration_info="oracle")
    for k in range(4):
        env.reset(options={"start": 3 * k})
        s_first = env.dur[1] == 1.0
        assert best_possible_wait(env) * 3600 == pytest.approx(4.0 if s_first else 5.0)
        assert run_episode(env, sjf_oracle, start=3 * k)[0]["avg_wait_h"] * 3600 == pytest.approx(6.0)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_dqn_learns_toy_schedule(seed):
    """With true run times visible, DQN must do at least as well as SJF with true
    run times on average (6 s). Untrained or random policies average 22-50 s.
    Over 20 checked runs (10 seeds, two settings) DQN averaged 4.5-6.0 s, often
    finding the optimum (4.5 s) by holding the GPU for the short job. It is not
    required to win every single episode: the exact policy learned varies with
    the seed and with the machine's floating-point details."""
    torch.set_num_threads(1)
    n_train, n_test = 200, 40
    env = GPUSchedEnv(toy_jobs(n_train, 0), capacity=1, episode_len=3, duration_info="oracle",
                      reward_scale=10.0)
    # gamma = 1: with a per-step discount below 1, WAIT pushes future costs one
    # step later and so looks cheaper than it is; the agent learns to stall.
    cfg = DQNConfig(hidden=[64, 64], lr=5e-4, gamma=1.0, buffer_size=20_000, batch_size=64,
                    learning_starts=200, train_freq=1, target_update_interval=250,
                    eps_fraction=0.4, eps_end=0.01, reward_multiplier=1.0)
    agent = MaskedDQN(env.observation_space.shape[0], env.action_space.n, cfg, seed=seed)
    agent.learn(env, 10_000, reset_options=lambda rng: {"start": 3 * int(rng.integers(n_train))})

    test_env = GPUSchedEnv(toy_jobs(n_test, 1), capacity=1, episode_len=3, duration_info="oracle")
    dqn, sjf = [], []
    for k in range(n_test):
        dqn.append(run_episode(test_env, lambda e: agent.predict(e._obs(), e.action_masks()), start=3 * k)[0]["avg_wait_h"])
        sjf.append(run_episode(test_env, sjf_oracle, start=3 * k)[0]["avg_wait_h"])
    assert np.mean(dqn) <= np.mean(sjf) + 1e-9


def test_slot_scorer_moves_score_with_the_job():
    """Swapping two jobs between queue slots swaps their scores; WAIT's score
    does not change (it sees only the cluster and the average job)."""
    from gpusched.nets import SlotScorer
    torch.manual_seed(0)
    net = SlotScorer(window=10, hidden=(16, 16))
    obs = torch.rand(1, 52)
    obs[0, 0:50:5] = 1.0  # all slots filled
    swapped = obs.clone()
    swapped[0, 5:10], swapped[0, 15:20] = obs[0, 15:20], obs[0, 5:10]  # swap slots 1 and 3
    a, b = net(obs)[0], net(swapped)[0]
    assert a.shape == (11,)
    assert torch.allclose(a[1], b[3]) and torch.allclose(a[3], b[1])
    assert torch.allclose(a[[0, 2, 4, 5, 6, 7, 8, 9]], b[[0, 2, 4, 5, 6, 7, 8, 9]])
    assert torch.allclose(a[10], b[10], atol=1e-6)


def test_slot_ppo_policy_builds_and_masks():
    pytest.importorskip("sb3_contrib")
    from sb3_contrib import MaskablePPO
    from gpusched.nets import SlotPolicy
    env = GPUSchedEnv(toy_jobs(5, 0), capacity=1, episode_len=3, duration_info="oracle")
    model = MaskablePPO(SlotPolicy, env, n_steps=32, batch_size=16, seed=0, device="cpu",
                        policy_kwargs={"net_arch": {"pi": [], "vf": [16]}, "slot_hidden": [16]})
    assert isinstance(model.policy.action_net, torch.nn.Module)
    env.reset(options={"start": 0})
    mask = env.action_masks()
    for _ in range(20):
        a, _ = model.predict(env._obs(), action_masks=mask, deterministic=False)
        assert mask[int(a)]
    model.learn(64)  # trains without errors


def test_dueling_q_is_value_plus_centred_advantage():
    from gpusched.nets import DuelingSlotQ
    torch.manual_seed(0)
    net = DuelingSlotQ(52, window=10, hidden=(16, 16), value_hidden=(16,))
    obs = torch.rand(3, 52)
    q = net(obs)
    a = net.adv(obs)
    expected = net.value(obs) + a - a.mean(dim=1, keepdim=True)
    assert q.shape == (3, 11) and torch.allclose(q, expected)
    assert torch.allclose(q.mean(dim=1, keepdim=True), net.value(obs), atol=1e-6)  # average Q = V(s)


def test_n_step_transitions_by_hand():
    """3-step returns: R = r0 + g r1 + g^2 r2, bootstrapping from the state after
    3 steps with discount g^3; at the episode end the remaining steps are folded
    into shorter returns with done = 1."""
    from gpusched.dqn import ReplayBuffer

    class Line:  # 4-step chain, reward -1 per step, always 2 valid actions
        observation_space_dim = 2

        def reset(self, seed=None, options=None):
            self.t = 0
            return np.array([0.0, 0.0], np.float32), {}

        def step(self, a):
            self.t += 1
            return np.array([self.t, 0.0], np.float32), -1.0, self.t == 4, False, {}

        def action_masks(self):
            return np.array([True, True])

    cfg = DQNConfig(n_step=3, gamma=0.5, learning_starts=10**9, reward_multiplier=1.0,
                    target_update_interval=10**9, buffer_size=100)
    agent = MaskedDQN(2, 2, cfg, seed=0)
    captured = []
    orig = ReplayBuffer.add
    ReplayBuffer.add = lambda self, *a: (captured.append(a), orig(self, *a))[1]
    try:
        agent.learn(Line(), 4)
    finally:
        ReplayBuffer.add = orig
    starts = [c[0][0] for c in captured]
    rets = [c[2] for c in captured]
    nexts = [c[3][0] for c in captured]
    dones = [c[4] for c in captured]
    discs = [c[6] for c in captured]
    assert starts == [0, 1, 2, 3]
    np.testing.assert_allclose(rets, [-1.75, -1.75, -1.5, -1.0])   # -(1 + .5 + .25), ..., -(1 + .5), -1
    assert nexts == [3, 4, 4, 4] and dones == [False, True, True, True]
    np.testing.assert_allclose(discs, [0.125, 0.125, 0.25, 0.5])
