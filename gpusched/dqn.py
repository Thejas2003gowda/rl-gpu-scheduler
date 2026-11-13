"""Deep Q-Network with invalid-action masking (own implementation, PyTorch).

What is in it, and why:
  * Double DQN targets (van Hasselt et al., 2016): the online network picks the
    next action, the target network scores it. Reduces over-estimation.
  * Masking everywhere: invalid actions are never chosen (epsilon-greedy picks
    among valid actions only, greedy takes the best valid action), and the
    target's max over next actions is over the NEXT state's valid actions only.
    That is why the replay buffer stores the next state's mask.
  * Target network with hard updates, Huber loss, gradient clipping, and a
    fixed reward multiplier (rewards here are heavy-tailed).
  * Optional n-step returns (``n_step``): the target adds the real rewards of
    the next n steps before bootstrapping, so less rests on the network's own
    (often optimistic) estimate of the future.
  * "done" in the target means the episode really ended (terminated). A
    truncated episode still bootstraps.

The environment must offer ``action_masks()`` (GPUSchedEnv does).
"""
from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .nets import make_q_network

NEG = -1e9  # value given to invalid actions before taking a max (finite, so 0 * NEG stays 0)


@dataclass
class DQNConfig:
    hidden: list = field(default_factory=lambda: [128, 128])
    lr: float = 3e-4
    gamma: float = 0.99
    buffer_size: int = 200_000
    batch_size: int = 128
    learning_starts: int = 5_000
    train_freq: int = 4              # environment steps between gradient steps
    target_update_interval: int = 2_000
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_fraction: float = 0.2        # share of training over which epsilon decays
    max_grad_norm: float = 10.0
    reward_multiplier: float = 0.1   # rewards are scaled by this before learning
    double: bool = True
    arch: str = "mlp"               # "mlp" (flat), "slot" or "slot_dueling" (gpusched/nets.py)
    n_step: int = 1                  # rewards of this many real steps before bootstrapping


def masked_argmax(q, mask):
    """Best valid action per row. q: (B, A) tensor, mask: (B, A) bool tensor."""
    return q.masked_fill(~mask, NEG).argmax(dim=1)


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, n_actions, seed=0):
        self.capacity = int(capacity)
        self.obs = np.zeros((self.capacity, obs_dim), np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), np.float32)
        self.action = np.zeros(self.capacity, np.int64)
        self.reward = np.zeros(self.capacity, np.float32)
        self.done = np.zeros(self.capacity, np.float32)
        self.discount = np.zeros(self.capacity, np.float32)
        self.next_mask = np.zeros((self.capacity, n_actions), bool)
        self.pos, self.full = 0, False
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return self.capacity if self.full else self.pos

    def add(self, obs, action, reward, next_obs, done, next_mask, discount):
        i = self.pos
        self.obs[i], self.action[i], self.reward[i] = obs, action, reward
        self.discount[i] = discount
        self.next_obs[i], self.done[i], self.next_mask[i] = next_obs, float(done), next_mask
        self.pos = (self.pos + 1) % self.capacity
        self.full = self.full or self.pos == 0

    def sample(self, batch_size):
        idx = self.rng.integers(0, len(self), size=batch_size)
        t = torch.as_tensor
        return (t(self.obs[idx]), t(self.action[idx]), t(self.reward[idx]),
                t(self.next_obs[idx]), t(self.done[idx]), t(self.next_mask[idx]), t(self.discount[idx]))


class MaskedDQN:
    def __init__(self, obs_dim, n_actions, config=None, seed=0):
        self.cfg = config or DQNConfig()
        self.obs_dim, self.n_actions, self.seed = int(obs_dim), int(n_actions), int(seed)
        torch.manual_seed(seed)
        self.q = make_q_network(self.cfg.arch, self.obs_dim, self.n_actions, self.cfg.hidden)
        self.q_target = make_q_network(self.cfg.arch, self.obs_dim, self.n_actions, self.cfg.hidden)
        self.q_target.load_state_dict(self.q.state_dict())
        self.opt = torch.optim.Adam(self.q.parameters(), lr=self.cfg.lr)
        self.rng = np.random.default_rng(seed)
        self.num_updates = 0

    # -------------------------------------------------------------- acting
    def predict(self, obs, mask):
        """Greedy valid action for one observation."""
        with torch.no_grad():
            q = self.q(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        return int(masked_argmax(q, torch.as_tensor(mask).unsqueeze(0))[0])

    def act(self, obs, mask, epsilon):
        if self.rng.random() < epsilon:
            return int(self.rng.choice(np.flatnonzero(mask)))
        return self.predict(obs, mask)

    def epsilon(self, step, total_steps):
        frac = min(step / max(self.cfg.eps_fraction * total_steps, 1), 1.0)
        return self.cfg.eps_start + frac * (self.cfg.eps_end - self.cfg.eps_start)

    # ------------------------------------------------------------ learning
    def targets(self, reward, next_obs, done, next_mask, discount=None):
        """R + discount * (1 - done) * Q_target(s', a*), a* = best VALID next action.
        R is the (n-step) reward sum; discount is gamma ** (steps taken), gamma by default."""
        with torch.no_grad():
            q_next_target = self.q_target(next_obs)
            if self.cfg.double:
                a_star = masked_argmax(self.q(next_obs), next_mask)
            else:
                a_star = masked_argmax(q_next_target, next_mask)
            q_next = q_next_target.gather(1, a_star.unsqueeze(1)).squeeze(1)
            disc = self.cfg.gamma if discount is None else discount
            return reward + disc * (1.0 - done) * q_next

    def train_step(self, batch):
        obs, action, reward, next_obs, done, next_mask, discount = batch
        target = self.targets(reward, next_obs, done, next_mask, discount)
        q_sa = self.q(obs).gather(1, action.unsqueeze(1)).squeeze(1)
        loss = nn.functional.smooth_l1_loss(q_sa, target)
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q.parameters(), self.cfg.max_grad_norm)
        self.opt.step()
        self.num_updates += 1
        return float(loss.detach())

    def learn(self, env, total_timesteps, callback=None, reset_options=None, log=None):
        """Train on ``env`` for ``total_timesteps`` environment steps.

        callback(step, agent) is called every step after the update; returning
        False stops training. reset_options(rng) -> dict lets the caller choose
        episode starts (default: the environment's own random start).
        log(dict) receives one record per finished episode.
        """
        cfg = self.cfg
        buf = ReplayBuffer(cfg.buffer_size, self.obs_dim, self.n_actions, seed=self.seed)

        def reset():
            opts = reset_options(self.rng) if reset_options else None
            o, _ = env.reset(seed=int(self.rng.integers(2**31)), options=opts)
            return o, env.action_masks()

        pending = deque()  # last n transitions, folded into n-step transitions

        def push_first():
            ret, disc = 0.0, 1.0
            for (_, _, r, _, _, _) in pending:
                ret += disc * r
                disc *= cfg.gamma
            o0, a0 = pending[0][0], pending[0][1]
            _, _, _, o_n, done_n, mask_n = pending[-1]
            buf.add(o0, a0, ret, o_n, done_n, mask_n, disc)
            pending.popleft()

        obs, mask = reset()
        ep_return, ep_len, episodes = 0.0, 0, 0
        for step in range(1, int(total_timesteps) + 1):
            action = self.act(obs, mask, self.epsilon(step, total_timesteps))
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = env.action_masks()
            pending.append((obs, action, reward * cfg.reward_multiplier, next_obs, terminated, next_mask))
            if len(pending) == cfg.n_step:
                push_first()
            obs, mask = next_obs, next_mask
            ep_return += reward
            ep_len += 1
            if terminated or truncated:
                while pending:  # flush: shorter returns at the end of an episode
                    push_first()
                episodes += 1
                if log is not None:
                    rec = {"step": step, "episode": episodes, "return": ep_return, "length": ep_len}
                    if "metrics" in info:
                        rec["avg_wait_h"] = info["metrics"]["avg_wait_h"]
                    log(rec)
                obs, mask = reset()
                ep_return, ep_len = 0.0, 0
            if step >= cfg.learning_starts and step % cfg.train_freq == 0:
                self.train_step(buf.sample(cfg.batch_size))
            if step % cfg.target_update_interval == 0:
                self.q_target.load_state_dict(self.q.state_dict())
            if callback is not None and callback(step, self) is False:
                break
        return self

    # ---------------------------------------------------------- save/load
    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"q": self.q.state_dict(), "config": asdict(self.cfg), "obs_dim": self.obs_dim,
                    "n_actions": self.n_actions, "seed": self.seed}, path)

    @classmethod
    def load(cls, path):
        d = torch.load(path, weights_only=False)
        agent = cls(d["obs_dim"], d["n_actions"], DQNConfig(**d["config"]), seed=d["seed"])
        agent.q.load_state_dict(d["q"])
        agent.q_target.load_state_dict(d["q"])
        return agent


def config_from_json(text_or_dict):
    d = json.loads(text_or_dict) if isinstance(text_or_dict, str) else dict(text_or_dict)
    return DQNConfig(**d)
