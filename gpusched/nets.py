"""Networks that score each queued job with the SAME small network.

A plain MLP on the flattened 52-value observation must learn "a short job is
good" separately for each of the 10 queue slots. Following RLScheduler (Zhang
et al., SC 2020; its "kernel" policy network), ``SlotScorer`` applies one shared
MLP to every slot (that slot's 5 features + the 2 cluster features) and gives
one score per slot; a second small MLP scores WAIT from the cluster features
and the average of the slot features. Moving a job to another slot moves its
score with it, so what is learned for one slot holds for all.

Used as the Q-network in DQN (scores = Q-values) and as the policy head in
MaskablePPO (scores = action logits).
"""
from __future__ import annotations

import torch
from torch import nn

SLOT_FEATURES, GLOBAL_FEATURES = 5, 2


def mlp(n_in, hidden, n_out):
    layers, last = [], n_in
    for h in hidden:
        layers += [nn.Linear(last, h), nn.ReLU()]
        last = h
    layers.append(nn.Linear(last, n_out))
    return nn.Sequential(*layers)


def extra_per_slot(obs_dim, window):
    """Extra inputs per queue slot beyond the standard 5 (0 for the 52-value observation;
    8 with duration_info="signals", where they follow the 2 global values)."""
    rest = obs_dim - window * SLOT_FEATURES - GLOBAL_FEATURES
    if rest < 0 or rest % window:
        raise ValueError(f"observation size {obs_dim} does not fit a {window}-slot layout")
    return rest // window


class SlotScorer(nn.Module):
    def __init__(self, window=10, hidden=(64, 64), extra=0):
        super().__init__()
        self.window, self.extra = window, int(extra)
        per_job = SLOT_FEATURES + self.extra
        self.slot_net = mlp(per_job + GLOBAL_FEATURES, hidden, 1)
        self.wait_net = mlp(per_job + GLOBAL_FEATURES, hidden, 1)

    def forward(self, obs):
        b = obs.shape[0]
        n = self.window * SLOT_FEATURES
        slots = obs[:, :n].reshape(b, self.window, SLOT_FEATURES)
        glob = obs[:, n:n + GLOBAL_FEATURES]
        filled = slots[:, :, :1]                                         # slot-filled flag
        if self.extra:
            e = n + GLOBAL_FEATURES
            extra = obs[:, e:e + self.window * self.extra].reshape(b, self.window, self.extra)
            slots = torch.cat([slots, extra], dim=2)
        x = torch.cat([slots, glob.unsqueeze(1).expand(b, self.window, GLOBAL_FEATURES)], dim=2)
        job_scores = self.slot_net(x).squeeze(-1)                       # (b, window)
        pooled = (slots * filled).sum(1) / filled.sum(1).clamp(min=1.0)  # mean over filled slots
        wait_score = self.wait_net(torch.cat([pooled, glob], dim=1))     # (b, 1)
        return torch.cat([job_scores, wait_score], dim=1)               # (b, window + 1)


class DuelingSlotQ(nn.Module):
    """Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)  (dueling DQN, Wang et al., 2016).

    DQN needs Q-values on an absolute scale, but SlotScorer sees one job at a
    time and cannot know the cost of the whole queue. Here a plain MLP on the
    full observation estimates the state's value V(s), and SlotScorer only
    supplies how much better each action is than average (A). PPO does not
    need this: its scores are only compared with each other."""

    def __init__(self, obs_dim, window=10, hidden=(64, 64), value_hidden=(128, 128)):
        super().__init__()
        self.value = mlp(obs_dim, value_hidden, 1)
        self.adv = SlotScorer(window, hidden, extra=extra_per_slot(obs_dim, window))

    def forward(self, obs):
        a = self.adv(obs)
        return self.value(obs) + a - a.mean(dim=1, keepdim=True)


def make_q_network(arch, obs_dim, n_actions, hidden):
    if arch == "slot":
        return SlotScorer(window=n_actions - 1, hidden=tuple(hidden),
                          extra=extra_per_slot(obs_dim, n_actions - 1))
    if arch == "slot_dueling":
        return DuelingSlotQ(obs_dim, window=n_actions - 1, hidden=tuple(hidden))
    if arch == "mlp":
        return mlp(obs_dim, hidden, n_actions)
    raise ValueError("arch must be 'mlp', 'slot' or 'slot_dueling'")


try:  # MaskablePPO policy (needs requirements-rl.txt)
    from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
except ImportError:  # pragma: no cover
    MaskableActorCriticPolicy = None

if MaskableActorCriticPolicy is not None:
    class SlotPolicy(MaskableActorCriticPolicy):
        """MaskablePPO policy whose action logits come from SlotScorer.

        Use with net_arch=dict(pi=[], vf=[...]): latent_pi is then the raw
        observation, and action_net (replaced here) turns it into slot scores.
        The value function stays a plain MLP on the whole observation."""

        def __init__(self, *args, slot_hidden=(64, 64), **kwargs):
            self._slot_hidden = tuple(slot_hidden)
            super().__init__(*args, **kwargs)

        def _build(self, lr_schedule):
            super()._build(lr_schedule)
            window = self.action_space.n - 1
            extra = extra_per_slot(self.observation_space.shape[0], window)
            self.action_net = SlotScorer(window, self._slot_hidden, extra=extra)
            # rebuild the optimizer so it trains the new action head
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
