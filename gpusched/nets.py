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


class SlotScorer(nn.Module):
    def __init__(self, window=10, hidden=(64, 64)):
        super().__init__()
        self.window = window
        self.slot_net = mlp(SLOT_FEATURES + GLOBAL_FEATURES, hidden, 1)
        self.wait_net = mlp(SLOT_FEATURES + GLOBAL_FEATURES, hidden, 1)

    def forward(self, obs):
        b = obs.shape[0]
        n = self.window * SLOT_FEATURES
        slots = obs[:, :n].reshape(b, self.window, SLOT_FEATURES)
        glob = obs[:, n:n + GLOBAL_FEATURES]
        x = torch.cat([slots, glob.unsqueeze(1).expand(b, self.window, GLOBAL_FEATURES)], dim=2)
        job_scores = self.slot_net(x).squeeze(-1)                       # (b, window)
        filled = slots[:, :, :1]                                         # slot-filled flag
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
        self.adv = SlotScorer(window, hidden)

    def forward(self, obs):
        a = self.adv(obs)
        return self.value(obs) + a - a.mean(dim=1, keepdim=True)


def make_q_network(arch, obs_dim, n_actions, hidden):
    if arch == "slot":
        return SlotScorer(window=n_actions - 1, hidden=tuple(hidden))
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
            self.action_net = SlotScorer(self.action_space.n - 1, self._slot_hidden)
            # rebuild the optimizer so it trains the new action head
            self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)
