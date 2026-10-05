"""GPU cluster scheduling environment (Gymnasium API).

The cluster is a single pool of ``capacity`` GPUs. Jobs arrive at their trace
submit times. Each job needs all of its GPUs at once (gang scheduling) and
holds them for its duration. There is no preemption and no placement on
specific machines (locality is ignored; see README "Limitations").

Warm start (optional, ``warm_start=True`` with a ``history`` table): each
episode begins with the background jobs that, according to the trace, were
still running when the episode's first job arrived. They are assumed to have
started at their submit time (the trace does not give their real waits), are
added in submission order until the cluster is full (the rest are skipped and
counted in ``bg_overflow_jobs``), hold their GPUs until they finish, and are
NOT included in the episode's metrics or its end condition. Utilization does
count them, because they really occupy GPUs.

Decision points: the simulator jumps from event to event (arrivals and
completions). It stops and asks the agent for an action whenever at least one
job in the visible window fits in the free GPUs.

Action (Discrete(window + 1)):
    0 .. window-1 : start the job in that queue slot (queue is in arrival order)
    window        : WAIT until the next arrival or completion
Invalid actions are blocked by ``action_masks()`` (for MaskablePPO). If an
invalid action is still sent, it is replaced by WAIT (or the first valid job)
and a small penalty is added to the reward.

Observation (all values scaled to [0, 1]), per visible slot:
    [slot filled, GPUs requested / capacity, fits now,
     log duration feature, log time waited so far]
plus 3 global values:
    [free GPUs / capacity, log queue length, log jobs still to arrive]

The duration feature depends on ``duration_info``:
    "estimate" -> user's past average (realistic)
    "oracle"   -> true duration (impossible in practice; upper reference)
    "none"     -> always 0 (no duration information)

Reward:
    "wait" (dense): minus the total waiting time accrued by ALL queued jobs
                    since the last step, divided by ``reward_scale``. Summed
                    over an episode it equals minus total waiting time.
    "bsld" (sparse): 0 until the episode ends, then minus the average bounded
                    slowdown of the episode's jobs.
"""
from __future__ import annotations

import heapq
import math

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .metrics import episode_metrics

SLOT_FEATURES = 5
GLOBAL_FEATURES = 3
MAX_TIME = 30 * 24 * 3600.0  # 30 days, for log scaling
LOG_MAX_TIME = math.log1p(MAX_TIME)
DURATION_INFO = ("estimate", "oracle", "none")
REWARD_MODES = ("wait", "bsld")


class GPUSchedEnv(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self, jobs, capacity, window=10, episode_len=256, reward_mode="wait",
                 duration_info="estimate", reward_scale=36_000.0, invalid_penalty=0.1,
                 warm_start=False, history=None, render_mode=None):
        if reward_mode not in REWARD_MODES:
            raise ValueError(f"reward_mode must be one of {REWARD_MODES}")
        if duration_info not in DURATION_INFO:
            raise ValueError(f"duration_info must be one of {DURATION_INFO}")
        if "est_duration" not in jobs.columns:
            raise ValueError("jobs needs an 'est_duration' column; call add_duration_estimates first")

        jobs = jobs.sort_values("submit_time", kind="stable").reset_index(drop=True)
        too_big = jobs["gpus"] > capacity
        self.n_dropped_too_big = int(too_big.sum())
        jobs = jobs[~too_big].reset_index(drop=True)
        if len(jobs) < episode_len:
            raise ValueError(f"Only {len(jobs)} usable jobs; need at least episode_len={episode_len}")

        self.capacity = int(capacity)
        self.window = int(window)
        self.episode_len = int(episode_len)
        self.reward_mode = reward_mode
        self.duration_info = duration_info
        self.reward_scale = float(reward_scale)
        self.invalid_penalty = float(invalid_penalty)
        self.render_mode = render_mode
        self.WAIT = self.window
        self.max_steps = 10 * self.episode_len

        self._all_submit = jobs["submit_time"].to_numpy(float)
        self._all_gpus = jobs["gpus"].to_numpy(int)
        self._all_dur = jobs["duration"].to_numpy(float)
        self._all_est = jobs["est_duration"].to_numpy(float)
        self.n_jobs_total = len(jobs)

        self.warm_start = bool(warm_start)
        if self.warm_start:
            hist = jobs if history is None else history
            hist = hist.sort_values("submit_time", kind="stable")
            self._h_submit = hist["submit_time"].to_numpy(float)
            self._h_end = self._h_submit + hist["duration"].to_numpy(float)
            self._h_gpus = hist["gpus"].to_numpy(int)

        self.action_space = spaces.Discrete(self.window + 1)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0,
            shape=(self.window * SLOT_FEATURES + GLOBAL_FEATURES,), dtype=np.float32)

    # ------------------------------------------------------------------ API
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        start = options.get("start")
        if start is None:
            start = int(self.np_random.integers(0, self.n_jobs_total - self.episode_len + 1))
        if not 0 <= start <= self.n_jobs_total - self.episode_len:
            raise ValueError(f"start={start} out of range")
        self.episode_start = int(start)

        sl = slice(start, start + self.episode_len)
        self.submit = self._all_submit[sl] - self._all_submit[start]
        self.gpus = self._all_gpus[sl]
        self.dur = self._all_dur[sl]
        self.est = self._all_est[sl]
        n = self.episode_len
        self.start_time = np.full(n, np.nan)
        self.finish_time = np.full(n, np.nan)

        self.now = 0.0
        self.free = self.capacity
        self.queue = []      # job indices, arrival order; slot i = queue[i]
        self.running = []    # heap of (finish time, job index)
        self.next_arrival = 0
        self.n_finished = 0
        self.steps = 0
        self.invalid_actions = 0
        self.total_wait = 0.0
        self._accrued = 0.0
        self.busy_gpu_seconds = 0.0
        self.bg_gpus = []          # GPUs held by each background job
        self.bg_overflow_jobs = 0
        if self.warm_start:
            self._add_background(self._all_submit[start])
        self.bg_jobs_at_start = len(self.bg_gpus)
        self.bg_gpus_at_start = self.capacity - self.free

        self._admit_arrivals()
        self._advance_to_decision()
        self._accrued = 0.0
        return self._obs(), self._info()

    def step(self, action):
        action = int(action)
        mask = self.action_masks()
        invalid = not (0 <= action < len(mask) and mask[action])
        if invalid:
            self.invalid_actions += 1
            action = self.WAIT if mask[self.WAIT] else int(np.flatnonzero(mask)[0])

        self._accrued = 0.0
        if action == self.WAIT:
            self._jump_to_next_event()
        else:
            self._start_job(action)
        terminated = self._advance_to_decision()
        self.steps += 1
        truncated = (not terminated) and self.steps >= self.max_steps

        reward = 0.0
        if self.reward_mode == "wait":
            reward = -self._accrued / self.reward_scale
        elif terminated:
            reward = -self.metrics()["avg_bsld"]
        if invalid:
            reward -= self.invalid_penalty

        info = self._info()
        if terminated:
            info["metrics"] = self.metrics()
        return self._obs(), float(reward), terminated, truncated, info

    def action_masks(self):
        """Boolean mask of valid actions (used by sb3_contrib MaskablePPO)."""
        mask = np.zeros(self.window + 1, dtype=bool)
        for slot, job in enumerate(self.queue[: self.window]):
            mask[slot] = self.gpus[job] <= self.free
        mask[self.WAIT] = bool(self.running) or self.next_arrival < self.episode_len
        if not mask.any():  # only at the very end of an episode
            mask[self.WAIT] = True
        return mask

    def metrics(self):
        return episode_metrics(self.submit, self.start_time, self.finish_time,
                               self.gpus, self.dur, self.capacity,
                               busy_gpu_seconds=self.busy_gpu_seconds)

    def visible_jobs(self):
        """Jobs in the visible window, for rule-based baselines."""
        return [
            {"slot": slot, "job": j, "gpus": int(self.gpus[j]), "fits": bool(self.gpus[j] <= self.free),
             "true_duration": float(self.dur[j]), "est_duration": float(self.est[j]),
             "waited": float(self.now - self.submit[j])}
            for slot, j in enumerate(self.queue[: self.window])
        ]

    def render(self):
        if self.render_mode != "ansi":
            return None
        return (f"t={self.now / 3600:8.2f}h free={self.free:3d}/{self.capacity} "
                f"queue={len(self.queue):3d} running={len(self.running):3d} "
                f"done={self.n_finished}/{self.episode_len}")

    # ------------------------------------------------------------ internals
    def _admit_arrivals(self):
        while self.next_arrival < self.episode_len and self.submit[self.next_arrival] <= self.now:
            self.queue.append(self.next_arrival)
            self.next_arrival += 1

    def _start_job(self, slot):
        job = self.queue.pop(slot)
        self.free -= int(self.gpus[job])
        self.start_time[job] = self.now
        heapq.heappush(self.running, (self.now + self.dur[job], job))

    def _jump_to_next_event(self):
        t_arrival = self.submit[self.next_arrival] if self.next_arrival < self.episode_len else math.inf
        t_finish = self.running[0][0] if self.running else math.inf
        t_next = min(t_arrival, t_finish)
        if math.isinf(t_next):
            raise RuntimeError("Simulator stuck: no future events but jobs remain")
        dt = t_next - self.now
        waited = len(self.queue) * dt
        self._accrued += waited
        self.total_wait += waited
        self.busy_gpu_seconds += (self.capacity - self.free) * dt
        self.now = t_next
        while self.running and self.running[0][0] <= self.now:
            t_done, job = heapq.heappop(self.running)
            if job < 0:  # background job (warm start): free GPUs, not counted
                self.free += self.bg_gpus[-job - 1]
                continue
            self.free += int(self.gpus[job])
            self.finish_time[job] = t_done
            self.n_finished += 1
        self._admit_arrivals()

    def _add_background(self, t_abs):
        """Jobs submitted before t_abs and (per the trace) still running at t_abs."""
        idx = np.flatnonzero((self._h_submit < t_abs) & (self._h_end > t_abs))
        for i in idx:  # already in submission order
            g = int(self._h_gpus[i])
            if g > self.free:
                self.bg_overflow_jobs += 1
                continue
            self.free -= g
            self.bg_gpus.append(g)
            heapq.heappush(self.running, (self._h_end[i] - t_abs, -len(self.bg_gpus)))

    def _any_fit(self):
        return any(self.gpus[j] <= self.free for j in self.queue[: self.window])

    def _advance_to_decision(self):
        """Move time forward until a decision is needed. Returns True when the episode is over."""
        while True:
            if self.n_finished == self.episode_len:
                return True
            if self.queue and self._any_fit():
                return False
            self._jump_to_next_event()

    def _duration_feature(self, job):
        if self.duration_info == "estimate":
            return self.est[job]
        if self.duration_info == "oracle":
            return self.dur[job]
        return 0.0

    def _obs(self):
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        for slot, job in enumerate(self.queue[: self.window]):
            b = slot * SLOT_FEATURES
            obs[b] = 1.0
            obs[b + 1] = self.gpus[job] / self.capacity
            obs[b + 2] = float(self.gpus[job] <= self.free)
            obs[b + 3] = math.log1p(self._duration_feature(job)) / LOG_MAX_TIME
            obs[b + 4] = math.log1p(max(self.now - self.submit[job], 0.0)) / LOG_MAX_TIME
        g = self.window * SLOT_FEATURES
        log_n = math.log1p(self.episode_len)
        obs[g] = self.free / self.capacity
        obs[g + 1] = math.log1p(len(self.queue)) / log_n
        obs[g + 2] = math.log1p(self.episode_len - self.next_arrival) / log_n
        return np.clip(obs, 0.0, 1.0)

    def _info(self):
        return {"time_h": self.now / 3600, "queue_len": len(self.queue), "free_gpus": self.free,
                "finished": self.n_finished, "invalid_actions": self.invalid_actions,
                "bg_gpus_at_start": self.bg_gpus_at_start,
                "action_mask": self.action_masks()}
