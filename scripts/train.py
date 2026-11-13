"""Train one RL agent (DQN or MaskablePPO) for one seed.

Training: random 256-job episodes from the FIT period (warm start).
Validation: every --eval-every steps, the greedy policy runs on the
non-overlapping 1,024-job validation episodes; the checkpoint with the lowest
total wait (vs first-fit on the same episodes) is kept as best.
The test period is never touched here.

Outputs in --out:
    config.json          everything needed to rerun or evaluate
    learning_curve.csv   validation score over training
    train_episodes.csv   return and average wait of every training episode
    best.pt / best.zip   best checkpoint on validation (DQN / PPO)

Examples:
    python scripts/train.py --algo ppo --seed 0 --timesteps 1000000 --out runs/ppo_c256_estimate/seed0
    python scripts/train.py --algo dqn --duration-info oracle --capacity 128 --out runs/dqn_c128_oracle/seed0
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched.baselines import fifo_ff, run_episode, sjf_oracle  # noqa: E402
from gpusched.experiment import CountingPolicy, load_vc_jobs, make_env, nonoverlap_starts  # noqa: E402

DEFAULT_CSV = "data/philly_jobs_pred_6214e9.csv"


def git_hash():
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True,
                                       cwd=Path(__file__).resolve().parents[1]).strip()
    except Exception:
        return "unknown"


class Validator:
    """Greedy policy on the validation episodes, scored against first-fit."""

    def __init__(self, env):
        self.env = env
        self.starts = nonoverlap_starts(env)
        self.ref = np.array([run_episode(env, fifo_ff, start=int(s))[0]["avg_wait_h"] for s in self.starts])
        sjf_true = np.array([run_episode(env, sjf_oracle, start=int(s))[0]["avg_wait_h"] for s in self.starts])
        self.sjf_true_change_pct = float(100 * (sjf_true.sum() / self.ref.sum() - 1))

    def __call__(self, act):
        pol = CountingPolicy(act)
        waits = np.array([run_episode(self.env, pol, start=int(s))[0]["avg_wait_h"] for s in self.starts])
        return {"val_change_pct": 100 * (waits.sum() / self.ref.sum() - 1), "val_avg_wait_h": waits.mean(),
                "val_wait_share": pol.waits / max(pol.steps, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--algo", choices=["dqn", "ppo"], required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--jobs-csv", default=DEFAULT_CSV)
    ap.add_argument("--vc", default="6214e9")
    ap.add_argument("--capacity", type=int, default=256)
    ap.add_argument("--duration-info", default="estimate", choices=["estimate", "predicted", "oracle", "none", "signals"])
    ap.add_argument("--no-wait", action="store_true", help="ablation: remove the WAIT action")
    ap.add_argument("--timesteps", type=int, default=1_000_000)
    ap.add_argument("--eval-every", type=int, default=50_000)
    ap.add_argument("--train-episode-len", type=int, default=256)
    ap.add_argument("--val-episode-len", type=int, default=1024)
    ap.add_argument("--config", default=None, help="JSON file with hyperparameters for the algorithm")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override one hyperparameter, e.g. --set gamma=0.999 (value read as JSON)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import torch
    torch.set_num_threads(1)  # one core per run; run several seeds in parallel instead

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hp = json.loads(Path(args.config).read_text()) if args.config else {}
    for item in args.set:
        key, _, value = item.partition("=")
        try:
            hp[key] = json.loads(value)
        except json.JSONDecodeError:
            hp[key] = value
    jobs = load_vc_jobs(args.jobs_csv, args.vc)
    train_env = make_env(jobs, "fit", args.capacity, args.duration_info, args.train_episode_len,
                         no_wait=args.no_wait)
    val_env = make_env(jobs, "validation", args.capacity, args.duration_info, args.val_episode_len,
                       no_wait=args.no_wait)
    validator = Validator(val_env)
    config = {**vars(args), "hyperparameters": hp, "git": git_hash(),
              "val_episodes": len(validator.starts), "val_first_fit_avg_wait_h": float(validator.ref.mean()),
              "val_sjf_true_change_pct": validator.sjf_true_change_pct}
    (out / "config.json").write_text(json.dumps(config, indent=2))

    curve, episodes = [], []
    best = {"val_change_pct": np.inf}
    t0 = time.time()

    def record(step, act, save):
        v = validator(act)
        v.update(step=step, minutes=(time.time() - t0) / 60)
        recent = [e["avg_wait_h"] for e in episodes[-50:] if "avg_wait_h" in e]
        v["train_recent_avg_wait_h"] = float(np.mean(recent)) if recent else np.nan
        curve.append(v)
        if v["val_change_pct"] < best["val_change_pct"]:
            best.update(v)
            save()
        pd.DataFrame(curve).to_csv(out / "learning_curve.csv", index=False)
        print(f"step {step:>9,} | val change vs first-fit {v['val_change_pct']:+6.2f}% "
              f"(best {best['val_change_pct']:+6.2f}%) | WAIT share {v['val_wait_share']:.3f} "
              f"| {v['minutes']:.1f} min", flush=True)

    if args.algo == "dqn":
        from gpusched.dqn import DQNConfig, MaskedDQN
        agent = MaskedDQN(train_env.observation_space.shape[0], train_env.action_space.n,
                          DQNConfig(**hp), seed=args.seed)

        def callback(step, ag):
            if step % args.eval_every == 0:
                record(step, ag.predict, lambda: ag.save(out / "best.pt"))
            return True
        agent.learn(train_env, args.timesteps, callback=callback, log=episodes.append)
        if not curve or curve[-1]["step"] != args.timesteps:
            record(args.timesteps, agent.predict, lambda: agent.save(out / "best.pt"))
        agent.save(out / "last.pt")
    else:
        import gymnasium as gym
        from sb3_contrib import MaskablePPO
        from sb3_contrib.common.wrappers import ActionMasker
        from stable_baselines3.common.callbacks import BaseCallback

        hp = dict(hp)
        mult = hp.pop("reward_multiplier", 0.1)
        net_arch = hp.pop("net_arch", [128, 128])
        arch = hp.pop("arch", "mlp")
        slot_hidden = hp.pop("slot_hidden", [64, 64])
        env = ActionMasker(gym.wrappers.TransformReward(train_env, lambda r: r * mult),
                           lambda e: e.unwrapped.action_masks())
        if arch == "slot":
            from gpusched.nets import SlotPolicy
            policy, policy_kwargs = SlotPolicy, {"net_arch": {"pi": [], "vf": net_arch}, "slot_hidden": slot_hidden}
        else:
            policy, policy_kwargs = "MlpPolicy", {"net_arch": net_arch}
        model = MaskablePPO(policy, env, seed=args.seed, device="cpu", verbose=0,
                            policy_kwargs=policy_kwargs, **hp)

        def act(obs, mask):
            a, _ = model.predict(obs, action_masks=mask, deterministic=True)
            return int(a)

        class Cb(BaseCallback):
            def _on_step(self):
                for info in self.locals.get("infos", []):
                    if "metrics" in info:
                        episodes.append({"step": self.num_timesteps, "avg_wait_h": info["metrics"]["avg_wait_h"]})
                if self.num_timesteps % args.eval_every == 0:
                    record(self.num_timesteps, act, lambda: model.save(out / "best.zip"))
                return True
        model.learn(total_timesteps=args.timesteps, callback=Cb())
        if not curve or curve[-1]["step"] != args.timesteps:
            record(args.timesteps, act, lambda: model.save(out / "best.zip"))
        model.save(out / "last.zip")

    pd.DataFrame(episodes).to_csv(out / "train_episodes.csv", index=False)
    summary = {"best_step": int(best["step"]), "best_val_change_pct": float(best["val_change_pct"]),
               "best_val_wait_share": float(best["val_wait_share"]),
               "val_sjf_true_change_pct": validator.sjf_true_change_pct,
               "minutes": (time.time() - t0) / 60}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Done:", summary)


if __name__ == "__main__":
    main()
