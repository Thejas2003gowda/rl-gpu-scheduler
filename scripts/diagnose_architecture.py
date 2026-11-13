"""Diagnostic: can each network even REPRESENT SJF with true run times?

Imitation (supervised) learning, no RL: collect states from training episodes
while SJF(true) is in control, train a network to pick SJF's action, then run
the network greedily on the validation episodes. If a network cannot imitate
SJF well, RL with that network cannot be expected to find SJF-like policies.

Usage:
    python scripts/diagnose_architecture.py --capacity 256
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gpusched.baselines import fifo_ff, run_episode, sjf_oracle  # noqa: E402
from gpusched.experiment import load_vc_jobs, make_env, nonoverlap_starts  # noqa: E402
from gpusched.nets import make_q_network  # noqa: E402


def collect(env, n_episodes, seed):
    rng = np.random.default_rng(seed)
    X, M, Y = [], [], []
    for _ in range(n_episodes):
        env.reset(seed=int(rng.integers(2**31)))
        done = False
        while not done:
            a = sjf_oracle(env)
            X.append(env._obs()), M.append(env.action_masks()), Y.append(a)
            _, _, term, trunc, _ = env.step(a)
            done = term or trunc
    return (torch.tensor(np.array(X)), torch.tensor(np.array(M)), torch.tensor(np.array(Y)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs-csv", default="data/philly_jobs_pred_6214e9.csv")
    ap.add_argument("--vc", default="6214e9")
    ap.add_argument("--capacity", type=int, default=256)
    ap.add_argument("--episodes", type=int, default=60)
    ap.add_argument("--epochs", type=int, default=30)
    args = ap.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(0)
    jobs = load_vc_jobs(args.jobs_csv, args.vc)
    train_env = make_env(jobs, "fit", args.capacity, "oracle", 256)
    val_env = make_env(jobs, "validation", args.capacity, "oracle", 1024)
    X, M, Y = collect(train_env, args.episodes, 0)
    Xt, Mt, Yt = collect(train_env, 20, 1)
    starts = nonoverlap_starts(val_env)
    ref = np.array([run_episode(val_env, fifo_ff, start=int(s))[0]["avg_wait_h"] for s in starts])
    sjf = np.array([run_episode(val_env, sjf_oracle, start=int(s))[0]["avg_wait_h"] for s in starts])
    print(f"{len(Y):,} training states; SJF(true) on validation: {100 * (sjf.sum() / ref.sum() - 1):+.1f}% vs first-fit")
    for arch, hidden in [("mlp", [128, 128]), ("slot", [64, 64])]:
        net = make_q_network(arch, X.shape[1], M.shape[1], hidden)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        for _ in range(args.epochs):
            perm = torch.randperm(len(Y))
            for i in range(0, len(Y), 256):
                b = perm[i:i + 256]
                logits = net(X[b]).masked_fill(~M[b], -1e9)
                loss = torch.nn.functional.cross_entropy(logits, Y[b])
                opt.zero_grad()
                loss.backward()
                opt.step()
        with torch.no_grad():
            acc = (net(Xt).masked_fill(~Mt, -1e9).argmax(1) == Yt).float().mean().item()

        def policy(env):
            with torch.no_grad():
                q = net(torch.as_tensor(env._obs()).unsqueeze(0))
            return int(q.masked_fill(~torch.as_tensor(env.action_masks()).unsqueeze(0), -1e9).argmax())
        w = np.array([run_episode(val_env, policy, start=int(s))[0]["avg_wait_h"] for s in starts])
        print(f"{arch:4s}: imitation accuracy {100 * acc:.1f}% on held-out states | "
              f"validation {100 * (w.sum() / ref.sum() - 1):+.1f}% vs first-fit")


if __name__ == "__main__":
    main()
