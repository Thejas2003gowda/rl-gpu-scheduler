# Frozen hyperparameters (chosen on validation only)

Evidence: `results/validation/` (batches A-A5, true run times, 256 and 128 GPUs).

* **PPO** (`ppo.json`): per-job scorer (`arch=slot`), gamma = 1.0. It matched SJF with true
  run times on validation (-20.6% at 256 GPUs, -19.2% at 128; SJF -20.4% / -17.6%).
  gamma = 0.99 was worse at 256 GPUs (-4.0%). Keeps the WAIT action.
* **DQN** (`dqn.json`): per-job scorer, gamma = 0.99, lr 1e-4, target update 5,000, and
  **always trained with `--no-wait`**: with WAIT available it chose WAIT 20-48% of the time
  and fell far behind SJF; dueling heads and 10-step returns made this worse. Without
  WAIT it reaches about -18.5% early (validation) and then drifts to about -5%, so the
  validation-selected checkpoint matters; evaluate every 10k steps.
* The flat MLP (`arch=mlp`) is kept only for comparison: it could not even imitate SJF
  (84% vs 99% accuracy; `results/validation/architecture_diagnostic_c256.txt`).
