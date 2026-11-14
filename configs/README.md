# Frozen hyperparameters (chosen on validation only)

These settings were fixed from single-seed capability checks (agents given true run times,
validation period; `results/validation/batch*.csv`) before any test-period result, and every
test result uses them. The checks were later repeated with 3 seeds and the original settings
(`batches/A*.txt`, `results/validation/rerun/`); what that showed is noted under each choice.

* **PPO** (`ppo.json`): per-job scorer (`arch=slot`), gamma = 1.0, keeps the WAIT action.
  - Chosen because one seed matched SJF with true run times (-20.6% at 256 GPUs; SJF -20.4%)
    while gamma = 0.99 reached only -4.0%.
  - With 3 seeds, the per-job scorer learns in 4 of 6 runs with gamma 1.0 and 3 of 6 with
    gamma 0.99, and the failed runs never choose WAIT. The per-job scorer is needed, but no
    clear difference supports the choice of gamma.
* **DQN** (`dqn.json`): per-job scorer, gamma = 0.99, lr 1e-4, target update 5,000, trained
  without WAIT (`--no-wait`), checkpoint chosen on validation every 10k steps.
  - Chosen because, with the earlier settings (lr 3e-4, target update 2,000), DQN with WAIT
    chose WAIT 20-48% of the time and stayed far from SJF; dueling heads made it worse, and
    10-step returns were unstable. Without WAIT and with the slower settings it reached about
    -18% early in training, then drifted to about -5%.
  - With 3 seeds these findings hold. One added check (batch A6) does not fit the choice:
    with the frozen settings, the WAIT action and 10-step returns, DQN reaches -32% on
    validation in all 3 seeds (-24% at the last checkpoint), beating SJF with true run times
    by keeping GPUs free. It was found after the test results, so the frozen settings were
    not changed. Tested on the test period as E4 (pre-registered), it at best matches SJF with
    true run times, and with user averages it is worse than first-fit and than the DQN
    without WAIT.
* **Flat MLP** (`arch=mlp`) is kept only for comparison: it does not learn (3 seeds) and
  cannot even imitate SJF (84% vs 99% agreement; `results/validation/`
  `architecture_diagnostic_c256.txt`, repeated in `rerun/`).
