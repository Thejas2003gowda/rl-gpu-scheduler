# Evaluation protocol (pre-registered)

Fixed **before any trained agent's result on the test period was used for anything**. In
the commit history this file precedes every test-period result of a trained agent.
Everything below was decided from the original study design, the rule-based baselines,
and validation-only training runs.

## What was already seen
* Test-period results of the rule-based baselines (`results/philly_*`, `results/trace_check/`).
* Two pipeline smoke checks of `scripts/evaluate.py` on the test period with throwaway
  agents (20k-step agents; a 6k-step DQN with true run times). Not used for any choice.
* Validation results of batches A-A5 (`results/validation/`), used to fix the networks and
  hyperparameters in `configs/` (see `configs/README.md`).

## Main test (as originally designed)
* Virtual cluster 6214e9, **256 GPUs**, warm start, user-average run times (`estimate`).
* 15 back-to-back, non-overlapping 1,024-job test episodes; reference = FIFO first-fit.
* **Success** (per algorithm, PPO and DQN separately): the interquartile mean over **10
  seeds** of the % change in total wait vs first-fit has a 95% bootstrap CI (resampling
  seeds and episodes) entirely below zero (`gpusched/stats.py: success_test`).
* Robustness: the same on the 30 randomly placed (overlapping) episodes.

## Secondary tests (reported whatever they show)
* **Scarcity:** the same at 128 and 192 GPUs, 5 seeds per algorithm.
* **E1, information levels** (PPO): none / user average / predicted / true run times, at
  256 and 128 GPUs, 5 seeds. Reported as the share of the first-fit -> SJF(true) gap closed.
* Every RL setting is also compared with the best realistic rule on the same episodes
  (smallest-GPU-first, SJF with user average, SJF with predicted run times, random).

## Fixed settings
* Algorithms and hyperparameters: `configs/ppo.json`, `configs/dqn.json` (frozen).
* PPO: per-job scorer, gamma 1.0, 500k steps, validation every 25k steps.
* DQN: per-job scorer, **no WAIT action** (with WAIT it stalled on validation), gamma 0.99,
  200k steps, validation every 10k steps.
* Checkpoint: the best on the 5 non-overlapping validation episodes. Never chosen on test.
* Seeds: 0-9 (main), 0-4 (everything else). All runs listed in `batches/B_main.txt`.

## Known caveats (reported with the results)
* All studied sizes (128-256 GPUs) are smaller than the real cluster (it used up to 696
  GPUs); they are GPU-scarcity scenarios. Recorded waits: 1.45 h mean vs 7.0 h simulated.
* The validation period is busier than the test period (offered load 1.41 vs 0.41 at 256).
* Later experiments (WAIT ablation with PPO, second virtual cluster 11cb48, 8-GPU machines)
  are added to this file before their test evaluation.

---

## Addendum: E3 (second virtual cluster)

Added after the main test results and before any E3 result.

* **Question (RQ3):** do the main agents, trained on 6214e9 at 256 GPUs with user-average
  run times, still beat first-fit on another virtual cluster without retraining?
* **Agents:** `runs/B/dqn_c256_estimate` and `runs/B/ppo_c256_estimate` (10 seeds each,
  best validation checkpoints). No retraining, no tuning.
* **Cluster:** 11cb48, its last 30% of jobs by submit time (5,807 jobs), warm start.
* **Size rule (from the original design, fixed before any 11cb48 result):** the cluster size
  that gives 11cb48's test period the same offered load as the main test (0.412 at 256
  GPUs), rounded to a multiple of 8: **72 GPUs** (load 0.424).
* **Main E3 result:** the 5 non-overlapping 1,024-job episodes, same success criterion
  form as the main test. With only 5 episodes the CIs will be wide; reported as is.
  Robustness: 30 overlapping episodes.
* **Already seen:** one pipeline smoke check of the `--vc/--capacity` options on 11cb48 at
  72 GPUs with two throwaway agents from a 4k/6k-step smoke test. Not used for any choice.
* **Dropped:** the PPO WAIT ablation (PPO chose WAIT in 0% of decisions in every
  setting of batch B, so removing WAIT cannot change it).
