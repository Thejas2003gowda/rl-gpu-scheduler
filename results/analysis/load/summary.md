# Exploratory: advantage vs how busy the test episode is

Same 15 non-overlapping test episodes as the main test (no new episodes chosen). Change in total wait vs first-fit within each half (agents: IQM over seeds, 95% CI resampling seeds and that half's episodes). Negative = less waiting. Spearman: rank correlation between an episode's offered load and the per-episode change (agents: averaged over seeds); negative = the policy gains more in busier episodes. Not part of any pre-registered criterion.

### 128 GPUs

Offered load of the 15 episodes: 0.13 to 2.22 (median 1.18); lighter half 8 episodes, busier half 7.

| Policy | Lighter half | Busier half | Spearman: load vs change |
|---|---|---|---|
| dqn_c128_estimate | -5.4% [-11.5, -0.6] | -8.6% [-15.4, -2.5] | -0.37 |
| ppo_c128_estimate | -1.9% [-7.7, +4.0] | +0.9% [-5.7, +4.2] | -0.05 |
| Smallest-GPU-first | -6.8% | -5.7% | -0.49 |
| SJF (user average) | -2.7% | -8.2% | -0.42 |
| SJF (predicted) | -2.5% | -9.1% | -0.44 |
| SJF (true run times) | -20.2% | -25.7% | -0.50 |

### 192 GPUs

Offered load of the 15 episodes: 0.08 to 1.48 (median 0.79); lighter half 8 episodes, busier half 7.

| Policy | Lighter half | Busier half | Spearman: load vs change |
|---|---|---|---|
| dqn_c192_estimate | -3.3% [-9.2, +0.0] | -5.5% [-13.7, +4.0] | +0.08 |
| ppo_c192_estimate | +0.2% [-5.2, +5.3] | +3.0% [-3.4, +17.2] | +0.19 |
| Smallest-GPU-first | -4.0% | -8.6% | +0.01 |
| SJF (user average) | -1.8% | -4.7% | -0.25 |
| SJF (predicted) | -3.6% | -2.8% | +0.20 |
| SJF (true run times) | -10.2% | -12.8% | -0.11 |

### 256 GPUs

Offered load of the 15 episodes: 0.06 to 1.11 (median 0.59); lighter half 8 episodes, busier half 7.

| Policy | Lighter half | Busier half | Spearman: load vs change |
|---|---|---|---|
| dqn_c256_estimate | -1.9% [-6.6, +0.0] | -2.7% [-4.1, -0.3] | -0.02 |
| ppo_c256_estimate | +0.2% [-4.8, +6.6] | +0.6% [-3.3, +5.0] | +0.21 |
| Smallest-GPU-first | -0.1% | -2.2% | -0.46 |
| SJF (user average) | +1.8% | -2.9% | -0.04 |
| SJF (predicted) | -0.4% | -1.7% | -0.13 |
| SJF (true run times) | -7.3% | -15.2% | -0.05 |
