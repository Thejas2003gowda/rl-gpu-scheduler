Test period, non-overlapping 1,024-job episodes, best checkpoints. Negative = less waiting than first-fit.

| Setting | Seeds | IQM change vs first-fit [95% CI] | Success? | Gap to SJF(true) closed | vs best realistic rule (median seed) | WAIT share |
|---|---|---|---|---|---|---|
| dqn_c128_estimate | 5 | -7.4% [-12.4, -3.3] | yes | 33% [22, 50] | -0.7% vs sjf_pred | 0.000 |
| dqn_c192_estimate | 5 | -4.5% [-9.6, +0.8] | no | 36% [3, 78] | +1.8% vs sgf | 0.000 |
| dqn_c256_estimate | 10 | -2.0% [-4.4, -0.4] | yes | 19% [7, 49] | -1.1% vs sjf_pred | 0.000 |
| ppo_c128_estimate | 5 | -0.2% [-5.1, +3.3] | no | 5% [-4, 18] | +7.5% vs sjf_pred | 0.000 |
| ppo_c128_none | 5 | +0.1% [-0.3, +0.5] | no | -0% [-1, 1] | +6.9% vs sjf_pred | 0.000 |
| ppo_c128_oracle | 5 | -22.9% [-32.0, -2.2] | yes | 73% [62, 77] | -17.8% vs sjf_pred | 0.000 |
| ppo_c128_predicted | 5 | -1.3% [-5.9, +1.7] | no | 9% [-1, 20] | +6.2% vs sjf_pred | 0.000 |
| ppo_c192_estimate | 5 | +1.5% [-3.4, +9.4] | no | -12% [-43, 23] | +6.2% vs sgf | 0.000 |
| ppo_c256_estimate | 10 | +0.4% [-3.3, +5.3] | no | -5% [-34, 52] | +1.6% vs sjf_pred | 0.000 |
| ppo_c256_none | 5 | -0.0% [-1.4, +4.3] | no | -7% [-25, 17] | +0.9% vs sjf_pred | 0.000 |
| ppo_c256_oracle | 5 | -10.3% [-15.7, +3.0] | no | 68% [44, 129] | -10.0% vs sjf_pred | 0.000 |
| ppo_c256_predicted | 5 | -0.3% [-2.7, +2.9] | no | 2% [-10, 36] | -0.0% vs sjf_pred | 0.000 |
