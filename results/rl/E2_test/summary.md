Test period, non-overlapping 1,024-job episodes, best checkpoints. Negative = less waiting than first-fit.

| Setting | Seeds | IQM change vs first-fit [95% CI] | Success? | Gap to SJF(true) closed | vs best realistic rule (median seed) | WAIT share |
|---|---|---|---|---|---|---|
| dqn_c128_estimate_m8 | 5 | -5.0% [-9.3, -1.0] | yes | 33% [11, 62] | +1.6% vs sjf_pred | 0.000 |
| dqn_c256_estimate_m8 | 10 | +1.7% [-3.3, +11.4] | no | -126% [-1840, 29] | -1.3% vs sgf | 0.000 |
| ppo_c128_estimate_m8 | 5 | -2.9% [-12.3, +7.7] | no | 22% [-22, 81] | +3.7% vs sjf_pred | 0.000 |
| ppo_c256_estimate_m8 | 10 | +0.2% [-3.8, +8.1] | no | -68% [-1001, 55] | -1.4% vs sgf | 0.000 |
