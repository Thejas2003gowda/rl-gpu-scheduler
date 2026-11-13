Test period, non-overlapping 1,024-job episodes, best checkpoints. Negative = less waiting than first-fit.

| Setting | Seeds | IQM change vs first-fit [95% CI] | Success? | Gap to SJF(true) closed | vs best realistic rule (median seed) | WAIT share |
|---|---|---|---|---|---|---|
| dqn_c256_estimate | 10 | -1.0% [-17.9, +15.7] | no | 12% [-1442, 84] | +0.3% vs random | 0.000 |
| ppo_c256_estimate | 10 | +1.4% [-10.3, +12.8] | no | -26% [-1134, 30] | +2.3% vs random | 0.000 |
