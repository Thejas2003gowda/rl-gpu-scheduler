Test period, 30 overlapping 1,024-job episodes, best checkpoints. Negative = less waiting than first-fit.

| Setting | Seeds | IQM change vs first-fit [95% CI] | Success? | Gap to SJF(true) closed | vs best realistic rule (median seed) | WAIT share |
|---|---|---|---|---|---|---|
| dqn_c256_estimate | 10 | -2.8% [-6.1, +1.7] | no | 30% [-24, 53] | -0.0% vs sjf | 0.000 |
| ppo_c256_estimate | 10 | -1.0% [-4.4, +2.6] | no | 5% [-28, 21] | +0.5% vs sjf | 0.000 |
