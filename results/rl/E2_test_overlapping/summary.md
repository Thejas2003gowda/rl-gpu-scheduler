Test period, 30 overlapping 1,024-job episodes, best checkpoints. Negative = less waiting than first-fit.

| Setting | Seeds | IQM change vs first-fit [95% CI] | Success? | Gap to SJF(true) closed | vs best realistic rule (median seed) | WAIT share |
|---|---|---|---|---|---|---|
| dqn_c256_estimate_m8 | 10 | +3.3% [-5.6, +16.4] | no | -29% [-133, 13] | +0.8% vs sgf | 0.000 |
| ppo_c256_estimate_m8 | 10 | -4.0% [-10.7, +4.5] | no | 18% [-40, 48] | -7.4% vs sgf | 0.000 |
