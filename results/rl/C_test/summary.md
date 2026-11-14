Test period, non-overlapping 1,024-job episodes, best checkpoints. Negative = less waiting than first-fit.

| Setting | Seeds | IQM change vs first-fit [95% CI] | Success? | Gap to SJF(true) closed | vs best realistic rule (median seed) | WAIT share |
|---|---|---|---|---|---|---|
| dqn_c128_signals | 5 | -8.2% [-14.2, -1.7] | yes | 32% [11, 57] | -1.9% vs sjf_pred | 0.000 |
| dqn_c256_signals | 10 | -0.2% [-4.0, +4.0] | no | 1% [-45, 39] | +0.9% vs sjf_pred | 0.000 |
| ppo_c128_signals | 5 | -4.1% [-11.7, +1.3] | no | 21% [10, 33] | +1.7% vs sjf_pred | 0.002 |
| ppo_c256_signals | 10 | -0.8% [-3.5, +2.0] | no | 3% [-15, 37] | -0.2% vs sjf_pred | 0.000 |
