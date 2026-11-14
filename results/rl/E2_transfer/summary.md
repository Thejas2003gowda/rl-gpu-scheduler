Test period, non-overlapping 1,024-job episodes, best checkpoints. Negative = less waiting than first-fit.

| Setting | Seeds | IQM change vs first-fit [95% CI] | Success? | Gap to SJF(true) closed | vs best realistic rule (median seed) | WAIT share |
|---|---|---|---|---|---|---|
| dqn_c256_estimate | 10 | +0.4% [-3.7, +6.8] | no | -47% [-927, 41] | -1.4% vs sgf | 0.000 |
| ppo_c256_estimate | 10 | +11.7% [-2.0, +35.4] | no | -359% [-3020, 31] | +9.7% vs sgf | 0.000 |
| dqn_c128_estimate | 5 | -5.2% [-12.1, -0.5] | yes | 36% [9, 68] | +1.1% vs sjf_pred | 0.000 |
| ppo_c128_estimate | 5 | -1.6% [-9.2, +7.7] | no | 11% [-27, 54] | +5.2% vs sjf_pred | 0.000 |
