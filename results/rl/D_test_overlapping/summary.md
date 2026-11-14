Test period, 30 overlapping 1,024-job episodes, best checkpoints. Negative = less waiting than first-fit.

| Setting | Seeds | IQM change vs first-fit [95% CI] | Success? | Gap to SJF(true) closed | vs best realistic rule (median seed) | WAIT share |
|---|---|---|---|---|---|---|
| dqn_c256_estimate_wait10 | 10 | +37.0% [-3.4, +149.6] | no | -465% [-881, -206] | +17.2% vs sgf | 0.304 |
| dqn_c256_oracle_wait10 | 10 | -11.6% [-26.4, +83.4] | no | -166% [-485, 7] | -9.0% vs sgf | 0.275 |
