Test period, non-overlapping 1,024-job episodes, best checkpoints. Negative = less waiting than first-fit.

| Setting | Seeds | IQM change vs first-fit [95% CI] | Success? | Gap to SJF(true) closed | vs best realistic rule (median seed) | WAIT share |
|---|---|---|---|---|---|---|
| dqn_c128_estimate_wait10 | 5 | +11.2% [-10.3, +116.7] | no | -144% [-412, -56] | -0.6% vs sjf_pred | 0.271 |
| dqn_c128_oracle_wait10 | 5 | -20.3% [-32.0, +83.2] | no | -30% [-232, 36] | -18.3% vs sjf_pred | 0.275 |
| dqn_c256_estimate_wait10 | 10 | +26.7% [+0.8, +137.9] | no | -686% [-1784, -285] | +5.6% vs sjf_pred | 0.307 |
| dqn_c256_oracle_wait10 | 10 | -1.4% [-12.7, +93.5] | no | -399% [-1303, -71] | -2.3% vs sjf_pred | 0.286 |
