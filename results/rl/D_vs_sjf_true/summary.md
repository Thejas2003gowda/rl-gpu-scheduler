Agents vs `sjf_oracle` on the same test episodes: IQM over seeds of the % change in total wait (negative = agents wait less), 95% CI resampling seeds and episodes.

| Setting | Seeds | Change vs sjf_oracle [95% CI] | Seeds below 0 | Beats the rule (CI < 0) |
|---|---|---|---|---|
| dqn_c128_estimate_wait10 | 5 | +45.3% [+11.2, +180.6] | 0 of 5 | no |
| dqn_c128_oracle_wait10 | 5 | +4.1% [-5.4, +141.1] | 3 of 5 | no |
| dqn_c256_estimate_wait10 | 10 | +40.9% [+10.2, +162.7] | 0 of 10 | no |
| dqn_c256_oracle_wait10 | 10 | +9.6% [-3.6, +114.3] | 2 of 10 | no |
