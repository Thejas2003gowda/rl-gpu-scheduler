signals vs estimate: change in total wait vs first-fit on the same test episodes (IQM over seeds, 95% CI). Difference = first minus second, in percentage points (negative = less waiting); CI resamples each agent's seeds separately and the episodes jointly.

| Agents | signals | estimate | Difference | signals better (CI < 0) |
|---|---|---|---|---|
| dqn_c128_signals vs dqn_c128_estimate | -8.2% [-14.2, -1.7] | -7.4% [-12.4, -3.3] | -0.8 [-7.1, +5.8] | no |
| dqn_c256_signals vs dqn_c256_estimate | -0.2% [-4.0, +4.0] | -2.0% [-4.4, -0.4] | +1.8 [-0.5, +5.4] | no |
| ppo_c128_signals vs ppo_c128_estimate | -4.1% [-11.7, +1.3] | -0.2% [-5.1, +3.3] | -3.9 [-11.6, +3.0] | no |
| ppo_c256_signals vs ppo_c256_estimate | -0.8% [-3.5, +2.0] | +0.4% [-3.3, +5.3] | -1.2 [-5.7, +2.9] | no |
