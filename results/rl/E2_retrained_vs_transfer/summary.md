estimate_m8 vs estimate: change in total wait vs first-fit on the same test episodes (IQM over seeds, 95% CI). Difference = first minus second, in percentage points (negative = less waiting); CI resamples each agent's seeds separately and the episodes jointly.

| Agents | estimate_m8 | estimate | Difference | estimate_m8 better (CI < 0) |
|---|---|---|---|---|
| dqn_c128_estimate_m8 vs dqn_c128_estimate | -5.0% [-9.3, -1.0] | -5.2% [-12.1, -0.5] | +0.2 [-4.0, +6.2] | no |
| dqn_c256_estimate_m8 vs dqn_c256_estimate | +1.7% [-3.3, +11.4] | +0.4% [-3.7, +6.8] | +1.3 [-3.5, +9.9] | no |
| ppo_c128_estimate_m8 vs ppo_c128_estimate | -2.9% [-12.3, +7.7] | -1.6% [-9.2, +7.7] | -1.3 [-9.7, +6.2] | no |
| ppo_c256_estimate_m8 vs ppo_c256_estimate | +0.2% [-3.8, +8.1] | +11.7% [-2.0, +35.4] | -11.5 [-37.4, +1.8] | no |
