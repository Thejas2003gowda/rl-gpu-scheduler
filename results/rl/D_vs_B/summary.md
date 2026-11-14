estimate_wait10 vs estimate: change in total wait vs first-fit on the same test episodes (IQM over seeds, 95% CI). Difference = first minus second, in percentage points (negative = less waiting); CI resamples each agent's seeds separately and the episodes jointly.

| Agents | estimate_wait10 | estimate | Difference | estimate_wait10 better (CI < 0) |
|---|---|---|---|---|
| dqn_c128_estimate_wait10 vs dqn_c128_estimate | +11.2% [-10.3, +116.7] | -7.4% [-12.4, -3.3] | +18.6 [-2.8, +124.0] | no |
| dqn_c256_estimate_wait10 vs dqn_c256_estimate | +26.7% [+0.8, +137.9] | -2.0% [-4.4, -0.4] | +28.6 [+3.3, +138.5] | no |
