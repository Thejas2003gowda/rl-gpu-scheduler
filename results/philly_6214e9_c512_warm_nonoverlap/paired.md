Paired comparison vs FIFO first-fit on the same 15 episodes (average wait; negative = less waiting)

| Policy | Wait difference (h): mean [95% CI] | Change in total wait: % [95% CI] | Episodes with less wait | Clear difference? |
|---|---|---|---|---|
| Random | -0.13 [-0.37, -0.00] | -7.5% [-25.3, -0.0] | 40% | yes |
| FIFO (strict) | +0.11 [+0.00, +0.28] | +6.3% [+0.3, +22.7] | 0% | yes |
| Smallest-GPU-first | -0.20 [-0.59, +0.00] | -11.7% [-39.4, +0.0] | 13% | no (CI includes 0) |
| SJF (user avg) | -0.21 [-0.63, +0.00] | -12.7% [-41.2, +0.1] | 27% | no (CI includes 0) |
| SJF (predicted) | -0.17 [-0.51, +0.00] | -10.0% [-33.9, +0.1] | 20% | no (CI includes 0) |
| SJF (oracle) | -0.32 [-0.85, -0.02] | -18.8% [-53.4, -3.3] | 40% | yes |
