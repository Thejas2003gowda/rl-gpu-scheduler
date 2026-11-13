Paired comparison vs FIFO first-fit on the same 30 episodes (average wait; negative = less waiting)

| Policy | Wait difference (h): mean [95% CI] | Change in total wait: % [95% CI] | Episodes with less wait | Clear difference? |
|---|---|---|---|---|
| Random | +0.18 [-0.07, +0.51] | +3.6% [-1.8, +9.0] | 47% | no (CI includes 0) |
| FIFO (strict) | +2.33 [+0.87, +4.17] | +46.8% [+21.1, +78.0] | 0% | yes |
| Smallest-GPU-first | -0.27 [-0.92, +0.22] | -5.4% [-16.9, +4.9] | 30% | no (CI includes 0) |
| SJF (user avg) | +0.91 [-0.10, +2.44] | +18.2% [-2.5, +47.4] | 43% | no (CI includes 0) |
| SJF (oracle) | -0.86 [-1.56, -0.35] | -17.3% [-27.8, -8.6] | 90% | yes |
