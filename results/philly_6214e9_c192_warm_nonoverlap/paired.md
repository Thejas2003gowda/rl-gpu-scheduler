Paired comparison vs FIFO first-fit on the same 15 episodes (average wait; negative = less waiting)

| Policy | Wait difference (h): mean [95% CI] | Change in total wait: % [95% CI] | Episodes with less wait | Clear difference? |
|---|---|---|---|---|
| Random | -0.17 [-0.68, +0.28] | -1.6% [-7.1, +2.5] | 80% | no (CI includes 0) |
| FIFO (strict) | +2.62 [+0.90, +4.82] | +24.0% [+9.3, +47.2] | 0% | yes |
| Smallest-GPU-first | -0.67 [-1.64, -0.06] | -6.2% [-13.3, -0.7] | 40% | yes |
| SJF (user avg) | -0.35 [-0.85, +0.22] | -3.2% [-7.3, +2.1] | 80% | no (CI includes 0) |
| SJF (predicted) | -0.35 [-0.64, -0.10] | -3.2% [-6.1, -1.0] | 67% | yes |
| SJF (oracle) | -1.25 [-1.91, -0.68] | -11.4% [-15.9, -7.7] | 93% | yes |
