Paired comparison vs FIFO first-fit on the same 15 episodes (average wait; negative = less waiting)

| Policy | Wait difference (h): mean [95% CI] | Change in total wait: % [95% CI] | Episodes with less wait | Clear difference? |
|---|---|---|---|---|
| Random | +0.22 [-0.01, +0.64] | +4.7% [-0.2, +14.0] | 47% | no (CI includes 0) |
| FIFO (strict) | +1.04 [+0.35, +1.92] | +22.8% [+10.9, +49.7] | 0% | yes |
| Smallest-GPU-first | -0.14 [-0.46, +0.04] | -3.1% [-8.2, +1.5] | 33% | no (CI includes 0) |
| SJF (user avg) | -0.03 [-0.49, +0.48] | -0.6% [-9.2, +10.4] | 67% | no (CI includes 0) |
| SJF (predicted) | -0.10 [-0.37, +0.06] | -2.1% [-6.5, +2.2] | 53% | no (CI includes 0) |
| SJF (oracle) | -0.54 [-1.33, -0.08] | -11.8% [-23.0, -3.2] | 80% | yes |
