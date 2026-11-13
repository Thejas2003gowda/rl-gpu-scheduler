Paired comparison vs FIFO first-fit on the same 15 episodes (average wait; negative = less waiting)

| Policy | Wait difference (h): mean [95% CI] | Change in total wait: % [95% CI] | Episodes with less wait | Clear difference? |
|---|---|---|---|---|
| Random | +0.12 [-0.10, +0.39] | +2.2% [-3.1, +5.7] | 47% | no (CI includes 0) |
| FIFO (strict) | +1.14 [+0.39, +2.06] | +21.8% [+11.0, +39.5] | 0% | yes |
| Smallest-GPU-first | -0.07 [-0.23, +0.04] | -1.2% [-3.6, +1.1] | 47% | no (CI includes 0) |
| SJF (user avg) | -0.01 [-0.35, +0.45] | -0.2% [-6.6, +8.1] | 47% | no (CI includes 0) |
| SJF (predicted) | -0.10 [-0.25, +0.02] | -1.9% [-4.6, +0.4] | 53% | no (CI includes 0) |
| SJF (oracle) | -0.68 [-1.46, -0.14] | -12.9% [-22.3, -4.2] | 87% | yes |
