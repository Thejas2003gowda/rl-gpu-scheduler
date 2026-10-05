Paired comparison vs FIFO first-fit on the same 15 episodes (average wait; negative = less waiting)

| Policy | Wait difference (h): mean [95% CI] | Change in total wait: % [95% CI] | Episodes with less wait | Clear difference? |
|---|---|---|---|---|
| Random | +0.07 [-0.09, +0.31] | +1.0% [-1.9, +4.0] | 53% | no (CI includes 0) |
| FIFO (strict) | +1.18 [+0.48, +2.02] | +16.8% [+9.2, +28.5] | 0% | yes |
| Smallest-GPU-first | -0.06 [-0.22, +0.06] | -0.9% [-2.5, +1.2] | 33% | no (CI includes 0) |
| SJF (user avg) | +0.01 [-0.29, +0.45] | +0.1% [-4.0, +6.2] | 53% | no (CI includes 0) |
| SJF (oracle) | -0.71 [-1.43, -0.20] | -10.1% [-16.2, -4.2] | 87% | yes |
