Paired comparison vs FIFO first-fit on the same 15 episodes (average wait; negative = less waiting)

| Policy | Wait difference (h): mean [95% CI] | Change in total wait: % [95% CI] | Episodes with less wait | Clear difference? |
|---|---|---|---|---|
| Random | -1.43 [-2.83, -0.26] | -5.0% [-8.6, -1.0] | 73% | yes |
| FIFO (strict) | +6.82 [+2.61, +13.37] | +23.8% [+10.2, +48.5] | 0% | yes |
| Smallest-GPU-first | -1.76 [-2.99, -0.64] | -6.1% [-8.3, -3.2] | 67% | yes |
| SJF (user avg) | -1.71 [-3.29, -0.24] | -5.9% [-10.2, -1.0] | 73% | yes |
| SJF (predicted) | -1.83 [-3.73, -0.26] | -6.4% [-10.6, -1.2] | 67% | yes |
| SJF (oracle) | -6.73 [-11.95, -2.69] | -23.5% [-33.1, -13.1] | 100% | yes |
