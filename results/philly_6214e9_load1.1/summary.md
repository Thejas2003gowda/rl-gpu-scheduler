Source: philly vc=6214e9 | capacity 208 GPUs | test offered load 0.51 | 30 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 4.24 [3.36, 5.22] | 3.83 | 0.40 | 1.18 | 82.6 | 3.02 | 0.13 | 1.000 |
| FIFO (strict) | 4.30 [3.41, 5.29] | 3.86 | 0.45 | 1.22 | 82.6 | 3.43 | 0.13 | 1.000 |
| FIFO first-fit | 4.24 [3.36, 5.22] | 3.83 | 0.40 | 1.18 | 82.6 | 3.02 | 0.13 | 1.000 |
| Smallest-GPU-first | 4.24 [3.36, 5.22] | 3.83 | 0.40 | 1.18 | 82.6 | 3.02 | 0.13 | 1.000 |
| SJF (user avg) | 4.24 [3.36, 5.22] | 3.83 | 0.40 | 1.18 | 82.6 | 3.20 | 0.13 | 1.000 |
| SJF (oracle) | 4.24 [3.36, 5.22] | 3.82 | 0.39 | 1.17 | 82.6 | 2.89 | 0.13 | 1.000 |
