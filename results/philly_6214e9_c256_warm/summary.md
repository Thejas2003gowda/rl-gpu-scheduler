Source: philly vc=6214e9 | capacity 256 GPUs | test offered load 0.41 | warm start ON (avg 219 GPUs busy at episode start) | 30 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 9.01 [7.31, 10.75] | 8.34 | 5.16 | 6.15 | 88.0 | 80.02 | 0.42 | 1.000 |
| FIFO (strict) | 11.16 [8.45, 14.15] | 9.27 | 7.31 | 8.49 | 89.4 | 118.93 | 0.41 | 1.078 |
| FIFO first-fit | 8.83 [7.29, 10.36] | 8.47 | 4.98 | 5.88 | 87.6 | 76.34 | 0.42 | 1.000 |
| Smallest-GPU-first | 8.56 [7.17, 9.96] | 8.26 | 4.71 | 5.46 | 87.3 | 73.64 | 0.42 | 1.000 |
| SJF (user avg) | 9.74 [7.60, 12.17] | 8.37 | 5.89 | 7.06 | 88.4 | 92.48 | 0.42 | 1.000 |
| SJF (oracle) | 7.96 [6.71, 9.28] | 7.76 | 4.12 | 4.98 | 87.8 | 55.60 | 0.42 | 0.981 |
