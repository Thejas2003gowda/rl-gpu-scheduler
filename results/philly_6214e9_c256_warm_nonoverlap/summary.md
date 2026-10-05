Source: philly vc=6214e9 | capacity 256 GPUs | test offered load 0.41 | warm start ON (avg 232 GPUs busy at episode start) | 15 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 11.07 [7.63, 15.04] | 9.23 | 7.06 | 6.92 | 95.0 | 148.15 | 0.47 | 1.000 |
| FIFO (strict) | 12.17 [8.39, 16.54] | 10.07 | 8.16 | 8.01 | 96.9 | 174.94 | 0.46 | 1.079 |
| FIFO first-fit | 11.00 [7.63, 14.84] | 9.30 | 6.99 | 6.86 | 94.6 | 150.99 | 0.47 | 1.003 |
| Smallest-GPU-first | 10.94 [7.60, 14.73] | 9.30 | 6.93 | 6.86 | 94.1 | 151.27 | 0.47 | 1.002 |
| SJF (user avg) | 11.00 [7.62, 14.92] | 9.20 | 6.99 | 7.09 | 96.8 | 147.96 | 0.47 | 1.000 |
| SJF (oracle) | 10.29 [7.27, 13.64] | 8.98 | 6.28 | 6.47 | 94.3 | 131.08 | 0.46 | 0.988 |
