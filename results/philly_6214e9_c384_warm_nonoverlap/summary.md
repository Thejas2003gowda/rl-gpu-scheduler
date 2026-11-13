Source: philly vc=6214e9 | capacity 384 GPUs | test offered load 0.27 | warm start ON (avg 311 GPUs busy at episode start) | 15 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 8.82 [5.85, 12.04] | 7.36 | 4.81 | 5.17 | 92.3 | 65.28 | 0.40 | 1.001 |
| FIFO (strict) | 9.64 [6.38, 13.17] | 7.95 | 5.63 | 6.13 | 93.5 | 84.97 | 0.40 | 1.032 |
| FIFO first-fit | 8.60 [5.76, 11.67] | 7.31 | 4.59 | 5.11 | 91.7 | 58.12 | 0.40 | 1.000 |
| Smallest-GPU-first | 8.46 [5.76, 11.32] | 7.33 | 4.45 | 5.07 | 91.4 | 57.35 | 0.40 | 1.000 |
| SJF (user avg) | 8.57 [5.81, 11.60] | 7.27 | 4.56 | 5.06 | 92.5 | 58.83 | 0.40 | 1.000 |
| SJF (predicted) | 8.50 [5.77, 11.40] | 7.34 | 4.49 | 5.05 | 91.2 | 57.63 | 0.40 | 1.000 |
| SJF (oracle) | 8.06 [5.63, 10.56] | 7.19 | 4.05 | 4.74 | 91.3 | 47.18 | 0.40 | 0.998 |
