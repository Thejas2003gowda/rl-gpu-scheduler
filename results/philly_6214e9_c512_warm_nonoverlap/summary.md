Source: philly vc=6214e9 | capacity 512 GPUs | test offered load 0.21 | warm start ON (avg 374 GPUs busy at episode start) | 15 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 5.57 [3.99, 7.32] | 5.39 | 1.56 | 2.14 | 88.1 | 12.17 | 0.36 | 1.000 |
| FIFO (strict) | 5.80 [4.12, 7.62] | 5.50 | 1.79 | 2.17 | 88.2 | 18.08 | 0.36 | 1.000 |
| FIFO first-fit | 5.70 [4.05, 7.49] | 5.46 | 1.69 | 2.14 | 88.2 | 14.76 | 0.36 | 1.000 |
| Smallest-GPU-first | 5.50 [3.98, 7.21] | 5.28 | 1.49 | 2.11 | 88.2 | 10.63 | 0.36 | 1.000 |
| SJF (user avg) | 5.48 [3.96, 7.20] | 5.25 | 1.47 | 2.07 | 88.3 | 11.31 | 0.35 | 1.000 |
| SJF (predicted) | 5.53 [3.99, 7.24] | 5.31 | 1.52 | 2.13 | 88.2 | 11.97 | 0.36 | 1.000 |
| SJF (oracle) | 5.38 [3.89, 7.06] | 5.14 | 1.37 | 2.00 | 88.2 | 9.15 | 0.35 | 1.000 |
