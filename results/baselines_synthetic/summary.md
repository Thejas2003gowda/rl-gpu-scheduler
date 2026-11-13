Source: synthetic | capacity 64 GPUs | test offered load 0.98 | warm start OFF | 30 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 6.65 [5.10, 8.43] | 5.09 | 3.80 | 3.23 | 56.0 | 26.10 | 0.47 | 1.246 |
| FIFO (strict) | 16.85 [12.07, 22.09] | 12.70 | 14.00 | 13.59 | 58.6 | 97.76 | 0.46 | 2.902 |
| FIFO first-fit | 6.37 [4.97, 7.95] | 5.06 | 3.53 | 3.11 | 52.9 | 24.75 | 0.47 | 1.196 |
| Smallest-GPU-first | 6.73 [5.23, 8.32] | 5.29 | 3.88 | 3.34 | 60.7 | 27.37 | 0.47 | 1.224 |
| SJF (user avg) | 5.13 [4.10, 6.33] | 4.11 | 2.29 | 1.72 | 53.5 | 14.53 | 0.48 | 1.000 |
| SJF (oracle) | 4.94 [3.99, 6.13] | 3.93 | 2.09 | 1.36 | 52.6 | 13.16 | 0.48 | 0.993 |
