Source: philly vc=6214e9 | capacity 256 GPUs | test offered load 0.41 | 30 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 4.06 [3.20, 4.99] | 3.64 | 0.22 | 0.99 | 82.4 | 2.01 | 0.10 | 1.000 |
| FIFO (strict) | 4.08 [3.21, 5.03] | 3.65 | 0.24 | 1.01 | 82.4 | 2.20 | 0.10 | 1.000 |
| FIFO first-fit | 4.06 [3.20, 4.99] | 3.64 | 0.21 | 0.99 | 82.4 | 2.01 | 0.10 | 1.000 |
| Smallest-GPU-first | 4.06 [3.20, 4.99] | 3.64 | 0.21 | 0.99 | 82.4 | 2.01 | 0.10 | 1.000 |
| SJF (user avg) | 4.06 [3.20, 4.99] | 3.64 | 0.22 | 0.99 | 82.4 | 2.09 | 0.10 | 1.000 |
| SJF (oracle) | 4.06 [3.19, 4.99] | 3.63 | 0.21 | 0.98 | 82.4 | 1.95 | 0.10 | 1.000 |
