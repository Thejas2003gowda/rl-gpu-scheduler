Source: philly vc=6214e9 | capacity 320 GPUs | test offered load 0.33 | 30 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 3.95 [3.10, 4.86] | 3.48 | 0.11 | 0.86 | 82.3 | 1.48 | 0.08 | 1.000 |
| FIFO (strict) | 3.95 [3.10, 4.87] | 3.49 | 0.11 | 0.86 | 82.3 | 1.48 | 0.08 | 1.000 |
| FIFO first-fit | 3.95 [3.10, 4.86] | 3.48 | 0.11 | 0.86 | 82.3 | 1.47 | 0.08 | 1.000 |
| Smallest-GPU-first | 3.95 [3.10, 4.86] | 3.48 | 0.11 | 0.86 | 82.3 | 1.47 | 0.08 | 1.000 |
| SJF (user avg) | 3.95 [3.10, 4.86] | 3.48 | 0.11 | 0.86 | 82.3 | 1.50 | 0.08 | 1.000 |
| SJF (oracle) | 3.95 [3.10, 4.86] | 3.48 | 0.10 | 0.85 | 82.3 | 1.44 | 0.08 | 1.000 |
