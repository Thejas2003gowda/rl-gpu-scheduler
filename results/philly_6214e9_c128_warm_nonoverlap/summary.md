Source: philly vc=6214e9 | capacity 128 GPUs | test offered load 0.82 | warm start ON (avg 127 GPUs busy at episode start) | 15 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 31.28 [22.66, 41.86] | 27.98 | 27.27 | 27.20 | 116.2 | 465.29 | 0.61 | 1.012 |
| FIFO (strict) | 39.54 [27.45, 52.98] | 36.89 | 35.53 | 35.73 | 122.3 | 661.46 | 0.60 | 1.142 |
| FIFO first-fit | 32.71 [23.40, 44.02] | 29.11 | 28.70 | 29.30 | 118.4 | 487.62 | 0.62 | 1.033 |
| Smallest-GPU-first | 30.95 [22.47, 41.10] | 27.95 | 26.95 | 27.56 | 116.2 | 464.41 | 0.61 | 1.003 |
| SJF (user avg) | 31.01 [22.53, 41.22] | 27.76 | 27.00 | 27.34 | 116.8 | 462.58 | 0.60 | 1.000 |
| SJF (predicted) | 30.88 [22.54, 40.47] | 28.15 | 26.87 | 28.05 | 119.3 | 469.37 | 0.61 | 1.001 |
| SJF (oracle) | 25.98 [19.60, 33.36] | 25.37 | 21.97 | 23.69 | 115.7 | 373.97 | 0.59 | 0.920 |
