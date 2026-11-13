Source: philly vc=6214e9 | capacity 320 GPUs | test offered load 0.33 | warm start ON (avg 273 GPUs busy at episode start) | 15 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 9.37 [6.37, 12.75] | 8.13 | 5.36 | 5.70 | 92.6 | 82.99 | 0.43 | 1.000 |
| FIFO (strict) | 10.39 [6.98, 14.17] | 8.91 | 6.38 | 6.78 | 94.3 | 111.54 | 0.42 | 1.076 |
| FIFO first-fit | 9.25 [6.33, 12.45] | 8.19 | 5.24 | 5.78 | 92.4 | 84.51 | 0.43 | 1.000 |
| Smallest-GPU-first | 9.19 [6.33, 12.34] | 8.18 | 5.18 | 5.76 | 92.3 | 83.87 | 0.43 | 1.000 |
| SJF (user avg) | 9.24 [6.33, 12.46] | 8.09 | 5.23 | 5.80 | 93.9 | 82.52 | 0.43 | 1.000 |
| SJF (predicted) | 9.16 [6.29, 12.29] | 8.14 | 5.15 | 5.64 | 92.1 | 77.97 | 0.43 | 1.000 |
| SJF (oracle) | 8.58 [6.07, 11.20] | 7.85 | 4.57 | 5.27 | 92.0 | 61.65 | 0.42 | 0.996 |
