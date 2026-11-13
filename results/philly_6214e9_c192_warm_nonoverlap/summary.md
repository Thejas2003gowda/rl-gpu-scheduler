Source: philly vc=6214e9 | capacity 192 GPUs | test offered load 0.55 | warm start ON (avg 183 GPUs busy at episode start) | 15 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 14.76 [10.67, 19.24] | 12.92 | 10.75 | 11.39 | 99.7 | 204.98 | 0.53 | 1.011 |
| FIFO (strict) | 17.55 [12.70, 22.74] | 15.54 | 13.54 | 13.74 | 101.6 | 267.38 | 0.52 | 1.089 |
| FIFO first-fit | 14.93 [10.97, 19.36] | 13.06 | 10.92 | 11.64 | 99.0 | 211.78 | 0.53 | 1.024 |
| Smallest-GPU-first | 14.25 [10.62, 18.37] | 12.75 | 10.24 | 10.77 | 98.8 | 203.35 | 0.53 | 1.004 |
| SJF (user avg) | 14.58 [10.71, 18.86] | 13.08 | 10.57 | 11.37 | 100.1 | 202.04 | 0.52 | 1.000 |
| SJF (predicted) | 14.58 [10.69, 18.83] | 12.79 | 10.57 | 11.15 | 98.8 | 204.33 | 0.53 | 1.001 |
| SJF (oracle) | 13.68 [10.04, 17.71] | 12.09 | 9.67 | 10.51 | 99.7 | 183.43 | 0.52 | 0.972 |
