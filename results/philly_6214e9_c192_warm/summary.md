Source: philly vc=6214e9 | capacity 192 GPUs | test offered load 0.55 | warm start ON (avg 178 GPUs busy at episode start) | 30 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 14.57 [12.06, 17.35] | 13.53 | 10.72 | 12.25 | 93.9 | 163.71 | 0.49 | 1.007 |
| FIFO (strict) | 18.03 [14.52, 21.88] | 16.40 | 14.18 | 16.04 | 95.7 | 227.08 | 0.48 | 1.085 |
| FIFO first-fit | 15.18 [12.35, 18.26] | 13.53 | 11.33 | 13.05 | 93.5 | 165.18 | 0.49 | 1.021 |
| Smallest-GPU-first | 13.66 [11.24, 16.46] | 12.12 | 9.82 | 11.03 | 93.0 | 148.37 | 0.49 | 1.001 |
| SJF (user avg) | 15.12 [12.23, 18.27] | 13.41 | 11.28 | 12.95 | 94.2 | 163.65 | 0.49 | 1.000 |
| SJF (oracle) | 12.91 [10.45, 15.87] | 10.93 | 9.06 | 10.35 | 93.2 | 123.22 | 0.48 | 0.969 |
