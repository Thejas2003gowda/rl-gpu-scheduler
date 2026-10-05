Source: philly vc=6214e9 | capacity 128 GPUs | test offered load 0.82 | warm start ON (avg 126 GPUs busy at episode start) | 30 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 34.01 [27.10, 41.56] | 31.53 | 30.17 | 34.76 | 114.1 | 522.41 | 0.57 | 0.999 |
| FIFO (strict) | 41.05 [31.81, 51.28] | 36.97 | 37.21 | 41.62 | 115.2 | 646.25 | 0.56 | 1.109 |
| FIFO first-fit | 34.67 [27.50, 42.56] | 30.65 | 30.82 | 35.10 | 113.2 | 560.05 | 0.57 | 1.001 |
| Smallest-GPU-first | 32.52 [26.21, 39.64] | 29.57 | 28.67 | 32.63 | 112.2 | 498.54 | 0.58 | 0.999 |
| SJF (user avg) | 33.94 [26.97, 41.55] | 31.60 | 30.09 | 34.06 | 115.1 | 526.07 | 0.57 | 1.000 |
| SJF (oracle) | 27.37 [22.74, 32.44] | 26.38 | 23.53 | 26.51 | 110.0 | 349.98 | 0.56 | 0.944 |
