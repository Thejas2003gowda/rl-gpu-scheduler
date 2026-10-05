Source: synthetic | capacity 48 GPUs | test offered load 1.30 | 30 episodes x 1024 jobs

| Policy | Avg JCT (h): mean [95% CI] | IQM | Avg wait (h) | Median JCT (h) | P99 JCT (h) | Avg bounded slowdown | GPU util. | Median JCT ratio vs SJF |
|---|---|---|---|---|---|---|---|---|
| Random | 17.98 [13.83, 22.41] | 16.02 | 15.14 | 13.50 | 89.6 | 107.25 | 0.59 | 1.179 |
| FIFO (strict) | 41.74 [31.94, 51.70] | 38.81 | 38.89 | 40.62 | 103.0 | 272.85 | 0.54 | 2.730 |
| FIFO first-fit | 19.92 [14.95, 25.05] | 16.85 | 17.08 | 15.55 | 81.4 | 120.01 | 0.60 | 1.282 |
| Smallest-GPU-first | 20.21 [15.46, 25.11] | 18.12 | 17.36 | 14.93 | 99.1 | 128.22 | 0.59 | 1.482 |
| SJF (user avg) | 14.49 [10.57, 18.84] | 10.56 | 11.65 | 8.60 | 88.4 | 80.36 | 0.60 | 1.000 |
| SJF (oracle) | 13.27 [9.43, 17.42] | 8.93 | 10.42 | 7.30 | 89.3 | 71.28 | 0.60 | 0.946 |
