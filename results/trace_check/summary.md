Virtual cluster 6214e9: real vs simulated waits on the same 15,360 test jobs (15 non-overlapping episodes)
(jobs with no recorded first start: 0)

| Source | Mean wait (h) | Median (h) | 90th pct (h) | 99th pct (h) | Waited > 1 min | Waited > 1 h |
|---|---|---|---|---|---|---|
| trace (recorded) | 1.45 | 0.068 | 1.01 | 5.2 | 69% | 10% |
| simulator, first-fit, 128 GPUs | 28.70 | 21.484 | 71.93 | 143.4 | 88% | 86% |
| simulator, first-fit, 192 GPUs | 10.92 | 7.113 | 27.38 | 55.7 | 80% | 77% |
| simulator, first-fit, 256 GPUs | 6.99 | 3.120 | 20.88 | 47.0 | 69% | 65% |

GPUs really in use by 6214e9 (from recorded attempt times, time-weighted)

| Period | Peak | Median | 90th pct | Average |
|---|---|---|---|---|
| test | 696 | 144 | 388 | 193 |
| whole trace | 891 | 316 | 502 | 275 |

Offered load (GPU-time requested / (GPUs x time span))

| GPUs | Training (fit) | Validation | Test |
|---|---|---|---|
| 128 | 2.04 | 2.82 | 0.82 |
| 192 | 1.36 | 1.88 | 0.55 |
| 256 | 1.02 | 1.41 | 0.41 |
