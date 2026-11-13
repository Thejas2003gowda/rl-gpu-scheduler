# Secondary measures (test period, non-overlapping episodes)

Change vs first-fit (negative = better); utilization is absolute. Agents: IQM over seeds of each seed's mean over the episodes. Medians and P99 are per episode, then averaged. Descriptive only: no criterion is attached to these.

### 128 GPUs

First-fit: average JCT 32.71 h, median 29.30 h, P99 118.4 h, bounded slowdown 487.6, utilization 62.0%.

| Policy | Average JCT | Median JCT | P99 JCT | Bounded slowdown | Utilization |
|---|---|---|---|---|---|
| random | -4.4% | -7.2% | -1.8% | -4.6% | 61.3% |
| fifo | +20.9% | +21.9% | +3.3% | +35.7% | 60.1% |
| sgf | -5.4% | -6.0% | -1.8% | -4.8% | 61.0% |
| sjf | -5.2% | -6.7% | -1.4% | -5.1% | 60.2% |
| sjf_pred | -5.6% | -4.3% | +0.8% | -3.7% | 60.8% |
| sjf_oracle | -20.6% | -19.2% | -2.3% | -23.3% | 59.4% |
| dqn_c128_estimate | -6.5% | -6.8% | -1.1% | -6.1% | 60.6% |
| ppo_c128_estimate | -0.2% | -1.2% | +1.1% | +0.1% | 61.6% |
| ppo_c128_none | +0.1% | +0.0% | +0.0% | +0.1% | 62.0% |
| ppo_c128_oracle | -20.1% | -18.3% | -2.6% | -23.5% | 59.7% |
| ppo_c128_predicted | -1.1% | -1.9% | +1.7% | -0.5% | 61.6% |

### 192 GPUs

First-fit: average JCT 14.93 h, median 11.64 h, P99 99.0 h, bounded slowdown 211.8, utilization 53.1%.

| Policy | Average JCT | Median JCT | P99 JCT | Bounded slowdown | Utilization |
|---|---|---|---|---|---|
| random | -1.1% | -2.1% | +0.7% | -3.2% | 53.1% |
| fifo | +17.6% | +18.0% | +2.6% | +26.3% | 52.4% |
| sgf | -4.5% | -7.5% | -0.3% | -4.0% | 53.0% |
| sjf | -2.3% | -2.3% | +1.1% | -4.6% | 52.4% |
| sjf_pred | -2.3% | -4.2% | -0.2% | -3.5% | 52.8% |
| sjf_oracle | -8.4% | -9.7% | +0.7% | -13.4% | 52.1% |
| dqn_c192_estimate | -3.3% | -4.8% | -0.1% | -3.2% | 52.8% |
| ppo_c192_estimate | +1.1% | +0.3% | +0.2% | +0.8% | 53.0% |

### 256 GPUs

First-fit: average JCT 11.00 h, median 6.86 h, P99 94.6 h, bounded slowdown 151.0, utilization 46.8%.

| Policy | Average JCT | Median JCT | P99 JCT | Bounded slowdown | Utilization |
|---|---|---|---|---|---|
| random | +0.6% | +0.9% | +0.4% | -1.9% | 46.7% |
| fifo | +10.7% | +16.8% | +2.4% | +15.9% | 46.4% |
| sgf | -0.5% | +0.1% | -0.5% | +0.2% | 46.8% |
| sjf | +0.1% | +3.4% | +2.3% | -2.0% | 46.5% |
| sjf_pred | -0.6% | -0.8% | -0.7% | -0.5% | 46.8% |
| sjf_oracle | -6.4% | -5.6% | -0.3% | -13.2% | 46.4% |
| dqn_c256_estimate | -1.3% | -0.6% | +0.1% | -3.9% | 46.6% |
| ppo_c256_estimate | +0.3% | +2.1% | +1.0% | -2.8% | 46.6% |
| ppo_c256_none | -0.0% | +0.0% | -0.0% | -0.0% | 46.8% |
| ppo_c256_oracle | -6.6% | -4.8% | +0.4% | -17.6% | 46.4% |
| ppo_c256_predicted | -0.2% | +0.0% | -0.2% | -1.8% | 46.8% |
