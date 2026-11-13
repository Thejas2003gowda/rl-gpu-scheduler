Run-time predictors, virtual cluster 6214e9 (validation 5,455 jobs, test 15,587 jobs)

Error factor: median of max(pred/true, true/pred) (1 = perfect). Pair-order accuracy: share of nearby job pairs (< 10 apart in the queue, run times differing by >= 10%) put in the right order (50% = coin flip). Higher is better.

| Predictor | Val. error factor | Val. pair-order acc. | Test error factor | Test pair-order acc. | Test Spearman |
|---|---|---|---|---|---|
| User average (current) | x11.62 | 54.5% | x5.05 | 52.0% | 0.34 |
| User geometric mean | x1.77 | 55.1% | x3.58 | 52.4% | 0.21 |
| User median | x2.01 | 54.2% | x2.56 | 52.9% | 0.37 |
| User's last 2 jobs | x1.75 | 55.5% | x2.58 | 54.0% | 0.37 |
| User average, same GPU count | x1.77 | 55.8% | x3.49 | 52.8% | 0.21 |
| Boosted-tree model | x2.66 | 58.3% | x5.83 | 52.8% | 0.13 |

Model chosen on validation: {'all_vcs': True, 'loss': 'squared_error', 'max_leaf_nodes': 15}
Decision rule (pre-set): keep the 'predicted' level if validation pair-order accuracy beats the user average by >= 3 points. Gain = +3.8 points -> KEEP.

Top features (validation, permutation importance):
u_fail_frac (0.319), ug_mean (0.204), u_mean (0.124), u_median (0.065), u_last2 (0.042), u_std (0.039)
