# Results table for run run-ff2df23c

Source: estimates_a0.json

| Estimate of top-3 effect on click | Effect (pp) | 95% interval | Gap to benchmark |
|---|---|---|---|
| Experimental benchmark (trimmed) | 5.25 | [5.09, 5.42] | n/a |
| Naive, observational arm | 9.77 | [9.64, 9.89] | 4.52 [4.31, 4.72] mismatch |
| Regression adjustment | 8.04 | [7.92, 8.17] | 2.80 [2.59, 3.00] mismatch |
| Inverse propensity weighting | 6.96 | [6.83, 7.09] | 1.71 [1.50, 1.92] mismatch |
| Doubly robust (AIPW) | 6.61 | [6.49, 6.73] | 1.36 [1.16, 1.57] mismatch |
| Double machine learning | 6.53 | [6.40, 6.65] | 1.28 [1.07, 1.49] mismatch |

Share of the naive gap removed by adjustment (primary estimator): 70%

Post-hoc, like-for-like (added after the pre-registered comparison):

| Comparison | Effect (pp) | Gap (pp) |
|---|---|---|
| Ranker's hotels: benchmark reweighted vs primary estimate | 5.89 [5.68, 6.12] | 0.72 [0.46, 0.98] match |
| Random hotels: observational effect carried over vs plain benchmark | 6.49 | 1.24 [1.07, 1.39] mismatch |

Spec sensitivity check: control group = every non-reserved position from 4 down (reference specification; overlap pass, 1% trimmed):

| Estimate | Effect (pp) | 95% interval | Gap to benchmark |
|---|---|---|---|
| Experimental benchmark (trimmed) | 7.59 | [7.45, 7.74] | n/a |
| Naive, observational arm | 12.94 | [12.84, 13.04] | 5.34 [5.16, 5.52] mismatch |
| Regression adjustment | 10.23 | [10.12, 10.34] | 2.63 [2.45, 2.82] mismatch |
| Inverse propensity weighting | 7.91 | [7.77, 8.05] | 0.32 [0.11, 0.52] match |
| Doubly robust (AIPW) | 7.33 | [7.20, 7.45] | -0.27 [-0.46, -0.08] match |
| Double machine learning | 8.05 | [7.94, 8.16] | 0.46 [0.27, 0.64] match |
