# ARIMAX Forecasting Analysis

## Scope
- Forecasting target: `TE_8313B.AV_0#`.
- Chronological 80/20 split.
- One forecast is produced for the full holdout period of each granularity.
- Nixtla `statsforecast` ARIMAX is run as fixed `ARIMA(1, d, 1)`.
- Tested differencing orders: `d=0`, `d=1`, and `d=2`; `d=2` is the second-difference case.
- Multivariate run: the non-target Candidate B variables are used as exogenous regressors.
- Future test-period exogenous values are supplied to `statsforecast` for evaluation.
- Exogenous regressors are standardized using training-split statistics only.
- Granularities above `12000` training rows are skipped for this API to avoid dense-SVD memory failures.
- Skipped `raw` ARIMAX: `69120` training rows exceeds `12000`.

## Metrics
| Granularity | Model | p | d | q | MAE | RMSE | MAPE % | sMAPE % | Bias | Warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30s | ARIMAX_p1_d0_q1 | 1 | 0 | 1 | 6.720 | 8.251 | 0.867 | 0.863 | 2.851 | 0 |
| 30s | ARIMAX_p1_d1_q1 | 1 | 1 | 1 | 7.970 | 9.759 | 1.027 | 1.023 | 3.438 | 0 |
| 30s | ARIMAX_p1_d2_q1 | 1 | 2 | 1 | 19.820 | 22.068 | 2.554 | 2.515 | 18.424 | 0 |
| 1min | ARIMAX_p1_d0_q1 | 1 | 0 | 1 | 6.000 | 7.451 | 0.774 | 0.771 | 2.691 | 0 |
| 1min | ARIMAX_p1_d1_q1 | 1 | 1 | 1 | 7.661 | 9.705 | 0.986 | 0.983 | 2.791 | 0 |
| 1min | ARIMAX_p1_d2_q1 | 1 | 2 | 1 | 10.147 | 12.291 | 1.308 | 1.298 | 7.243 | 0 |
| 5min | ARIMAX_p1_d0_q1 | 1 | 0 | 1 | 3.724 | 4.739 | 0.480 | 0.478 | 2.792 | 0 |
| 5min | ARIMAX_p1_d1_q1 | 1 | 1 | 1 | 4.512 | 5.418 | 0.579 | 0.581 | -3.096 | 0 |
| 5min | ARIMAX_p1_d2_q1 | 1 | 2 | 1 | 9.826 | 11.296 | 1.260 | 1.270 | -8.868 | 0 |

## Initial Reading
- Lowest MAE in this run: `5min` `ARIMAX_p1_d0_q1` with MAE `3.724`.
- These are initial holdout results, not yet rolling-origin validation.
- Convergence warnings are counted in the metrics table and should be reviewed before treating a model as final.