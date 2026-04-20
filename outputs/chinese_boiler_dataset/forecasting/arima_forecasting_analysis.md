# ARIMA Forecasting Analysis

## Scope
- Forecasting target: `TE_8313B.AV_0#`.
- Chronological 80/20 split.
- One forecast is produced for the full holdout period of each granularity.
- Nixtla `statsforecast` ARIMA is run as fixed `ARIMA(1, d, 1)`.
- Tested differencing orders: `d=0`, `d=1`, and `d=2`; `d=2` is the second-difference case.

## Metrics
| Granularity | Model | p | d | q | MAE | RMSE | MAPE % | sMAPE % | Bias | Warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | ARIMA_p1_d0_q1 | 1 | 0 | 1 | 7.910 | 9.757 | 1.013 | 1.017 | -3.254 | 0 |
| raw | ARIMA_p1_d1_q1 | 1 | 1 | 1 | 10.676 | 12.935 | 1.381 | 1.369 | 8.850 | 0 |
| raw | ARIMA_p1_d2_q1 | 1 | 2 | 1 | 648.289 | 751.272 | 83.263 | 125.944 | -648.197 | 0 |
| 30s | ARIMA_p1_d0_q1 | 1 | 0 | 1 | 8.280 | 10.433 | 1.058 | 1.064 | -4.488 | 0 |
| 30s | ARIMA_p1_d1_q1 | 1 | 1 | 1 | 10.875 | 13.153 | 1.407 | 1.394 | 9.172 | 0 |
| 30s | ARIMA_p1_d2_q1 | 1 | 2 | 1 | 33.984 | 37.216 | 4.378 | 4.267 | 33.225 | 0 |
| 1min | ARIMA_p1_d0_q1 | 1 | 0 | 1 | 8.298 | 10.460 | 1.060 | 1.067 | -4.560 | 0 |
| 1min | ARIMA_p1_d1_q1 | 1 | 1 | 1 | 10.352 | 12.611 | 1.339 | 1.328 | 8.399 | 0 |
| 1min | ARIMA_p1_d2_q1 | 1 | 2 | 1 | 75.105 | 84.579 | 9.662 | 9.108 | 74.766 | 0 |
| 5min | ARIMA_p1_d0_q1 | 1 | 0 | 1 | 8.194 | 10.280 | 1.047 | 1.053 | -4.520 | 0 |
| 5min | ARIMA_p1_d1_q1 | 1 | 1 | 1 | 14.644 | 16.791 | 1.894 | 1.871 | 14.055 | 0 |
| 5min | ARIMA_p1_d2_q1 | 1 | 2 | 1 | 124.303 | 139.415 | 15.983 | 14.538 | 124.273 | 0 |

## Initial Reading
- Lowest MAE in this run: `raw` `ARIMA_p1_d0_q1` with MAE `7.910`.
- These are initial holdout results, not yet rolling-origin validation.
- Convergence warnings are counted in the metrics table and should be reviewed before treating a model as final.