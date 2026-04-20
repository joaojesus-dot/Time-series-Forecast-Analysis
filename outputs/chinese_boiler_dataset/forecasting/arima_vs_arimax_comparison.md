# ARIMA vs ARIMAX Comparison

## Scope
- Compares fixed-order univariate ARIMA against fixed-order ARIMAX on overlapping granularities.
- Negative delta means ARIMAX improved the metric.
- ARIMAX uses observed future exogenous values from the test period, so this is an oracle-style evaluation unless those variables are known or forecasted at prediction time.

## Metrics
| Granularity | d | ARIMA MAE | ARIMAX MAE | MAE Change % | ARIMA RMSE | ARIMAX RMSE | RMSE Change % |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1min | 0 | 8.298 | 6.000 | -27.7 | 10.460 | 7.451 | -28.8 |
| 1min | 1 | 10.352 | 7.661 | -26.0 | 12.611 | 9.705 | -23.0 |
| 1min | 2 | 75.105 | 10.147 | -86.5 | 84.579 | 12.291 | -85.5 |
| 30s | 0 | 8.280 | 6.720 | -18.8 | 10.433 | 8.251 | -20.9 |
| 30s | 1 | 10.875 | 7.970 | -26.7 | 13.153 | 9.759 | -25.8 |
| 30s | 2 | 33.984 | 19.820 | -41.7 | 37.216 | 22.068 | -40.7 |
| 5min | 0 | 8.194 | 3.724 | -54.6 | 10.280 | 4.739 | -53.9 |
| 5min | 1 | 14.644 | 4.512 | -69.2 | 16.791 | 5.418 | -67.7 |
| 5min | 2 | 124.303 | 9.826 | -92.1 | 139.415 | 11.296 | -91.9 |

## Initial Reading
- Largest MAE improvement: `5min` with `d=2`, from `124.303` to `9.826`.
- `d=0` remains the best differencing choice among the tested ARIMAX models.