# Univariate ARIMA Analysis

## Scope
- Analysis family: univariate.
- Forecasting target: `TE_8313B.AV_0#`.
- Active granularities: `raw`.
- Chronological 80/20 split.
- One forecast is produced for the full holdout period.
- Nixtla `statsforecast` ARIMA is run as fixed `ARIMA(1, 0, 1)`.

## Metrics
| Granularity | Model | p | d | q | MAE | RMSE | MAPE % | sMAPE % | Bias | Warnings |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | ARIMA_p1_d0_q1 | 1 | 0 | 1 | 7.910 | 9.757 | 1.013 | 1.017 | -3.254 | 0 |

## Initial Reading
- Lowest MAE in this run: `raw` `ARIMA_p1_d0_q1` with MAE `7.910`.
- The raw target scaling artifact prepares the next MLP baseline.