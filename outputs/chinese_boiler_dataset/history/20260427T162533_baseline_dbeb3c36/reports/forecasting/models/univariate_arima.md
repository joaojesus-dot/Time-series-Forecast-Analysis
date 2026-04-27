# Univariate ARIMA

## Scope
- Analysis family: `univariate`.
- Forecasting target: `TE_8313B.AV_0#`.
- Report status: `evaluated`.
- Target horizon: `10min`.
- Active granularities: `raw, 30s, 1min, 5min`.
- This report follows the shared organizer contract: scope, configuration, comparable metrics, and interpretation.

## Active Configuration
| Key | Value |
|---|---|
| enabled | True |
| order | [1, 0, 1] |
| season_length | 1 |
| seasonal_order | [0, 0, 0] |
| method | CSS-ML |

## Shared Protocol Snapshot
| Key | Value |
|---|---|
| dataset_prefix | subset_B |
| forecast_horizon | 10min |
| lookback_window | 10min |
| target_granularities | ["raw", "30s", "1min", "5min"] |
| splits.train | 0.7 |
| splits.validation | 0.1 |
| splits.test | 0.2 |
| metrics | ["mae", "rmse", "mape", "smape", "bias"] |

## Comparable Metrics
| Granularity | P | D | Q | Mae | Rmse | Mape | Smape | Bias | Warning Count |
|---|---|---|---|---|---|---|---|---|---|
| 1min | 1 | 0 | 1 | 8.287 | 10.446 | 1.059 | 1.065 | -4.518 | 0 |
| 30s | 1 | 0 | 1 | 8.280 | 10.435 | 1.058 | 1.064 | -4.501 | 0 |
| 5min | 1 | 0 | 1 | 8.194 | 10.280 | 1.047 | 1.053 | -4.520 | 0 |
| raw | 1 | 0 | 1 | 7.910 | 9.757 | 1.013 | 1.017 | -3.254 | 0 |

## Reading Notes
- Uses the shared forecasting protocol and reports holdout test metrics.
- Acts as the fixed-order statistical baseline for equal-footing comparison.

## Initial Reading
- Lowest MAE in this report: `raw` with MAE `7.910` and RMSE `9.757`.
- Compare this report against `../model_comparison.md` rather than reading it in isolation.