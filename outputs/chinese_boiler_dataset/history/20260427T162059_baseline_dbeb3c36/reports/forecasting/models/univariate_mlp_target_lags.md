# Univariate MLP Target Lags

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
| hidden_layer_sizes | [64, 32] |
| activation | relu |
| max_iter | 300 |
| random_state | 42 |
| write_window_data | False |

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
| Split | Granularity | Mae | Rmse | Mape | Smape | Bias | Warning Count |
|---|---|---|---|---|---|---|---|
| validation | raw | 7.479 | 9.424 | 0.973 | 0.971 | 2.049 | 0 |
| test | raw | 6.837 | 8.803 | 0.879 | 0.879 | -0.408 | 0 |
| validation | 30s | 6.483 | 8.069 | 0.844 | 0.843 | 1.133 | 1 |
| test | 30s | 5.803 | 7.461 | 0.745 | 0.746 | -0.812 | 1 |
| validation | 1min | 5.918 | 7.345 | 0.770 | 0.769 | 0.583 | 0 |
| test | 1min | 5.614 | 7.165 | 0.721 | 0.722 | -1.393 | 0 |
| validation | 5min | 5.475 | 6.947 | 0.713 | 0.711 | 1.222 | 0 |
| test | 5min | 5.060 | 6.653 | 0.649 | 0.650 | -1.059 | 0 |

## Reading Notes
- Uses the shared forecasting protocol and train-fitted scaling.
- Window preparation and model training are separate stages: this report shows both the prepared windows and the fitted-model metrics.
- Validation and test metrics are separated so later tuning does not pollute final comparison.

## Target Scaling
| Granularity | Scaler | Train Rows | Validation Rows | Test Rows | Train Mean | Train Std |
|---|---|---|---|---|---|---|
| raw | standard | 60479 | 8641 | 17280 | 773.534 | 10.800 |
| 30s | standard | 10080 | 1439 | 2881 | 773.534 | 10.791 |
| 1min | standard | 5040 | 719 | 1441 | 773.534 | 10.773 |
| 5min | standard | 1007 | 145 | 288 | 773.534 | 10.565 |

## Window Preparation
| Granularity | Status | Lookback Duration | Lookback Steps | Horizon Duration | Horizon Steps | Train Windows | Validation Windows | Test Windows |
|---|---|---|---|---|---|---|---|---|
| raw | window_data_prepared | 10min | 120 | 10min | 120 | 60240 | 8402 | 17041 |
| 30s | window_data_prepared | 10min | 20 | 10min | 20 | 10041 | 1400 | 2842 |
| 1min | window_data_prepared | 10min | 10 | 10min | 10 | 5021 | 700 | 1422 |
| 5min | window_data_prepared | 10min | 2 | 10min | 2 | 1004 | 142 | 285 |

## Initial Reading
- Lowest MAE in this report: `5min test` with MAE `5.060` and RMSE `6.653`.
- Compare this report against `../model_comparison.md` rather than reading it in isolation.