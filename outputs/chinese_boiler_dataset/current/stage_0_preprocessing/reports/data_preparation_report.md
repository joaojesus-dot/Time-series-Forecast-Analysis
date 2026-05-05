# Data Preparation Report

## Preparation Sequence
- Missing-value repair.
- Train-fitted target scaling.
- Aggregation to the modeled granularities.
- First- and second-difference candidate generation.
- Train-only smoothing for MLP candidate selection.

## Missing-Value Repair
- Missing cells before repair: `30`.
- Cells repaired from `data_AutoReg.csv`: `30`.
- The repaired frame becomes the only source for later transformations and forecasts.

## Scaling
- Active scaling method: `standard`.
- Scaling methods compared before forecasting: `minmax, standard`.
- Scaling is fitted on train only so test remains unseen during preprocessing.
- A linear scaler is used so fitted forecasts can be converted back to the original target scale.

## Scaling Diagnostics
| Granularity | Scaling Method | Transform | Std | Min | Max | Skewness |
|---|---|---|---|---|---|---|
| raw | minmax | scaled_level | 0.163 | 0.000 | 1.224 | -0.232 |
| raw | minmax | scaled_first_difference | 0.004 | -0.017 | 0.020 | 0.039 |
| raw | standard | scaled_level | 0.995 | -3.295 | 4.191 | -0.232 |
| raw | standard | scaled_first_difference | 0.025 | -0.106 | 0.121 | 0.039 |
| 30s | minmax | scaled_level | 0.165 | 0.000 | 1.232 | -0.233 |
| 30s | minmax | scaled_first_difference | 0.020 | -0.078 | 0.097 | 0.015 |
| 30s | standard | scaled_level | 0.995 | -3.256 | 4.176 | -0.233 |
| 30s | standard | scaled_first_difference | 0.119 | -0.473 | 0.583 | 0.015 |
| 1min | minmax | scaled_level | 0.166 | 0.000 | 1.236 | -0.235 |
| 1min | minmax | scaled_first_difference | 0.031 | -0.129 | 0.143 | -0.015 |
| 1min | standard | scaled_level | 0.995 | -3.220 | 4.178 | -0.235 |
| 1min | standard | scaled_first_difference | 0.188 | -0.772 | 0.855 | -0.015 |

## Scaling Recommendation
- Recommended method for the next forecasting run: `standard`.
- Train-fitted min-max scaling reaches a test maximum of `1.236`, so future values move outside the nominal `[0, 1]` range.
- Standard scaling keeps the scaled level near unit variance (mean std across granularities `0.995`), which is a stable input scale for the MLP.
- Differenced signals are much less compressed with standard scaling (mean first-difference std `0.111` vs `0.018` for min-max), so the MLP has a stronger learning signal after differencing.
- Both methods are linear and preserve the same temporal shape and skewness; the choice is therefore about numerical conditioning, not changing the signal.

## Aggregation Levels
| Granularity | Rows | Start Date | End Date |
|---|---|---|---|
| raw | 86400 | 2022-03-27 14:28:54 | 2022-04-01 14:28:49 |
| 30s | 14400 | 2022-03-27 14:29:24 | 2022-04-01 14:28:54 |
| 1min | 7200 | 2022-03-27 14:29:54 | 2022-04-01 14:28:54 |

## Resampling Policy
| Key | Value |
|---|---|
| timestamp_column | date |
| input_frequency | 5s |
| label | right |
| closed | left |
| origin | start |
| drop_partial_windows | True |
| default_aggregation | mean |
| column_aggregations.TV_8329ZC.AV_0# | last |
| column_aggregations.YJJWSLL.AV_0# | mean |

## Smoothing Diagnostics
| Granularity | Window | Window Steps | Std |
|---|---|---|---|
| raw | 15s | 3 | 10.680 |
| raw | 30s | 6 | 10.673 |
| raw | 1min | 12 | 10.655 |
| raw | 2min | 24 | 10.608 |
| raw | 3min | 36 | 10.556 |
| 30s | 1min | 2 | 10.655 |
| 30s | 2min | 4 | 10.608 |
| 30s | 3min | 6 | 10.556 |
| 1min | 2min | 2 | 10.608 |
| 1min | 3min | 3 | 10.556 |

## Differencing Diagnostics
| Granularity | Transform | Adf Pvalue | Kpss Pvalue | Stationary Recommended |
|---|---|---|---|---|
| raw | first_difference | 0.000 | 0.100 | True |
| raw | second_difference | 0.000 | 0.100 | True |
| 30s | first_difference | 0.000 | 0.100 | True |
| 30s | second_difference | 0.000 | 0.100 | True |
| 1min | first_difference | 0.000 | 0.100 | True |
| 1min | second_difference | 0.000 | 0.100 | True |