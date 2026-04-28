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
- Active scaling method: `minmax`.
- Scaling is fitted on train only so test remains unseen during preprocessing.
- A linear scaler is used so fitted forecasts can be converted back to the original target scale.

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