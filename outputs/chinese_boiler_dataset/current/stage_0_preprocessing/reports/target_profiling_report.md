# Target Profiling Report

## Goal
- Profile target `TE_8313B.AV_0#` in terms of granularity, autocorrelation, and stationarity.
- Summarize which non-target variables move most strongly with the target.

## Granularity Summary
| Id Key | Granularity | Rows | Start Date | End Date |
|---|---|---|---|---|
| subset_B_raw | raw | 86400 | 2022-03-27 14:28:54 | 2022-04-01 14:28:49 |
| subset_B_30s | 30s | 14400 | 2022-03-27 14:29:24 | 2022-04-01 14:28:54 |
| subset_B_1min | 1min | 7200 | 2022-03-27 14:29:54 | 2022-04-01 14:28:54 |

## Autocorrelation Summary
| Granularity | Frequency | Acf 30S | Acf 1Min | Acf 5Min | Acf 10Min |
|---|---|---|---|---|---|
| raw | 5s | 0.992 | 0.977 | 0.846 | 0.716 |
| 30s | 30s | 0.993 | 0.979 | 0.847 | 0.717 |
| 1min | 1min | 0.982 | 0.982 | 0.850 | 0.720 |

## Stationarity Summary
| Granularity | Transform | Adf Pvalue | Kpss Pvalue | Stationary Recommended |
|---|---|---|---|---|
| raw | original | 2.718e-21 | 0.010 | False |
| raw | first_difference | 0.000 | 0.100 | True |
| raw | second_difference | 0.000 | 0.100 | True |
| 30s | original | 1.41e-12 | 0.054 | True |
| 30s | first_difference | 0.000 | 0.100 | True |
| 30s | second_difference | 0.000 | 0.100 | True |
| 1min | original | 3.155e-09 | 0.100 | True |
| 1min | first_difference | 0.000 | 0.100 | True |
| 1min | second_difference | 0.000 | 0.100 | True |

## Strongest Target Correlations
| Granularity | Variable | Correlation | Abs Correlation |
|---|---|---|---|
| 1min | ZZQBCHLL.AV_0# | 0.877 | 0.877 |
| 1min | YFJ3_AI.AV_0# | 0.751 | 0.751 |
| 1min | TE_8319A.AV_0# | 0.586 | 0.586 |
| 1min | TE_8303.AV_0# | 0.397 | 0.397 |
| 1min | PTCA_8322A.AV_0# | 0.395 | 0.395 |
| 30s | ZZQBCHLL.AV_0# | 0.872 | 0.872 |
| 30s | YFJ3_AI.AV_0# | 0.742 | 0.742 |
| 30s | TE_8319A.AV_0# | 0.585 | 0.585 |
| 30s | TE_8303.AV_0# | 0.396 | 0.396 |
| 30s | PTCA_8322A.AV_0# | 0.394 | 0.394 |
| raw | ZZQBCHLL.AV_0# | 0.869 | 0.869 |
| raw | YFJ3_AI.AV_0# | 0.652 | 0.652 |
| raw | TE_8319A.AV_0# | 0.584 | 0.584 |
| raw | TE_8303.AV_0# | 0.396 | 0.396 |
| raw | PTCA_8322A.AV_0# | 0.393 | 0.393 |