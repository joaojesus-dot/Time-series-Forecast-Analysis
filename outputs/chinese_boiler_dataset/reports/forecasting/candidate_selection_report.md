# Candidate Selection Report

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Review dataset: chronological test block from the `20%` holdout split.
- Forecast horizon used for this review: `1step`.
- This is a chronological test-block candidate review, not random resampling or a separate model-selection split.
- `1step` means one row ahead at each granularity, so raw, 30s, and 1min scores are useful for within-granularity ranking but are not equal real-time forecast horizons.

## Selected Candidate Combinations
| Granularity | Candidate Label | Role | Transform Name | Training Smoothing Window | Learning Rate Init | Mae | Rmse | R2 |
|---|---|---|---|---|---|---|---|---|
| raw | raw_d1_smooth_none | best_short_horizon_baseline | first_difference | none | 0.0001 | 0.104 | 0.132 | 1.000 |
| raw | raw_d2_smooth_none | raw_stationary_alternative | second_difference | none | 0.0001 | 0.105 | 0.132 | 1.000 |
| 30s | 30s_d1_smooth_none | best_mid_granularity_candidate | first_difference | none | 0.0001 | 0.876 | 1.107 | 0.986 |
| 30s | 30s_d0_smooth_none | 30s_level_alternative | level | none | 0.001 | 0.878 | 1.109 | 0.986 |
| 1min | 1min_d1_smooth_none | best_coarse_granularity_candidate | first_difference | none | 0.0001 | 1.508 | 1.891 | 0.960 |
| 1min | 1min_d0_smooth_none | 1min_level_alternative | level | none | 0.010 | 1.512 | 1.903 | 0.960 |

## Metric Evidence
- The selected combinations keep the best two candidates per granularity so the next stage can compare raw, 30s, and 1min behavior without mixing unequal one-step horizons into a single final decision.
- All selected candidates use `standard` scaling, matching the preprocessing recommendation.
- No selected candidate uses train-only smoothing because smoothing consistently worsened one-step test metrics.
- `0.0001` is retained as the main learning rate because it produced the best candidate in each granularity; `0.001` and `0.01` are retained only where they won a level-transformed alternative.

## Best Three Per Granularity
| Granularity | Candidate Label | Transform Name | Training Smoothing Window | Learning Rate Init | Mae | Rmse | R2 |
|---|---|---|---|---|---|---|---|
| 1min | 1min_d1_smooth_none | first_difference | none | 0.0001 | 1.508 | 1.891 | 0.960 |
| 1min | 1min_d0_smooth_none | level | none | 0.010 | 1.512 | 1.903 | 0.960 |
| 1min | 1min_d1_smooth_none | first_difference | none | 0.005 | 1.513 | 1.899 | 0.960 |
| 30s | 30s_d1_smooth_none | first_difference | none | 0.0001 | 0.876 | 1.107 | 0.986 |
| 30s | 30s_d0_smooth_none | level | none | 0.001 | 0.878 | 1.109 | 0.986 |
| 30s | 30s_d0_smooth_none | level | none | 0.010 | 0.886 | 1.118 | 0.986 |
| raw | raw_d1_smooth_none | first_difference | none | 0.0001 | 0.104 | 0.132 | 1.000 |
| raw | raw_d2_smooth_none | second_difference | none | 0.0001 | 0.105 | 0.132 | 1.000 |
| raw | raw_d1_smooth_none | first_difference | none | 0.001 | 0.107 | 0.135 | 1.000 |

## Smoothing Evidence
| Granularity | Training Smoothing Window | Mean Mae | Mean Rmse | Mean R2 |
|---|---|---|---|---|
| 1min | 2min | 2.543 | 3.201 | 0.882 |
| 1min | none | 1.558 | 1.958 | 0.957 |
| 30s | 1min | 1.631 | 2.047 | 0.949 |
| 30s | none | 0.908 | 1.147 | 0.985 |
| raw | 30s | 0.259 | 0.328 | 0.999 |
| raw | none | 0.133 | 0.168 | 1.000 |

## Transform Evidence
| Granularity | Transform Name | Mean Mae | Mean Rmse | Mean R2 |
|---|---|---|---|---|
| 1min | first_difference | 2.042 | 2.550 | 0.922 |
| 1min | level | 1.813 | 2.288 | 0.939 |
| 1min | second_difference | 2.295 | 2.901 | 0.897 |
| 30s | first_difference | 1.377 | 1.729 | 0.962 |
| 30s | level | 0.964 | 1.219 | 0.983 |
| 30s | second_difference | 1.466 | 1.844 | 0.957 |
| raw | first_difference | 0.155 | 0.194 | 1.000 |
| raw | level | 0.236 | 0.299 | 0.999 |
| raw | second_difference | 0.198 | 0.251 | 0.999 |

## Learning-Rate Evidence
| Learning Rate Init | Mean Mae | Mean Rmse | Mean R2 |
|---|---|---|---|
| 0.0001 | 1.054 | 1.326 | 0.970 |
| 0.001 | 1.209 | 1.523 | 0.959 |
| 0.005 | 1.225 | 1.544 | 0.958 |
| 0.010 | 1.200 | 1.506 | 0.961 |

## Next Stage
- Use the selected combinations as the reduced candidate set for the next forecasting stage.
- The final project objective still requires a `10min` horizon run. At that stage, raw, 30s, and 1min will correspond to 120, 20, and 10 forecast steps respectively.
- These one-step results justify preprocessing and candidate reduction; they should not be presented as the final 10-minute forecasting result.