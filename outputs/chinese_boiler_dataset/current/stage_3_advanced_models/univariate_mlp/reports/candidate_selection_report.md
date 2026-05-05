# Candidate Selection Report

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Review dataset: chronological test block from the `20%` holdout split.
- Forecast horizon used for this review: `3min`.
- This is a chronological test-block candidate review, not random resampling or a separate model-selection split.
- `1step` means one row ahead at each granularity, so raw, 30s, and 1min scores are useful for within-granularity ranking but are not equal real-time forecast horizons.

## Selected Candidate Combinations
| Granularity | Candidate Label | Role | Transform Name | Training Smoothing Window | Learning Rate Init | Mae | Rmse | R2 |
|---|---|---|---|---|---|---|---|---|
| raw | raw_d1_smooth_none | best_project_horizon_raw_candidate | first_difference | none | 0.0001 | 0.310 | 0.393 | 0.802 |
| 30s | 30s_d0_smooth_none | best_project_horizon_30s_candidate | level | none | 0.010 | 0.304 | 0.383 | 0.811 |
| 1min | 1min_d1_smooth_none | best_project_horizon_1min_candidate | first_difference | none | 0.010 | 0.303 | 0.384 | 0.810 |

## Metric Evidence
- The selected combinations keep the best candidates per granularity so later stages can compare raw, 30s, and 1min behavior without mixing unequal one-step horizons into a single final decision.
- All selected candidates use `standard` scaling, matching the preprocessing recommendation.
- Smoothing candidates are optional preprocessing alternatives; when included, they should be compared against the no-smoothing candidates for the same granularity.
- Learning rate is treated as an MLP hyperparameter, not as part of the preprocessing candidate identity; selected rows below show the best observed LR only as supporting evidence.

## Best Three Per Granularity
| Granularity | Candidate Label | Transform Name | Training Smoothing Window | Learning Rate Init | Mae | Rmse | R2 |
|---|---|---|---|---|---|---|---|
| 1min | 1min_d1_smooth_none | first_difference | none | 0.010 | 0.303 | 0.384 | 0.810 |
| 1min | 1min_d1_smooth_none | first_difference | none | 0.010 | 0.306 | 0.388 | 0.806 |
| 30s | 30s_d0_smooth_none | level | none | 0.010 | 0.304 | 0.383 | 0.811 |
| 30s | 30s_d0_smooth_none | level | none | 0.010 | 0.309 | 0.391 | 0.804 |
| raw | raw_d1_smooth_none | first_difference | none | 0.0001 | 0.310 | 0.393 | 0.802 |
| raw | raw_d1_smooth_none | first_difference | none | 0.0001 | 0.310 | 0.394 | 0.801 |

## Smoothing Evidence
| Granularity | Training Smoothing Window | Mean Mae | Mean Rmse | Mean R2 |
|---|---|---|---|---|
| 1min | none | 0.304 | 0.386 | 0.808 |
| 30s | none | 0.306 | 0.387 | 0.808 |
| raw | none | 0.310 | 0.393 | 0.802 |

## Transform Evidence
| Granularity | Transform Name | Mean Mae | Mean Rmse | Mean R2 |
|---|---|---|---|---|
| 1min | first_difference | 0.304 | 0.386 | 0.808 |
| 30s | level | 0.306 | 0.387 | 0.808 |
| raw | first_difference | 0.310 | 0.393 | 0.802 |

## Learning-Rate Evidence
| Learning Rate Init | Mean Mae | Mean Rmse | Mean R2 |
|---|---|---|---|
| 0.0001 | 0.310 | 0.393 | 0.802 |
| 0.010 | 0.305 | 0.387 | 0.808 |

## Next Stage
- Use the selected combinations as the reduced candidate set for the next forecasting stage.
- The project-objective horizon is `3min`. At that horizon, forecast steps are: raw=36, 30s=6, 1min=3.
- One-step results justify preprocessing and candidate reduction; they should not be presented as the final `3min` forecasting result.