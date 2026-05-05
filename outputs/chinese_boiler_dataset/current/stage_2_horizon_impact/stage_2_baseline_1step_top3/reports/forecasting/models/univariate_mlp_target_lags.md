# Univariate MLP Target Lags

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Forecast horizon: `1step`.
- Lookback window: `3min`.

## Active Configuration
| Key | Value |
|---|---|
| enabled | True |
| engine | neuralforecast |
| hidden_units_strategy | match_lookback |
| activation | relu |
| optimizer | adam |
| learning_rate_grid | [0.0001] |
| min_steps | 5000 |
| max_steps | 7500 |
| num_layers | 1 |
| batch_size | 1 |
| windows_batch_size | 4096 |
| dataloader_num_workers | 0 |
| dataloader_pin_memory | True |
| accelerator | gpu |
| devices | 1 |
| random_state | 42 |
| selection_mode | stage_2_baseline_1step_top3 |
| use_selected_candidate_combinations | True |
| use_selected_learning_rates | False |
| model_variants | [{"name": "neuralforecast_single_hidden_match_lookback", "engine": "neuralforecast", "num_layers": 1, "hidden_units_strategy": "match_lookback"}] |
| selected_candidate_combinations | [{"granularity": "raw", "difference_order": 1, "transform_name": "first_difference", "training_smoothing_window": "none", "role": "best_project_horizon_raw_candidate"}, {"granularity": "30s", "difference_order": 0, "transform_name": "level", "training_smoothing_window": "none", "role": "best_project_horizon_30s_candidate"}, {"granularity": "1min", "difference_order": 1, "transform_name": "first_difference", "training_smoothing_window": "none", "role": "best_project_horizon_1min_candidate"}] |
| candidate_limit | None |
| learning_rate_limit | None |
| write_window_data | False |
| candidate_selection | candidate_per_granularity |
| best_learning_rate_init | 0.0001 |
| forecast_output_scale | scaled |

## Metrics
| Split | Granularity | Mae | Rmse | Mape | Smape | Bias | R2 | Transform Name | Training Smoothing Window | Learning Rate Init | Min Steps | Max Steps | Training Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 0.141 | 0.177 | 194.270 | 38.353 | 0.0004628 | 0.960 | first_difference | none | 0.0001 | 5000 | 7500 | 62.090 |
| test | 30s | 0.100 | 0.126 | 74.357 | 31.048 | -0.0006596 | 0.980 | level | none | 0.0001 | 5000 | 7500 | 61.758 |
| test | raw | 0.010 | 0.012 | 7.638 | 4.857 | -0.0001106 | 1.000 | first_difference | none | 0.0001 | 5000 | 7500 | 80.909 |

## Test Comparison Metrics
| Split | Granularity | Candidate Label | Selection Mode | Difference Order | Training Smoothing Window | Learning Rate Init | Min Steps | Max Steps | Training Seconds | Mae | Rmse | R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1min_d1_smooth_none | stage_2_baseline_1step_top3 | 1 | none | 0.0001 | 5000 | 7500 | 62.090 | 0.141 | 0.177 | 0.960 |
| test | 30s | 30s_d0_smooth_none | stage_2_baseline_1step_top3 | 0 | none | 0.0001 | 5000 | 7500 | 61.758 | 0.100 | 0.126 | 0.980 |
| test | raw | raw_d1_smooth_none | stage_2_baseline_1step_top3 | 1 | none | 0.0001 | 5000 | 7500 | 80.909 | 0.010 | 0.012 | 1.000 |

## Prepared Windows
| Granularity | Candidate Label | Transform Name | Training Smoothing Window | Train Windows | Test Windows |
|---|---|---|---|---|---|
| raw | raw_d0_smooth_none | level | none | 69084 | 17244 |
| raw | raw_d0_smooth_30s | level | 30s | 69079 | 17244 |
| raw | raw_d1_smooth_none | first_difference | none | 69083 | 17243 |
| raw | raw_d1_smooth_30s | first_difference | 30s | 69078 | 17243 |
| raw | raw_d2_smooth_none | second_difference | none | 69082 | 17242 |
| raw | raw_d2_smooth_30s | second_difference | 30s | 69077 | 17242 |
| 30s | 30s_d0_smooth_none | level | none | 11514 | 2874 |
| 30s | 30s_d0_smooth_1min | level | 1min | 11513 | 2874 |
| 30s | 30s_d1_smooth_none | first_difference | none | 11513 | 2873 |
| 30s | 30s_d1_smooth_1min | first_difference | 1min | 11512 | 2873 |
| 30s | 30s_d2_smooth_none | second_difference | none | 11512 | 2872 |
| 30s | 30s_d2_smooth_1min | second_difference | 1min | 11511 | 2872 |
| 1min | 1min_d0_smooth_none | level | none | 5757 | 1437 |
| 1min | 1min_d0_smooth_2min | level | 2min | 5756 | 1437 |
| 1min | 1min_d1_smooth_none | first_difference | none | 5756 | 1436 |
| 1min | 1min_d1_smooth_2min | first_difference | 2min | 5755 | 1436 |
| 1min | 1min_d2_smooth_none | second_difference | none | 5755 | 1435 |
| 1min | 1min_d2_smooth_2min | second_difference | 2min | 5754 | 1435 |