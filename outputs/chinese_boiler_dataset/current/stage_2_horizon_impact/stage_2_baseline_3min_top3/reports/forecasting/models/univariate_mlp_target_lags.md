# Univariate MLP Target Lags

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Forecast horizon: `3min`.
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
| selection_mode | stage_2_baseline_3min_top3 |
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
| test | 1min | 0.303 | 0.384 | 287.657 | 62.908 | 0.003 | 0.810 | first_difference | none | 0.0001 | 5000 | 7500 | 92.594 |
| test | 30s | 0.311 | 0.393 | 146.659 | 66.945 | -0.040 | 0.802 | level | none | 0.0001 | 5000 | 7500 | 168.677 |
| test | raw | 0.310 | 0.392 | 214.262 | 64.629 | -0.005 | 0.803 | first_difference | none | 0.0001 | 5000 | 7500 | 32022.668 |

## Test Comparison Metrics
| Split | Granularity | Candidate Label | Selection Mode | Difference Order | Training Smoothing Window | Learning Rate Init | Min Steps | Max Steps | Training Seconds | Mae | Rmse | R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1min_d1_smooth_none | stage_2_baseline_3min_top3 | 1 | none | 0.0001 | 5000 | 7500 | 92.594 | 0.303 | 0.384 | 0.810 |
| test | 30s | 30s_d0_smooth_none | stage_2_baseline_3min_top3 | 0 | none | 0.0001 | 5000 | 7500 | 168.677 | 0.311 | 0.393 | 0.802 |
| test | raw | raw_d1_smooth_none | stage_2_baseline_3min_top3 | 1 | none | 0.0001 | 5000 | 7500 | 32022.668 | 0.310 | 0.392 | 0.803 |

## Prepared Windows
| Granularity | Candidate Label | Transform Name | Training Smoothing Window | Train Windows | Test Windows |
|---|---|---|---|---|---|
| raw | raw_d0_smooth_none | level | none | 69049 | 17209 |
| raw | raw_d0_smooth_30s | level | 30s | 69044 | 17209 |
| raw | raw_d1_smooth_none | first_difference | none | 69048 | 17208 |
| raw | raw_d1_smooth_30s | first_difference | 30s | 69043 | 17208 |
| raw | raw_d2_smooth_none | second_difference | none | 69047 | 17207 |
| raw | raw_d2_smooth_30s | second_difference | 30s | 69042 | 17207 |
| 30s | 30s_d0_smooth_none | level | none | 11509 | 2869 |
| 30s | 30s_d0_smooth_1min | level | 1min | 11508 | 2869 |
| 30s | 30s_d1_smooth_none | first_difference | none | 11508 | 2868 |
| 30s | 30s_d1_smooth_1min | first_difference | 1min | 11507 | 2868 |
| 30s | 30s_d2_smooth_none | second_difference | none | 11507 | 2867 |
| 30s | 30s_d2_smooth_1min | second_difference | 1min | 11506 | 2867 |
| 1min | 1min_d0_smooth_none | level | none | 5755 | 1435 |
| 1min | 1min_d0_smooth_2min | level | 2min | 5754 | 1435 |
| 1min | 1min_d1_smooth_none | first_difference | none | 5754 | 1434 |
| 1min | 1min_d1_smooth_2min | first_difference | 2min | 5753 | 1434 |
| 1min | 1min_d2_smooth_none | second_difference | none | 5753 | 1433 |
| 1min | 1min_d2_smooth_2min | second_difference | 2min | 5752 | 1433 |