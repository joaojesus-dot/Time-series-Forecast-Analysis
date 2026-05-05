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
| selection_mode | stage_3_univariate_mlp_3min |
| use_selected_candidate_combinations | True |
| use_selected_learning_rates | False |
| model_variants | [{"name": "neuralforecast_2_hidden_match_lookback", "engine": "neuralforecast", "num_layers": 2, "hidden_units_strategy": "match_lookback"}, {"name": "neuralforecast_2_hidden_fixed_16", "engine": "neuralforecast", "num_layers": 2, "hidden_units_strategy": "fixed", "hidden_units": 16}] |
| selected_candidate_combinations | [{"granularity": "raw", "difference_order": 1, "transform_name": "first_difference", "training_smoothing_window": "none", "role": "best_project_horizon_raw_candidate"}, {"granularity": "30s", "difference_order": 0, "transform_name": "level", "training_smoothing_window": "none", "role": "best_project_horizon_30s_candidate"}, {"granularity": "1min", "difference_order": 1, "transform_name": "first_difference", "training_smoothing_window": "none", "role": "best_project_horizon_1min_candidate"}] |
| candidate_limit | None |
| learning_rate_limit | None |
| write_window_data | False |
| candidate_selection | candidate_per_granularity |
| candidate_settings.raw_d1_smooth_none.learning_rate_init | 0.0001 |
| candidate_settings.raw_d1_smooth_none.min_steps | 5000 |
| candidate_settings.raw_d1_smooth_none.max_steps | 5000 |
| candidate_settings.30s_d0_smooth_none.learning_rate_init | 0.01 |
| candidate_settings.30s_d0_smooth_none.min_steps | 5000 |
| candidate_settings.30s_d0_smooth_none.max_steps | 12000 |
| candidate_settings.1min_d1_smooth_none.learning_rate_init | 0.01 |
| candidate_settings.1min_d1_smooth_none.min_steps | 5000 |
| candidate_settings.1min_d1_smooth_none.max_steps | 5000 |
| forecast_output_scale | scaled |

## Metrics
| Split | Granularity | Mae | Rmse | Mape | Smape | Bias | R2 | Transform Name | Training Smoothing Window | Learning Rate Init | Min Steps | Max Steps | Training Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 0.306 | 0.388 | 282.191 | 62.638 | 0.004 | 0.806 | first_difference | none | 0.010 | 5000 | 5000 | 83.832 |
| test | 1min | 0.303 | 0.384 | 293.529 | 62.922 | 0.003 | 0.810 | first_difference | none | 0.010 | 5000 | 5000 | 82.992 |
| test | 30s | 0.309 | 0.391 | 162.169 | 66.226 | -0.055 | 0.804 | level | none | 0.010 | 5000 | 12000 | 235.673 |
| test | 30s | 0.304 | 0.383 | 153.591 | 65.568 | -0.033 | 0.811 | level | none | 0.010 | 5000 | 12000 | 233.517 |
| test | raw | 0.310 | 0.394 | 213.659 | 64.736 | -0.005 | 0.801 | first_difference | none | 0.0001 | 5000 | 5000 | 3377.593 |
| test | raw | 0.310 | 0.393 | 214.055 | 64.638 | -0.005 | 0.802 | first_difference | none | 0.0001 | 5000 | 5000 | 4771.502 |

## Test Comparison Metrics
| Split | Granularity | Candidate Label | Selection Mode | Difference Order | Training Smoothing Window | Learning Rate Init | Min Steps | Max Steps | Training Seconds | Mae | Rmse | R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1min_d1_smooth_none | stage_3_univariate_mlp_3min | 1 | none | 0.010 | 5000 | 5000 | 83.832 | 0.306 | 0.388 | 0.806 |
| test | 1min | 1min_d1_smooth_none | stage_3_univariate_mlp_3min | 1 | none | 0.010 | 5000 | 5000 | 82.992 | 0.303 | 0.384 | 0.810 |
| test | 30s | 30s_d0_smooth_none | stage_3_univariate_mlp_3min | 0 | none | 0.010 | 5000 | 12000 | 235.673 | 0.309 | 0.391 | 0.804 |
| test | 30s | 30s_d0_smooth_none | stage_3_univariate_mlp_3min | 0 | none | 0.010 | 5000 | 12000 | 233.517 | 0.304 | 0.383 | 0.811 |
| test | raw | raw_d1_smooth_none | stage_3_univariate_mlp_3min | 1 | none | 0.0001 | 5000 | 5000 | 3377.593 | 0.310 | 0.394 | 0.801 |
| test | raw | raw_d1_smooth_none | stage_3_univariate_mlp_3min | 1 | none | 0.0001 | 5000 | 5000 | 4771.502 | 0.310 | 0.393 | 0.802 |

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