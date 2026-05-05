# Univariate MLP Target Lags

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Forecast horizon: `1step`.
- Lookback window: `10min`.

## Active Configuration
| Key | Value |
|---|---|
| enabled | True |
| engine | neuralforecast |
| hidden_units_strategy | match_lookback |
| activation | relu |
| optimizer | adam |
| learning_rate_grid | [0.0001, 0.001, 0.005, 0.01] |
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
| selection_mode | all_candidates_1step_review |
| use_selected_candidate_combinations | False |
| selected_candidate_combinations | [{"granularity": "raw", "difference_order": 1, "transform_name": "first_difference", "training_smoothing_window": "none", "learning_rate_init": 0.0001, "role": "best_short_horizon_baseline"}, {"granularity": "raw", "difference_order": 2, "transform_name": "second_difference", "training_smoothing_window": "none", "learning_rate_init": 0.0001, "role": "raw_stationary_alternative"}, {"granularity": "30s", "difference_order": 1, "transform_name": "first_difference", "training_smoothing_window": "none", "learning_rate_init": 0.0001, "role": "best_mid_granularity_candidate"}, {"granularity": "30s", "difference_order": 0, "transform_name": "level", "training_smoothing_window": "none", "learning_rate_init": 0.001, "role": "30s_level_alternative"}, {"granularity": "1min", "difference_order": 1, "transform_name": "first_difference", "training_smoothing_window": "none", "learning_rate_init": 0.0001, "role": "best_coarse_granularity_candidate"}, {"granularity": "1min", "difference_order": 0, "transform_name": "level", "training_smoothing_window": "none", "learning_rate_init": 0.01, "role": "1min_level_alternative"}] |
| candidate_limit | None |
| learning_rate_limit | None |
| write_window_data | False |

## Metrics
| Split | Granularity | Mae | Rmse | Mape | Smape | Bias | R2 | Transform Name | Training Smoothing Window | Learning Rate Init | Min Steps | Max Steps | Training Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1.573 | 1.973 | 0.202 | 0.202 | -0.067 | 0.957 | level | 2min | 0.0001 | 5000 | 7500 | 274.929 |
| test | 1min | 2.237 | 2.839 | 0.288 | 0.288 | -0.022 | 0.910 | level | 2min | 0.001 | 5000 | 7500 | 66.153 |
| test | 1min | 2.533 | 3.197 | 0.326 | 0.326 | -0.032 | 0.886 | level | 2min | 0.005 | 5000 | 7500 | 65.591 |
| test | 1min | 2.548 | 3.202 | 0.328 | 0.328 | -0.089 | 0.886 | level | 2min | 0.010 | 5000 | 7500 | 66.101 |
| test | 1min | 1.540 | 1.930 | 0.198 | 0.198 | -0.079 | 0.958 | level | none | 0.0001 | 5000 | 7500 | 78.413 |
| test | 1min | 1.519 | 1.915 | 0.195 | 0.195 | -0.074 | 0.959 | level | none | 0.001 | 5000 | 7500 | 77.310 |
| test | 1min | 1.517 | 1.908 | 0.195 | 0.195 | -0.047 | 0.959 | level | none | 0.005 | 5000 | 7500 | 84.032 |
| test | 1min | 1.516 | 1.907 | 0.195 | 0.195 | -0.113 | 0.959 | level | none | 0.010 | 5000 | 7500 | 72.622 |
| test | 1min | 2.635 | 3.282 | 0.339 | 0.339 | -0.185 | 0.880 | first_difference | 2min | 0.0001 | 5000 | 7500 | 66.007 |
| test | 1min | 2.691 | 3.352 | 0.346 | 0.346 | -0.096 | 0.875 | first_difference | 2min | 0.001 | 5000 | 7500 | 65.626 |
| test | 1min | 2.669 | 3.338 | 0.343 | 0.344 | -0.363 | 0.876 | first_difference | 2min | 0.005 | 5000 | 7500 | 65.953 |
| test | 1min | 2.701 | 3.355 | 0.347 | 0.348 | -0.154 | 0.875 | first_difference | 2min | 0.010 | 5000 | 7500 | 65.938 |
| test | 1min | 1.512 | 1.898 | 0.194 | 0.194 | 0.026 | 0.960 | first_difference | none | 0.0001 | 5000 | 7500 | 67.922 |
| test | 1min | 1.515 | 1.898 | 0.195 | 0.195 | 0.022 | 0.960 | first_difference | none | 0.001 | 5000 | 7500 | 66.074 |
| test | 1min | 1.524 | 1.913 | 0.196 | 0.196 | 0.041 | 0.959 | first_difference | none | 0.005 | 5000 | 7500 | 66.372 |
| test | 1min | 1.510 | 1.896 | 0.194 | 0.194 | 0.053 | 0.960 | first_difference | none | 0.010 | 5000 | 7500 | 65.838 |
| test | 1min | 2.896 | 3.623 | 0.372 | 0.373 | -0.354 | 0.854 | second_difference | 2min | 0.0001 | 5000 | 7500 | 65.438 |
| test | 1min | 3.150 | 4.000 | 0.405 | 0.405 | -0.149 | 0.822 | second_difference | 2min | 0.001 | 5000 | 7500 | 65.584 |
| test | 1min | 3.117 | 3.990 | 0.401 | 0.401 | -0.327 | 0.823 | second_difference | 2min | 0.005 | 5000 | 7500 | 65.701 |
| test | 1min | 2.695 | 3.367 | 0.347 | 0.347 | 0.225 | 0.874 | second_difference | 2min | 0.010 | 5000 | 7500 | 65.780 |
| test | 1min | 1.613 | 2.029 | 0.207 | 0.207 | -0.0005503 | 0.954 | second_difference | none | 0.0001 | 5000 | 7500 | 65.239 |
| test | 1min | 1.617 | 2.047 | 0.208 | 0.208 | 0.026 | 0.953 | second_difference | none | 0.001 | 5000 | 7500 | 65.034 |
| test | 1min | 1.635 | 2.043 | 0.210 | 0.210 | -0.023 | 0.953 | second_difference | none | 0.005 | 5000 | 7500 | 67.545 |
| test | 1min | 1.627 | 2.054 | 0.209 | 0.209 | -0.018 | 0.953 | second_difference | none | 0.010 | 5000 | 7500 | 65.722 |
| test | 30s | 0.917 | 1.156 | 0.118 | 0.118 | -0.048 | 0.985 | level | 1min | 0.0001 | 5000 | 7500 | 69.857 |
| test | 30s | 1.075 | 1.365 | 0.138 | 0.138 | 0.008 | 0.979 | level | 1min | 0.001 | 5000 | 7500 | 86.470 |
| test | 30s | 1.143 | 1.456 | 0.147 | 0.147 | -0.119 | 0.976 | level | 1min | 0.005 | 5000 | 7500 | 84.227 |
| test | 30s | 1.123 | 1.428 | 0.144 | 0.144 | -0.052 | 0.977 | level | 1min | 0.010 | 5000 | 7500 | 70.614 |
| test | 30s | 0.908 | 1.145 | 0.117 | 0.117 | -0.074 | 0.985 | level | none | 0.0001 | 5000 | 7500 | 79.935 |
| test | 30s | 0.885 | 1.114 | 0.114 | 0.114 | -0.087 | 0.986 | level | none | 0.001 | 5000 | 7500 | 79.948 |
| test | 30s | 0.884 | 1.120 | 0.114 | 0.114 | 0.020 | 0.986 | level | none | 0.005 | 5000 | 7500 | 75.104 |
| test | 30s | 0.884 | 1.111 | 0.114 | 0.114 | -0.010 | 0.986 | level | none | 0.010 | 5000 | 7500 | 86.206 |
| test | 30s | 1.532 | 1.936 | 0.197 | 0.197 | -0.050 | 0.958 | first_difference | 1min | 0.0001 | 5000 | 7500 | 82.244 |
| test | 30s | 2.173 | 2.727 | 0.279 | 0.279 | -0.096 | 0.917 | first_difference | 1min | 0.001 | 5000 | 7500 | 80.603 |
| test | 30s | 2.081 | 2.605 | 0.267 | 0.268 | -0.124 | 0.925 | first_difference | 1min | 0.005 | 5000 | 7500 | 83.282 |
| test | 30s | 2.101 | 2.638 | 0.270 | 0.270 | -0.076 | 0.923 | first_difference | 1min | 0.010 | 5000 | 7500 | 76.785 |
| test | 30s | 0.879 | 1.109 | 0.113 | 0.113 | -0.011 | 0.986 | first_difference | none | 0.0001 | 5000 | 7500 | 86.460 |
| test | 30s | 0.903 | 1.139 | 0.116 | 0.116 | -0.005 | 0.986 | first_difference | none | 0.001 | 5000 | 7500 | 81.797 |
| test | 30s | 0.907 | 1.142 | 0.117 | 0.117 | -0.035 | 0.986 | first_difference | none | 0.005 | 5000 | 7500 | 75.409 |
| test | 30s | 0.899 | 1.134 | 0.115 | 0.115 | -0.006 | 0.986 | first_difference | none | 0.010 | 5000 | 7500 | 85.392 |
| test | 30s | 2.001 | 2.502 | 0.257 | 0.257 | -0.091 | 0.931 | second_difference | 1min | 0.0001 | 5000 | 7500 | 83.177 |
| test | 30s | 2.217 | 2.782 | 0.285 | 0.285 | 0.091 | 0.914 | second_difference | 1min | 0.001 | 5000 | 7500 | 83.997 |
| test | 30s | 2.231 | 2.825 | 0.287 | 0.287 | -0.032 | 0.911 | second_difference | 1min | 0.005 | 5000 | 7500 | 71.551 |
| test | 30s | 2.070 | 2.612 | 0.266 | 0.266 | 0.270 | 0.924 | second_difference | 1min | 0.010 | 5000 | 7500 | 91.232 |
| test | 30s | 0.913 | 1.159 | 0.117 | 0.117 | -0.008 | 0.985 | second_difference | none | 0.0001 | 5000 | 7500 | 78.111 |
| test | 30s | 0.932 | 1.177 | 0.120 | 0.120 | -0.017 | 0.985 | second_difference | none | 0.001 | 5000 | 7500 | 83.086 |
| test | 30s | 0.938 | 1.191 | 0.121 | 0.121 | 0.048 | 0.984 | second_difference | none | 0.005 | 5000 | 7500 | 86.057 |
| test | 30s | 0.939 | 1.194 | 0.121 | 0.121 | 0.012 | 0.984 | second_difference | none | 0.010 | 5000 | 7500 | 71.388 |
| test | raw | 0.218 | 0.274 | 0.028 | 0.028 | 0.003 | 0.999 | level | 30s | 0.0001 | 5000 | 7500 | 262.704 |
| test | raw | 0.234 | 0.293 | 0.030 | 0.030 | 0.046 | 0.999 | level | 30s | 0.001 | 5000 | 7500 | 229.554 |
| test | raw | 0.338 | 0.473 | 0.043 | 0.043 | -0.230 | 0.998 | level | 30s | 0.005 | 5000 | 7500 | 231.600 |
| test | raw | 0.265 | 0.337 | 0.034 | 0.034 | -0.104 | 0.999 | level | 30s | 0.010 | 5000 | 7500 | 181.410 |
| test | raw | 0.175 | 0.221 | 0.023 | 0.023 | -0.012 | 0.999 | level | none | 0.0001 | 5000 | 7500 | 84.478 |
| test | raw | 0.186 | 0.237 | 0.024 | 0.024 | -0.059 | 0.999 | level | none | 0.001 | 5000 | 7500 | 170.038 |
| test | raw | 0.170 | 0.214 | 0.022 | 0.022 | 0.012 | 0.999 | level | none | 0.005 | 5000 | 7500 | 248.375 |
| test | raw | 0.175 | 0.222 | 0.023 | 0.023 | -0.047 | 0.999 | level | none | 0.010 | 5000 | 7500 | 188.748 |
| test | raw | 0.191 | 0.238 | 0.025 | 0.025 | 0.002 | 0.999 | first_difference | 30s | 0.0001 | 5000 | 7500 | 233.149 |
| test | raw | 0.247 | 0.307 | 0.032 | 0.032 | 0.004 | 0.999 | first_difference | 30s | 0.001 | 5000 | 7500 | 26432.620 |
| test | raw | 0.235 | 0.290 | 0.030 | 0.030 | -0.002 | 0.999 | first_difference | 30s | 0.005 | 5000 | 7500 | 86.342 |
| test | raw | 0.232 | 0.289 | 0.030 | 0.030 | -0.034 | 0.999 | first_difference | 30s | 0.010 | 5000 | 7500 | 89.500 |
| test | raw | 0.104 | 0.132 | 0.013 | 0.013 | -0.007 | 1.000 | first_difference | none | 0.0001 | 5000 | 7500 | 234.739 |
| test | raw | 0.110 | 0.138 | 0.014 | 0.014 | -0.004 | 1.000 | first_difference | none | 0.001 | 5000 | 7500 | 218.520 |
| test | raw | 0.114 | 0.143 | 0.015 | 0.015 | -0.021 | 1.000 | first_difference | none | 0.005 | 5000 | 7500 | 166.934 |
| test | raw | 0.115 | 0.145 | 0.015 | 0.015 | -0.020 | 1.000 | first_difference | none | 0.010 | 5000 | 7500 | 209.383 |
| test | raw | 0.265 | 0.328 | 0.034 | 0.034 | 0.000201 | 0.999 | second_difference | 30s | 0.0001 | 5000 | 7500 | 87.608 |
| test | raw | 0.350 | 0.433 | 0.045 | 0.045 | 0.011 | 0.998 | second_difference | 30s | 0.001 | 5000 | 7500 | 102.350 |
| test | raw | 0.336 | 0.415 | 0.043 | 0.043 | -0.039 | 0.998 | second_difference | 30s | 0.005 | 5000 | 7500 | 124.183 |
| test | raw | 0.346 | 0.455 | 0.045 | 0.045 | -0.153 | 0.998 | second_difference | 30s | 0.010 | 5000 | 7500 | 118.056 |
| test | raw | 0.106 | 0.133 | 0.014 | 0.014 | -0.001 | 1.000 | second_difference | none | 0.0001 | 5000 | 7500 | 93.252 |
| test | raw | 0.110 | 0.139 | 0.014 | 0.014 | -0.004 | 1.000 | second_difference | none | 0.001 | 5000 | 7500 | 129.499 |
| test | raw | 0.114 | 0.143 | 0.015 | 0.015 | -0.008 | 1.000 | second_difference | none | 0.005 | 5000 | 7500 | 89.439 |
| test | raw | 0.112 | 0.142 | 0.014 | 0.014 | -0.012 | 1.000 | second_difference | none | 0.010 | 5000 | 7500 | 86.202 |

## Test Comparison Metrics
| Split | Granularity | Candidate Label | Selection Mode | Difference Order | Training Smoothing Window | Learning Rate Init | Min Steps | Max Steps | Training Seconds | Mae | Rmse | R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1min_d0_smooth_2min | all_candidates_1step_review | 0 | 2min | 0.0001 | 5000 | 7500 | 274.929 | 1.573 | 1.973 | 0.957 |
| test | 1min | 1min_d0_smooth_2min | all_candidates_1step_review | 0 | 2min | 0.001 | 5000 | 7500 | 66.153 | 2.237 | 2.839 | 0.910 |
| test | 1min | 1min_d0_smooth_2min | all_candidates_1step_review | 0 | 2min | 0.005 | 5000 | 7500 | 65.591 | 2.533 | 3.197 | 0.886 |
| test | 1min | 1min_d0_smooth_2min | all_candidates_1step_review | 0 | 2min | 0.010 | 5000 | 7500 | 66.101 | 2.548 | 3.202 | 0.886 |
| test | 1min | 1min_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.0001 | 5000 | 7500 | 78.413 | 1.540 | 1.930 | 0.958 |
| test | 1min | 1min_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.001 | 5000 | 7500 | 77.310 | 1.519 | 1.915 | 0.959 |
| test | 1min | 1min_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.005 | 5000 | 7500 | 84.032 | 1.517 | 1.908 | 0.959 |
| test | 1min | 1min_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.010 | 5000 | 7500 | 72.622 | 1.516 | 1.907 | 0.959 |
| test | 1min | 1min_d1_smooth_2min | all_candidates_1step_review | 1 | 2min | 0.0001 | 5000 | 7500 | 66.007 | 2.635 | 3.282 | 0.880 |
| test | 1min | 1min_d1_smooth_2min | all_candidates_1step_review | 1 | 2min | 0.001 | 5000 | 7500 | 65.626 | 2.691 | 3.352 | 0.875 |
| test | 1min | 1min_d1_smooth_2min | all_candidates_1step_review | 1 | 2min | 0.005 | 5000 | 7500 | 65.953 | 2.669 | 3.338 | 0.876 |
| test | 1min | 1min_d1_smooth_2min | all_candidates_1step_review | 1 | 2min | 0.010 | 5000 | 7500 | 65.938 | 2.701 | 3.355 | 0.875 |
| test | 1min | 1min_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.0001 | 5000 | 7500 | 67.922 | 1.512 | 1.898 | 0.960 |
| test | 1min | 1min_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.001 | 5000 | 7500 | 66.074 | 1.515 | 1.898 | 0.960 |
| test | 1min | 1min_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.005 | 5000 | 7500 | 66.372 | 1.524 | 1.913 | 0.959 |
| test | 1min | 1min_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.010 | 5000 | 7500 | 65.838 | 1.510 | 1.896 | 0.960 |
| test | 1min | 1min_d2_smooth_2min | all_candidates_1step_review | 2 | 2min | 0.0001 | 5000 | 7500 | 65.438 | 2.896 | 3.623 | 0.854 |
| test | 1min | 1min_d2_smooth_2min | all_candidates_1step_review | 2 | 2min | 0.001 | 5000 | 7500 | 65.584 | 3.150 | 4.000 | 0.822 |
| test | 1min | 1min_d2_smooth_2min | all_candidates_1step_review | 2 | 2min | 0.005 | 5000 | 7500 | 65.701 | 3.117 | 3.990 | 0.823 |
| test | 1min | 1min_d2_smooth_2min | all_candidates_1step_review | 2 | 2min | 0.010 | 5000 | 7500 | 65.780 | 2.695 | 3.367 | 0.874 |
| test | 1min | 1min_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.0001 | 5000 | 7500 | 65.239 | 1.613 | 2.029 | 0.954 |
| test | 1min | 1min_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.001 | 5000 | 7500 | 65.034 | 1.617 | 2.047 | 0.953 |
| test | 1min | 1min_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.005 | 5000 | 7500 | 67.545 | 1.635 | 2.043 | 0.953 |
| test | 1min | 1min_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.010 | 5000 | 7500 | 65.722 | 1.627 | 2.054 | 0.953 |
| test | 30s | 30s_d0_smooth_1min | all_candidates_1step_review | 0 | 1min | 0.0001 | 5000 | 7500 | 69.857 | 0.917 | 1.156 | 0.985 |
| test | 30s | 30s_d0_smooth_1min | all_candidates_1step_review | 0 | 1min | 0.001 | 5000 | 7500 | 86.470 | 1.075 | 1.365 | 0.979 |
| test | 30s | 30s_d0_smooth_1min | all_candidates_1step_review | 0 | 1min | 0.005 | 5000 | 7500 | 84.227 | 1.143 | 1.456 | 0.976 |
| test | 30s | 30s_d0_smooth_1min | all_candidates_1step_review | 0 | 1min | 0.010 | 5000 | 7500 | 70.614 | 1.123 | 1.428 | 0.977 |
| test | 30s | 30s_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.0001 | 5000 | 7500 | 79.935 | 0.908 | 1.145 | 0.985 |
| test | 30s | 30s_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.001 | 5000 | 7500 | 79.948 | 0.885 | 1.114 | 0.986 |
| test | 30s | 30s_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.005 | 5000 | 7500 | 75.104 | 0.884 | 1.120 | 0.986 |
| test | 30s | 30s_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.010 | 5000 | 7500 | 86.206 | 0.884 | 1.111 | 0.986 |
| test | 30s | 30s_d1_smooth_1min | all_candidates_1step_review | 1 | 1min | 0.0001 | 5000 | 7500 | 82.244 | 1.532 | 1.936 | 0.958 |
| test | 30s | 30s_d1_smooth_1min | all_candidates_1step_review | 1 | 1min | 0.001 | 5000 | 7500 | 80.603 | 2.173 | 2.727 | 0.917 |
| test | 30s | 30s_d1_smooth_1min | all_candidates_1step_review | 1 | 1min | 0.005 | 5000 | 7500 | 83.282 | 2.081 | 2.605 | 0.925 |
| test | 30s | 30s_d1_smooth_1min | all_candidates_1step_review | 1 | 1min | 0.010 | 5000 | 7500 | 76.785 | 2.101 | 2.638 | 0.923 |
| test | 30s | 30s_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.0001 | 5000 | 7500 | 86.460 | 0.879 | 1.109 | 0.986 |
| test | 30s | 30s_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.001 | 5000 | 7500 | 81.797 | 0.903 | 1.139 | 0.986 |
| test | 30s | 30s_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.005 | 5000 | 7500 | 75.409 | 0.907 | 1.142 | 0.986 |
| test | 30s | 30s_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.010 | 5000 | 7500 | 85.392 | 0.899 | 1.134 | 0.986 |
| test | 30s | 30s_d2_smooth_1min | all_candidates_1step_review | 2 | 1min | 0.0001 | 5000 | 7500 | 83.177 | 2.001 | 2.502 | 0.931 |
| test | 30s | 30s_d2_smooth_1min | all_candidates_1step_review | 2 | 1min | 0.001 | 5000 | 7500 | 83.997 | 2.217 | 2.782 | 0.914 |
| test | 30s | 30s_d2_smooth_1min | all_candidates_1step_review | 2 | 1min | 0.005 | 5000 | 7500 | 71.551 | 2.231 | 2.825 | 0.911 |
| test | 30s | 30s_d2_smooth_1min | all_candidates_1step_review | 2 | 1min | 0.010 | 5000 | 7500 | 91.232 | 2.070 | 2.612 | 0.924 |
| test | 30s | 30s_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.0001 | 5000 | 7500 | 78.111 | 0.913 | 1.159 | 0.985 |
| test | 30s | 30s_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.001 | 5000 | 7500 | 83.086 | 0.932 | 1.177 | 0.985 |
| test | 30s | 30s_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.005 | 5000 | 7500 | 86.057 | 0.938 | 1.191 | 0.984 |
| test | 30s | 30s_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.010 | 5000 | 7500 | 71.388 | 0.939 | 1.194 | 0.984 |
| test | raw | raw_d0_smooth_30s | all_candidates_1step_review | 0 | 30s | 0.0001 | 5000 | 7500 | 262.704 | 0.218 | 0.274 | 0.999 |
| test | raw | raw_d0_smooth_30s | all_candidates_1step_review | 0 | 30s | 0.001 | 5000 | 7500 | 229.554 | 0.234 | 0.293 | 0.999 |
| test | raw | raw_d0_smooth_30s | all_candidates_1step_review | 0 | 30s | 0.005 | 5000 | 7500 | 231.600 | 0.338 | 0.473 | 0.998 |
| test | raw | raw_d0_smooth_30s | all_candidates_1step_review | 0 | 30s | 0.010 | 5000 | 7500 | 181.410 | 0.265 | 0.337 | 0.999 |
| test | raw | raw_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.0001 | 5000 | 7500 | 84.478 | 0.175 | 0.221 | 0.999 |
| test | raw | raw_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.001 | 5000 | 7500 | 170.038 | 0.186 | 0.237 | 0.999 |
| test | raw | raw_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.005 | 5000 | 7500 | 248.375 | 0.170 | 0.214 | 0.999 |
| test | raw | raw_d0_smooth_none | all_candidates_1step_review | 0 | none | 0.010 | 5000 | 7500 | 188.748 | 0.175 | 0.222 | 0.999 |
| test | raw | raw_d1_smooth_30s | all_candidates_1step_review | 1 | 30s | 0.0001 | 5000 | 7500 | 233.149 | 0.191 | 0.238 | 0.999 |
| test | raw | raw_d1_smooth_30s | all_candidates_1step_review | 1 | 30s | 0.001 | 5000 | 7500 | 26432.620 | 0.247 | 0.307 | 0.999 |
| test | raw | raw_d1_smooth_30s | all_candidates_1step_review | 1 | 30s | 0.005 | 5000 | 7500 | 86.342 | 0.235 | 0.290 | 0.999 |
| test | raw | raw_d1_smooth_30s | all_candidates_1step_review | 1 | 30s | 0.010 | 5000 | 7500 | 89.500 | 0.232 | 0.289 | 0.999 |
| test | raw | raw_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.0001 | 5000 | 7500 | 234.739 | 0.104 | 0.132 | 1.000 |
| test | raw | raw_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.001 | 5000 | 7500 | 218.520 | 0.110 | 0.138 | 1.000 |
| test | raw | raw_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.005 | 5000 | 7500 | 166.934 | 0.114 | 0.143 | 1.000 |
| test | raw | raw_d1_smooth_none | all_candidates_1step_review | 1 | none | 0.010 | 5000 | 7500 | 209.383 | 0.115 | 0.145 | 1.000 |
| test | raw | raw_d2_smooth_30s | all_candidates_1step_review | 2 | 30s | 0.0001 | 5000 | 7500 | 87.608 | 0.265 | 0.328 | 0.999 |
| test | raw | raw_d2_smooth_30s | all_candidates_1step_review | 2 | 30s | 0.001 | 5000 | 7500 | 102.350 | 0.350 | 0.433 | 0.998 |
| test | raw | raw_d2_smooth_30s | all_candidates_1step_review | 2 | 30s | 0.005 | 5000 | 7500 | 124.183 | 0.336 | 0.415 | 0.998 |
| test | raw | raw_d2_smooth_30s | all_candidates_1step_review | 2 | 30s | 0.010 | 5000 | 7500 | 118.056 | 0.346 | 0.455 | 0.998 |
| test | raw | raw_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.0001 | 5000 | 7500 | 93.252 | 0.106 | 0.133 | 1.000 |
| test | raw | raw_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.001 | 5000 | 7500 | 129.499 | 0.110 | 0.139 | 1.000 |
| test | raw | raw_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.005 | 5000 | 7500 | 89.439 | 0.114 | 0.143 | 1.000 |
| test | raw | raw_d2_smooth_none | all_candidates_1step_review | 2 | none | 0.010 | 5000 | 7500 | 86.202 | 0.112 | 0.142 | 1.000 |

## Prepared Windows
| Granularity | Candidate Label | Transform Name | Training Smoothing Window | Train Windows | Test Windows |
|---|---|---|---|---|---|
| raw | raw_d0_smooth_none | level | none | 69000 | 17160 |
| raw | raw_d0_smooth_30s | level | 30s | 68995 | 17160 |
| raw | raw_d1_smooth_none | first_difference | none | 68999 | 17159 |
| raw | raw_d1_smooth_30s | first_difference | 30s | 68994 | 17159 |
| raw | raw_d2_smooth_none | second_difference | none | 68998 | 17158 |
| raw | raw_d2_smooth_30s | second_difference | 30s | 68993 | 17158 |
| 30s | 30s_d0_smooth_none | level | none | 11500 | 2860 |
| 30s | 30s_d0_smooth_1min | level | 1min | 11499 | 2860 |
| 30s | 30s_d1_smooth_none | first_difference | none | 11499 | 2859 |
| 30s | 30s_d1_smooth_1min | first_difference | 1min | 11498 | 2859 |
| 30s | 30s_d2_smooth_none | second_difference | none | 11498 | 2858 |
| 30s | 30s_d2_smooth_1min | second_difference | 1min | 11497 | 2858 |
| 1min | 1min_d0_smooth_none | level | none | 5750 | 1430 |
| 1min | 1min_d0_smooth_2min | level | 2min | 5749 | 1430 |
| 1min | 1min_d1_smooth_none | first_difference | none | 5749 | 1429 |
| 1min | 1min_d1_smooth_2min | first_difference | 2min | 5748 | 1429 |
| 1min | 1min_d2_smooth_none | second_difference | none | 5748 | 1428 |
| 1min | 1min_d2_smooth_2min | second_difference | 2min | 5747 | 1428 |