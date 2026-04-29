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
| windows_batch_size | 1024 |
| dataloader_num_workers | 0 |
| dataloader_pin_memory | True |
| accelerator | gpu |
| devices | 1 |
| random_state | 42 |
| selection_mode | test_comparison |
| selected_candidate_combinations | [{"granularity": "raw", "difference_order": 1, "transform_name": "first_difference", "training_smoothing_window": "none", "learning_rate_init": 0.0001, "role": "best_short_horizon_baseline"}, {"granularity": "raw", "difference_order": 2, "transform_name": "second_difference", "training_smoothing_window": "none", "learning_rate_init": 0.0001, "role": "raw_stationary_alternative"}, {"granularity": "30s", "difference_order": 1, "transform_name": "first_difference", "training_smoothing_window": "none", "learning_rate_init": 0.0001, "role": "best_mid_granularity_candidate"}, {"granularity": "30s", "difference_order": 0, "transform_name": "level", "training_smoothing_window": "none", "learning_rate_init": 0.001, "role": "30s_level_alternative"}, {"granularity": "1min", "difference_order": 1, "transform_name": "first_difference", "training_smoothing_window": "none", "learning_rate_init": 0.0001, "role": "best_coarse_granularity_candidate"}, {"granularity": "1min", "difference_order": 0, "transform_name": "level", "training_smoothing_window": "none", "learning_rate_init": 0.01, "role": "1min_level_alternative"}] |
| candidate_limit | None |
| learning_rate_limit | None |
| write_window_data | False |

## Metrics
| Split | Granularity | Mae | Rmse | Mape | Smape | Bias | R2 | Transform Name | Training Smoothing Window | Learning Rate Init | Min Steps | Max Steps |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1.588 | 2.001 | 0.204 | 0.204 | -0.061 | 0.955 | level | 2min | 0.0001 | 5000 | 7500 |
| test | 1min | 1.981 | 2.522 | 0.255 | 0.255 | -0.092 | 0.929 | level | 2min | 0.001 | 5000 | 7500 |
| test | 1min | 2.359 | 2.999 | 0.303 | 0.303 | -0.173 | 0.900 | level | 2min | 0.005 | 5000 | 7500 |
| test | 1min | 2.413 | 3.041 | 0.310 | 0.310 | -0.054 | 0.897 | level | 2min | 0.010 | 5000 | 7500 |
| test | 1min | 1.606 | 2.012 | 0.206 | 0.206 | -0.057 | 0.955 | level | none | 0.0001 | 5000 | 7500 |
| test | 1min | 1.519 | 1.909 | 0.195 | 0.195 | -0.131 | 0.959 | level | none | 0.001 | 5000 | 7500 |
| test | 1min | 1.527 | 1.919 | 0.196 | 0.196 | -0.246 | 0.959 | level | none | 0.005 | 5000 | 7500 |
| test | 1min | 1.512 | 1.903 | 0.194 | 0.194 | -0.117 | 0.960 | level | none | 0.010 | 5000 | 7500 |
| test | 1min | 2.209 | 2.777 | 0.284 | 0.284 | -0.175 | 0.914 | first_difference | 2min | 0.0001 | 5000 | 7500 |
| test | 1min | 2.731 | 3.394 | 0.351 | 0.351 | -0.244 | 0.872 | first_difference | 2min | 0.001 | 5000 | 7500 |
| test | 1min | 2.693 | 3.350 | 0.346 | 0.347 | -0.299 | 0.875 | first_difference | 2min | 0.005 | 5000 | 7500 |
| test | 1min | 2.650 | 3.286 | 0.341 | 0.341 | -0.177 | 0.880 | first_difference | 2min | 0.010 | 5000 | 7500 |
| test | 1min | 1.508 | 1.891 | 0.194 | 0.194 | 0.008 | 0.960 | first_difference | none | 0.0001 | 5000 | 7500 |
| test | 1min | 1.518 | 1.902 | 0.195 | 0.195 | 0.032 | 0.960 | first_difference | none | 0.001 | 5000 | 7500 |
| test | 1min | 1.513 | 1.899 | 0.194 | 0.194 | 0.047 | 0.960 | first_difference | none | 0.005 | 5000 | 7500 |
| test | 1min | 1.515 | 1.901 | 0.195 | 0.195 | -0.058 | 0.960 | first_difference | none | 0.010 | 5000 | 7500 |
| test | 1min | 2.761 | 3.463 | 0.355 | 0.355 | -0.253 | 0.866 | second_difference | 2min | 0.0001 | 5000 | 7500 |
| test | 1min | 3.220 | 4.117 | 0.414 | 0.414 | -0.277 | 0.811 | second_difference | 2min | 0.001 | 5000 | 7500 |
| test | 1min | 3.171 | 4.065 | 0.408 | 0.408 | -0.219 | 0.816 | second_difference | 2min | 0.005 | 5000 | 7500 |
| test | 1min | 2.738 | 3.402 | 0.352 | 0.352 | 0.190 | 0.871 | second_difference | 2min | 0.010 | 5000 | 7500 |
| test | 1min | 1.608 | 2.026 | 0.207 | 0.207 | 0.001 | 0.954 | second_difference | none | 0.0001 | 5000 | 7500 |
| test | 1min | 1.620 | 2.040 | 0.208 | 0.208 | -0.009 | 0.954 | second_difference | none | 0.001 | 5000 | 7500 |
| test | 1min | 1.617 | 2.040 | 0.208 | 0.208 | 0.064 | 0.954 | second_difference | none | 0.005 | 5000 | 7500 |
| test | 1min | 1.629 | 2.053 | 0.209 | 0.209 | 0.164 | 0.953 | second_difference | none | 0.010 | 5000 | 7500 |
| test | 30s | 0.935 | 1.181 | 0.120 | 0.120 | -0.034 | 0.985 | level | 1min | 0.0001 | 5000 | 7500 |
| test | 30s | 1.014 | 1.281 | 0.130 | 0.130 | -0.046 | 0.982 | level | 1min | 0.001 | 5000 | 7500 |
| test | 30s | 1.069 | 1.358 | 0.137 | 0.137 | 0.020 | 0.980 | level | 1min | 0.005 | 5000 | 7500 |
| test | 30s | 1.075 | 1.364 | 0.138 | 0.138 | -0.021 | 0.979 | level | 1min | 0.010 | 5000 | 7500 |
| test | 30s | 0.933 | 1.174 | 0.120 | 0.120 | -0.055 | 0.985 | level | none | 0.0001 | 5000 | 7500 |
| test | 30s | 0.878 | 1.109 | 0.113 | 0.113 | 0.004 | 0.986 | level | none | 0.001 | 5000 | 7500 |
| test | 30s | 0.925 | 1.167 | 0.119 | 0.119 | -0.231 | 0.985 | level | none | 0.005 | 5000 | 7500 |
| test | 30s | 0.886 | 1.118 | 0.114 | 0.114 | 0.069 | 0.986 | level | none | 0.010 | 5000 | 7500 |
| test | 30s | 1.292 | 1.633 | 0.166 | 0.166 | -0.037 | 0.970 | first_difference | 1min | 0.0001 | 5000 | 7500 |
| test | 30s | 2.081 | 2.606 | 0.267 | 0.267 | -0.102 | 0.925 | first_difference | 1min | 0.001 | 5000 | 7500 |
| test | 30s | 2.051 | 2.568 | 0.264 | 0.264 | -0.172 | 0.927 | first_difference | 1min | 0.005 | 5000 | 7500 |
| test | 30s | 2.022 | 2.521 | 0.260 | 0.260 | -0.219 | 0.929 | first_difference | 1min | 0.010 | 5000 | 7500 |
| test | 30s | 0.876 | 1.107 | 0.113 | 0.113 | -0.014 | 0.986 | first_difference | none | 0.0001 | 5000 | 7500 |
| test | 30s | 0.890 | 1.122 | 0.114 | 0.114 | -0.011 | 0.986 | first_difference | none | 0.001 | 5000 | 7500 |
| test | 30s | 0.906 | 1.143 | 0.116 | 0.116 | 0.097 | 0.985 | first_difference | none | 0.005 | 5000 | 7500 |
| test | 30s | 0.898 | 1.131 | 0.115 | 0.115 | 0.119 | 0.986 | first_difference | none | 0.010 | 5000 | 7500 |
| test | 30s | 1.713 | 2.156 | 0.220 | 0.220 | -0.070 | 0.948 | second_difference | 1min | 0.0001 | 5000 | 7500 |
| test | 30s | 2.136 | 2.671 | 0.275 | 0.275 | -0.053 | 0.921 | second_difference | 1min | 0.001 | 5000 | 7500 |
| test | 30s | 2.084 | 2.609 | 0.268 | 0.268 | 0.040 | 0.924 | second_difference | 1min | 0.005 | 5000 | 7500 |
| test | 30s | 2.098 | 2.620 | 0.270 | 0.270 | 0.013 | 0.924 | second_difference | 1min | 0.010 | 5000 | 7500 |
| test | 30s | 0.911 | 1.154 | 0.117 | 0.117 | 0.0001883 | 0.985 | second_difference | none | 0.0001 | 5000 | 7500 |
| test | 30s | 0.920 | 1.164 | 0.118 | 0.118 | -0.051 | 0.985 | second_difference | none | 0.001 | 5000 | 7500 |
| test | 30s | 0.938 | 1.185 | 0.121 | 0.121 | -0.089 | 0.984 | second_difference | none | 0.005 | 5000 | 7500 |
| test | 30s | 0.927 | 1.191 | 0.119 | 0.119 | -0.014 | 0.984 | second_difference | none | 0.010 | 5000 | 7500 |
| test | raw | 0.229 | 0.289 | 0.029 | 0.029 | -0.022 | 0.999 | level | 30s | 0.0001 | 5000 | 7500 |
| test | raw | 0.312 | 0.398 | 0.040 | 0.040 | -0.156 | 0.998 | level | 30s | 0.001 | 5000 | 7500 |
| test | raw | 0.291 | 0.368 | 0.037 | 0.037 | -0.220 | 0.998 | level | 30s | 0.005 | 5000 | 7500 |
| test | raw | 0.330 | 0.421 | 0.042 | 0.042 | -0.204 | 0.998 | level | 30s | 0.010 | 5000 | 7500 |
| test | raw | 0.197 | 0.248 | 0.025 | 0.025 | -0.041 | 0.999 | level | none | 0.0001 | 5000 | 7500 |
| test | raw | 0.185 | 0.235 | 0.024 | 0.024 | -0.088 | 0.999 | level | none | 0.001 | 5000 | 7500 |
| test | raw | 0.172 | 0.216 | 0.022 | 0.022 | 0.026 | 0.999 | level | none | 0.005 | 5000 | 7500 |
| test | raw | 0.174 | 0.220 | 0.022 | 0.022 | -0.039 | 0.999 | level | none | 0.010 | 5000 | 7500 |
| test | raw | 0.171 | 0.213 | 0.022 | 0.022 | -0.002 | 0.999 | first_difference | 30s | 0.0001 | 5000 | 7500 |
| test | raw | 0.220 | 0.273 | 0.028 | 0.028 | 0.003 | 0.999 | first_difference | 30s | 0.001 | 5000 | 7500 |
| test | raw | 0.211 | 0.262 | 0.027 | 0.027 | -0.022 | 0.999 | first_difference | 30s | 0.005 | 5000 | 7500 |
| test | raw | 0.207 | 0.256 | 0.027 | 0.027 | -0.001 | 0.999 | first_difference | 30s | 0.010 | 5000 | 7500 |
| test | raw | 0.104 | 0.132 | 0.013 | 0.013 | -0.012 | 1.000 | first_difference | none | 0.0001 | 5000 | 7500 |
| test | raw | 0.107 | 0.135 | 0.014 | 0.014 | -0.013 | 1.000 | first_difference | none | 0.001 | 5000 | 7500 |
| test | raw | 0.110 | 0.138 | 0.014 | 0.014 | 0.008 | 1.000 | first_difference | none | 0.005 | 5000 | 7500 |
| test | raw | 0.111 | 0.139 | 0.014 | 0.014 | 0.004 | 1.000 | first_difference | none | 0.010 | 5000 | 7500 |
| test | raw | 0.229 | 0.285 | 0.029 | 0.029 | 0.003 | 0.999 | second_difference | 30s | 0.0001 | 5000 | 7500 |
| test | raw | 0.318 | 0.394 | 0.041 | 0.041 | 0.003 | 0.998 | second_difference | 30s | 0.001 | 5000 | 7500 |
| test | raw | 0.295 | 0.371 | 0.038 | 0.038 | -0.069 | 0.998 | second_difference | 30s | 0.005 | 5000 | 7500 |
| test | raw | 0.301 | 0.406 | 0.039 | 0.039 | -0.143 | 0.998 | second_difference | 30s | 0.010 | 5000 | 7500 |
| test | raw | 0.105 | 0.132 | 0.013 | 0.013 | 0.0003153 | 1.000 | second_difference | none | 0.0001 | 5000 | 7500 |
| test | raw | 0.110 | 0.139 | 0.014 | 0.014 | 0.008 | 1.000 | second_difference | none | 0.001 | 5000 | 7500 |
| test | raw | 0.111 | 0.140 | 0.014 | 0.014 | -0.002 | 1.000 | second_difference | none | 0.005 | 5000 | 7500 |
| test | raw | 0.113 | 0.142 | 0.015 | 0.015 | -0.009 | 1.000 | second_difference | none | 0.010 | 5000 | 7500 |

## Test Comparison Metrics
| Split | Granularity | Candidate Label | Selection Mode | Difference Order | Training Smoothing Window | Learning Rate Init | Min Steps | Max Steps | Mae | Rmse | R2 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1min_d0_smooth_2min | test_comparison | 0 | 2min | 0.0001 | 5000 | 7500 | 1.588 | 2.001 | 0.955 |
| test | 1min | 1min_d0_smooth_2min | test_comparison | 0 | 2min | 0.001 | 5000 | 7500 | 1.981 | 2.522 | 0.929 |
| test | 1min | 1min_d0_smooth_2min | test_comparison | 0 | 2min | 0.005 | 5000 | 7500 | 2.359 | 2.999 | 0.900 |
| test | 1min | 1min_d0_smooth_2min | test_comparison | 0 | 2min | 0.010 | 5000 | 7500 | 2.413 | 3.041 | 0.897 |
| test | 1min | 1min_d0_smooth_none | test_comparison | 0 | none | 0.0001 | 5000 | 7500 | 1.606 | 2.012 | 0.955 |
| test | 1min | 1min_d0_smooth_none | test_comparison | 0 | none | 0.001 | 5000 | 7500 | 1.519 | 1.909 | 0.959 |
| test | 1min | 1min_d0_smooth_none | test_comparison | 0 | none | 0.005 | 5000 | 7500 | 1.527 | 1.919 | 0.959 |
| test | 1min | 1min_d0_smooth_none | test_comparison | 0 | none | 0.010 | 5000 | 7500 | 1.512 | 1.903 | 0.960 |
| test | 1min | 1min_d1_smooth_2min | test_comparison | 1 | 2min | 0.0001 | 5000 | 7500 | 2.209 | 2.777 | 0.914 |
| test | 1min | 1min_d1_smooth_2min | test_comparison | 1 | 2min | 0.001 | 5000 | 7500 | 2.731 | 3.394 | 0.872 |
| test | 1min | 1min_d1_smooth_2min | test_comparison | 1 | 2min | 0.005 | 5000 | 7500 | 2.693 | 3.350 | 0.875 |
| test | 1min | 1min_d1_smooth_2min | test_comparison | 1 | 2min | 0.010 | 5000 | 7500 | 2.650 | 3.286 | 0.880 |
| test | 1min | 1min_d1_smooth_none | test_comparison | 1 | none | 0.0001 | 5000 | 7500 | 1.508 | 1.891 | 0.960 |
| test | 1min | 1min_d1_smooth_none | test_comparison | 1 | none | 0.001 | 5000 | 7500 | 1.518 | 1.902 | 0.960 |
| test | 1min | 1min_d1_smooth_none | test_comparison | 1 | none | 0.005 | 5000 | 7500 | 1.513 | 1.899 | 0.960 |
| test | 1min | 1min_d1_smooth_none | test_comparison | 1 | none | 0.010 | 5000 | 7500 | 1.515 | 1.901 | 0.960 |
| test | 1min | 1min_d2_smooth_2min | test_comparison | 2 | 2min | 0.0001 | 5000 | 7500 | 2.761 | 3.463 | 0.866 |
| test | 1min | 1min_d2_smooth_2min | test_comparison | 2 | 2min | 0.001 | 5000 | 7500 | 3.220 | 4.117 | 0.811 |
| test | 1min | 1min_d2_smooth_2min | test_comparison | 2 | 2min | 0.005 | 5000 | 7500 | 3.171 | 4.065 | 0.816 |
| test | 1min | 1min_d2_smooth_2min | test_comparison | 2 | 2min | 0.010 | 5000 | 7500 | 2.738 | 3.402 | 0.871 |
| test | 1min | 1min_d2_smooth_none | test_comparison | 2 | none | 0.0001 | 5000 | 7500 | 1.608 | 2.026 | 0.954 |
| test | 1min | 1min_d2_smooth_none | test_comparison | 2 | none | 0.001 | 5000 | 7500 | 1.620 | 2.040 | 0.954 |
| test | 1min | 1min_d2_smooth_none | test_comparison | 2 | none | 0.005 | 5000 | 7500 | 1.617 | 2.040 | 0.954 |
| test | 1min | 1min_d2_smooth_none | test_comparison | 2 | none | 0.010 | 5000 | 7500 | 1.629 | 2.053 | 0.953 |
| test | 30s | 30s_d0_smooth_1min | test_comparison | 0 | 1min | 0.0001 | 5000 | 7500 | 0.935 | 1.181 | 0.985 |
| test | 30s | 30s_d0_smooth_1min | test_comparison | 0 | 1min | 0.001 | 5000 | 7500 | 1.014 | 1.281 | 0.982 |
| test | 30s | 30s_d0_smooth_1min | test_comparison | 0 | 1min | 0.005 | 5000 | 7500 | 1.069 | 1.358 | 0.980 |
| test | 30s | 30s_d0_smooth_1min | test_comparison | 0 | 1min | 0.010 | 5000 | 7500 | 1.075 | 1.364 | 0.979 |
| test | 30s | 30s_d0_smooth_none | test_comparison | 0 | none | 0.0001 | 5000 | 7500 | 0.933 | 1.174 | 0.985 |
| test | 30s | 30s_d0_smooth_none | test_comparison | 0 | none | 0.001 | 5000 | 7500 | 0.878 | 1.109 | 0.986 |
| test | 30s | 30s_d0_smooth_none | test_comparison | 0 | none | 0.005 | 5000 | 7500 | 0.925 | 1.167 | 0.985 |
| test | 30s | 30s_d0_smooth_none | test_comparison | 0 | none | 0.010 | 5000 | 7500 | 0.886 | 1.118 | 0.986 |
| test | 30s | 30s_d1_smooth_1min | test_comparison | 1 | 1min | 0.0001 | 5000 | 7500 | 1.292 | 1.633 | 0.970 |
| test | 30s | 30s_d1_smooth_1min | test_comparison | 1 | 1min | 0.001 | 5000 | 7500 | 2.081 | 2.606 | 0.925 |
| test | 30s | 30s_d1_smooth_1min | test_comparison | 1 | 1min | 0.005 | 5000 | 7500 | 2.051 | 2.568 | 0.927 |
| test | 30s | 30s_d1_smooth_1min | test_comparison | 1 | 1min | 0.010 | 5000 | 7500 | 2.022 | 2.521 | 0.929 |
| test | 30s | 30s_d1_smooth_none | test_comparison | 1 | none | 0.0001 | 5000 | 7500 | 0.876 | 1.107 | 0.986 |
| test | 30s | 30s_d1_smooth_none | test_comparison | 1 | none | 0.001 | 5000 | 7500 | 0.890 | 1.122 | 0.986 |
| test | 30s | 30s_d1_smooth_none | test_comparison | 1 | none | 0.005 | 5000 | 7500 | 0.906 | 1.143 | 0.985 |
| test | 30s | 30s_d1_smooth_none | test_comparison | 1 | none | 0.010 | 5000 | 7500 | 0.898 | 1.131 | 0.986 |
| test | 30s | 30s_d2_smooth_1min | test_comparison | 2 | 1min | 0.0001 | 5000 | 7500 | 1.713 | 2.156 | 0.948 |
| test | 30s | 30s_d2_smooth_1min | test_comparison | 2 | 1min | 0.001 | 5000 | 7500 | 2.136 | 2.671 | 0.921 |
| test | 30s | 30s_d2_smooth_1min | test_comparison | 2 | 1min | 0.005 | 5000 | 7500 | 2.084 | 2.609 | 0.924 |
| test | 30s | 30s_d2_smooth_1min | test_comparison | 2 | 1min | 0.010 | 5000 | 7500 | 2.098 | 2.620 | 0.924 |
| test | 30s | 30s_d2_smooth_none | test_comparison | 2 | none | 0.0001 | 5000 | 7500 | 0.911 | 1.154 | 0.985 |
| test | 30s | 30s_d2_smooth_none | test_comparison | 2 | none | 0.001 | 5000 | 7500 | 0.920 | 1.164 | 0.985 |
| test | 30s | 30s_d2_smooth_none | test_comparison | 2 | none | 0.005 | 5000 | 7500 | 0.938 | 1.185 | 0.984 |
| test | 30s | 30s_d2_smooth_none | test_comparison | 2 | none | 0.010 | 5000 | 7500 | 0.927 | 1.191 | 0.984 |
| test | raw | raw_d0_smooth_30s | test_comparison | 0 | 30s | 0.0001 | 5000 | 7500 | 0.229 | 0.289 | 0.999 |
| test | raw | raw_d0_smooth_30s | test_comparison | 0 | 30s | 0.001 | 5000 | 7500 | 0.312 | 0.398 | 0.998 |
| test | raw | raw_d0_smooth_30s | test_comparison | 0 | 30s | 0.005 | 5000 | 7500 | 0.291 | 0.368 | 0.998 |
| test | raw | raw_d0_smooth_30s | test_comparison | 0 | 30s | 0.010 | 5000 | 7500 | 0.330 | 0.421 | 0.998 |
| test | raw | raw_d0_smooth_none | test_comparison | 0 | none | 0.0001 | 5000 | 7500 | 0.197 | 0.248 | 0.999 |
| test | raw | raw_d0_smooth_none | test_comparison | 0 | none | 0.001 | 5000 | 7500 | 0.185 | 0.235 | 0.999 |
| test | raw | raw_d0_smooth_none | test_comparison | 0 | none | 0.005 | 5000 | 7500 | 0.172 | 0.216 | 0.999 |
| test | raw | raw_d0_smooth_none | test_comparison | 0 | none | 0.010 | 5000 | 7500 | 0.174 | 0.220 | 0.999 |
| test | raw | raw_d1_smooth_30s | test_comparison | 1 | 30s | 0.0001 | 5000 | 7500 | 0.171 | 0.213 | 0.999 |
| test | raw | raw_d1_smooth_30s | test_comparison | 1 | 30s | 0.001 | 5000 | 7500 | 0.220 | 0.273 | 0.999 |
| test | raw | raw_d1_smooth_30s | test_comparison | 1 | 30s | 0.005 | 5000 | 7500 | 0.211 | 0.262 | 0.999 |
| test | raw | raw_d1_smooth_30s | test_comparison | 1 | 30s | 0.010 | 5000 | 7500 | 0.207 | 0.256 | 0.999 |
| test | raw | raw_d1_smooth_none | test_comparison | 1 | none | 0.0001 | 5000 | 7500 | 0.104 | 0.132 | 1.000 |
| test | raw | raw_d1_smooth_none | test_comparison | 1 | none | 0.001 | 5000 | 7500 | 0.107 | 0.135 | 1.000 |
| test | raw | raw_d1_smooth_none | test_comparison | 1 | none | 0.005 | 5000 | 7500 | 0.110 | 0.138 | 1.000 |
| test | raw | raw_d1_smooth_none | test_comparison | 1 | none | 0.010 | 5000 | 7500 | 0.111 | 0.139 | 1.000 |
| test | raw | raw_d2_smooth_30s | test_comparison | 2 | 30s | 0.0001 | 5000 | 7500 | 0.229 | 0.285 | 0.999 |
| test | raw | raw_d2_smooth_30s | test_comparison | 2 | 30s | 0.001 | 5000 | 7500 | 0.318 | 0.394 | 0.998 |
| test | raw | raw_d2_smooth_30s | test_comparison | 2 | 30s | 0.005 | 5000 | 7500 | 0.295 | 0.371 | 0.998 |
| test | raw | raw_d2_smooth_30s | test_comparison | 2 | 30s | 0.010 | 5000 | 7500 | 0.301 | 0.406 | 0.998 |
| test | raw | raw_d2_smooth_none | test_comparison | 2 | none | 0.0001 | 5000 | 7500 | 0.105 | 0.132 | 1.000 |
| test | raw | raw_d2_smooth_none | test_comparison | 2 | none | 0.001 | 5000 | 7500 | 0.110 | 0.139 | 1.000 |
| test | raw | raw_d2_smooth_none | test_comparison | 2 | none | 0.005 | 5000 | 7500 | 0.111 | 0.140 | 1.000 |
| test | raw | raw_d2_smooth_none | test_comparison | 2 | none | 0.010 | 5000 | 7500 | 0.113 | 0.142 | 1.000 |

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