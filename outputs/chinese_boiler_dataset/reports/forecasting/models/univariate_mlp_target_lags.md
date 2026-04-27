# Univariate MLP Target Lags

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Forecast horizon: `10min`.
- Lookback window: `10min`.

## Active Configuration
| Key | Value |
|---|---|
| enabled | True |
| hidden_units_strategy | match_lookback |
| activation | relu |
| solver | sgd |
| learning_rate | adaptive |
| initial_learning_rate_grid | [0.001, 0.005, 0.01] |
| max_iter | 5000 |
| random_state | 42 |
| selection_mode | validation_recommendation |
| write_window_data | False |

## Metrics
| Split | Granularity | Mae | Rmse | Mape | Smape | Bias | Selection Basis | Transform Name | Training Smoothing Window | Learning Rate Init |
|---|---|---|---|---|---|---|---|---|---|---|
| validation | 1min | 2.032 | 2.576 | 0.264 | 0.264 | -0.010 | validation | first_difference | 1min | 0.010 |
| test | 1min | 2.137 | 2.693 | 0.275 | 0.275 | 0.027 | validation | first_difference | 1min | 0.010 |
| validation | 30s | 1.228 | 1.550 | 0.160 | 0.160 | -0.009 | validation | first_difference | 30s | 0.010 |
| test | 30s | 1.231 | 1.554 | 0.158 | 0.158 | 0.003 | validation | first_difference | 30s | 0.010 |

## Preparation Candidate Ranking
| Granularity | Candidate Label | Selection Mode | Difference Order | Training Smoothing Window | Learning Rate Init | Validation Mae | Test Mae |
|---|---|---|---|---|---|---|---|
| 1min | 1min_d0_smooth_1min | validation_recommendation | 0 | 1min | 0.010 | 6.417 | 5.972 |
| 1min | 1min_d0_smooth_none | validation_recommendation | 0 | none | 0.010 | 6.417 | 5.972 |
| 1min | 1min_d1_smooth_1min | validation_recommendation | 1 | 1min | 0.010 | 2.032 | 2.137 |
| 1min | 1min_d1_smooth_none | validation_recommendation | 1 | none | 0.010 | 2.032 | 2.137 |
| 1min | 1min_d2_smooth_1min | validation_recommendation | 2 | 1min | 0.010 | 2.338 | 2.470 |
| 1min | 1min_d2_smooth_none | validation_recommendation | 2 | none | 0.010 | 2.338 | 2.470 |
| 30s | 30s_d0_smooth_1min | validation_recommendation | 0 | 1min | 0.010 | 5.859 | 5.505 |
| 30s | 30s_d0_smooth_30s | validation_recommendation | 0 | 30s | 0.010 | 5.868 | 5.508 |
| 30s | 30s_d0_smooth_none | validation_recommendation | 0 | none | 0.010 | 5.868 | 5.508 |
| 30s | 30s_d1_smooth_1min | validation_recommendation | 1 | 1min | 0.010 | 1.236 | 1.239 |
| 30s | 30s_d1_smooth_30s | validation_recommendation | 1 | 30s | 0.010 | 1.228 | 1.231 |
| 30s | 30s_d1_smooth_none | validation_recommendation | 1 | none | 0.010 | 1.228 | 1.231 |
| 30s | 30s_d2_smooth_1min | validation_recommendation | 2 | 1min | 0.010 | 1.487 | 1.463 |
| 30s | 30s_d2_smooth_30s | validation_recommendation | 2 | 30s | 0.010 | 1.308 | 1.296 |
| 30s | 30s_d2_smooth_none | validation_recommendation | 2 | none | 0.010 | 1.308 | 1.296 |

## Parameter Effects
| Granularity | Candidate Label | Hidden Units | Learning Rate Init | Validation Mae | Validation Rmse | Test Mae | Test Rmse |
|---|---|---|---|---|---|---|---|
| 30s | 30s_d0_smooth_none | 20 | 0.001 | 6.070 | 7.761 | 5.719 | 7.368 |
| 30s | 30s_d0_smooth_none | 20 | 0.005 | 5.946 | 7.621 | 5.574 | 7.215 |
| 30s | 30s_d0_smooth_none | 20 | 0.010 | 5.868 | 7.527 | 5.508 | 7.143 |
| 30s | 30s_d0_smooth_30s | 20 | 0.001 | 6.070 | 7.761 | 5.719 | 7.368 |
| 30s | 30s_d0_smooth_30s | 20 | 0.005 | 5.946 | 7.621 | 5.574 | 7.215 |
| 30s | 30s_d0_smooth_30s | 20 | 0.010 | 5.868 | 7.527 | 5.508 | 7.143 |
| 30s | 30s_d0_smooth_1min | 20 | 0.001 | 6.072 | 7.763 | 5.719 | 7.370 |
| 30s | 30s_d0_smooth_1min | 20 | 0.005 | 5.949 | 7.627 | 5.581 | 7.227 |
| 30s | 30s_d0_smooth_1min | 20 | 0.010 | 5.859 | 7.515 | 5.505 | 7.138 |
| 30s | 30s_d1_smooth_none | 20 | 0.001 | 1.570 | 1.955 | 1.559 | 1.958 |
| 30s | 30s_d1_smooth_none | 20 | 0.005 | 1.262 | 1.591 | 1.264 | 1.597 |
| 30s | 30s_d1_smooth_none | 20 | 0.010 | 1.228 | 1.550 | 1.231 | 1.554 |
| 30s | 30s_d1_smooth_30s | 20 | 0.001 | 1.570 | 1.955 | 1.559 | 1.958 |
| 30s | 30s_d1_smooth_30s | 20 | 0.005 | 1.262 | 1.591 | 1.264 | 1.597 |
| 30s | 30s_d1_smooth_30s | 20 | 0.010 | 1.228 | 1.550 | 1.231 | 1.554 |
| 30s | 30s_d1_smooth_1min | 20 | 0.001 | 1.580 | 1.971 | 1.571 | 1.975 |
| 30s | 30s_d1_smooth_1min | 20 | 0.005 | 1.280 | 1.615 | 1.282 | 1.620 |
| 30s | 30s_d1_smooth_1min | 20 | 0.010 | 1.236 | 1.559 | 1.239 | 1.565 |
| 30s | 30s_d2_smooth_none | 20 | 0.001 | 1.605 | 1.995 | 1.577 | 1.990 |
| 30s | 30s_d2_smooth_none | 20 | 0.005 | 1.343 | 1.668 | 1.329 | 1.674 |
| 30s | 30s_d2_smooth_none | 20 | 0.010 | 1.308 | 1.622 | 1.296 | 1.630 |
| 30s | 30s_d2_smooth_30s | 20 | 0.001 | 1.605 | 1.995 | 1.577 | 1.990 |
| 30s | 30s_d2_smooth_30s | 20 | 0.005 | 1.343 | 1.668 | 1.329 | 1.674 |
| 30s | 30s_d2_smooth_30s | 20 | 0.010 | 1.308 | 1.622 | 1.296 | 1.630 |
| 30s | 30s_d2_smooth_1min | 20 | 0.001 | 1.630 | 2.029 | 1.602 | 2.021 |
| 30s | 30s_d2_smooth_1min | 20 | 0.005 | 1.568 | 1.946 | 1.544 | 1.946 |
| 30s | 30s_d2_smooth_1min | 20 | 0.010 | 1.487 | 1.843 | 1.463 | 1.846 |
| 1min | 1min_d0_smooth_none | 10 | 0.001 | 7.130 | 9.108 | 6.964 | 9.010 |
| 1min | 1min_d0_smooth_none | 10 | 0.005 | 6.505 | 8.304 | 6.203 | 8.036 |
| 1min | 1min_d0_smooth_none | 10 | 0.010 | 6.417 | 8.188 | 5.972 | 7.741 |
| 1min | 1min_d0_smooth_1min | 10 | 0.001 | 7.130 | 9.108 | 6.964 | 9.010 |
| 1min | 1min_d0_smooth_1min | 10 | 0.005 | 6.505 | 8.304 | 6.203 | 8.036 |
| 1min | 1min_d0_smooth_1min | 10 | 0.010 | 6.417 | 8.188 | 5.972 | 7.741 |
| 1min | 1min_d1_smooth_none | 10 | 0.001 | 2.184 | 2.774 | 2.290 | 2.888 |
| 1min | 1min_d1_smooth_none | 10 | 0.005 | 2.098 | 2.660 | 2.203 | 2.778 |
| 1min | 1min_d1_smooth_none | 10 | 0.010 | 2.032 | 2.576 | 2.137 | 2.693 |
| 1min | 1min_d1_smooth_1min | 10 | 0.001 | 2.184 | 2.774 | 2.290 | 2.888 |
| 1min | 1min_d1_smooth_1min | 10 | 0.005 | 2.098 | 2.660 | 2.203 | 2.778 |
| 1min | 1min_d1_smooth_1min | 10 | 0.010 | 2.032 | 2.576 | 2.137 | 2.693 |
| 1min | 1min_d2_smooth_none | 10 | 0.001 | 2.546 | 3.154 | 2.675 | 3.373 |
| 1min | 1min_d2_smooth_none | 10 | 0.005 | 2.429 | 3.016 | 2.564 | 3.230 |
| 1min | 1min_d2_smooth_none | 10 | 0.010 | 2.338 | 2.908 | 2.470 | 3.109 |
| 1min | 1min_d2_smooth_1min | 10 | 0.001 | 2.546 | 3.154 | 2.675 | 3.373 |
| 1min | 1min_d2_smooth_1min | 10 | 0.005 | 2.429 | 3.016 | 2.564 | 3.230 |
| 1min | 1min_d2_smooth_1min | 10 | 0.010 | 2.338 | 2.908 | 2.470 | 3.109 |

## Prepared Windows
| Granularity | Candidate Label | Transform Name | Training Smoothing Window | Train Windows | Validation Windows | Test Windows |
|---|---|---|---|---|---|---|
| 30s | 30s_d0_smooth_none | level | none | 10041 | 1400 | 2842 |
| 30s | 30s_d0_smooth_30s | level | 30s | 10041 | 1400 | 2842 |
| 30s | 30s_d0_smooth_1min | level | 1min | 10040 | 1400 | 2842 |
| 30s | 30s_d1_smooth_none | first_difference | none | 10040 | 1399 | 2841 |
| 30s | 30s_d1_smooth_30s | first_difference | 30s | 10040 | 1399 | 2841 |
| 30s | 30s_d1_smooth_1min | first_difference | 1min | 10039 | 1399 | 2841 |
| 30s | 30s_d2_smooth_none | second_difference | none | 10039 | 1398 | 2840 |
| 30s | 30s_d2_smooth_30s | second_difference | 30s | 10039 | 1398 | 2840 |
| 30s | 30s_d2_smooth_1min | second_difference | 1min | 10038 | 1398 | 2840 |
| 1min | 1min_d0_smooth_none | level | none | 5021 | 700 | 1422 |
| 1min | 1min_d0_smooth_1min | level | 1min | 5021 | 700 | 1422 |
| 1min | 1min_d1_smooth_none | first_difference | none | 5020 | 699 | 1421 |
| 1min | 1min_d1_smooth_1min | first_difference | 1min | 5020 | 699 | 1421 |
| 1min | 1min_d2_smooth_none | second_difference | none | 5019 | 698 | 1420 |
| 1min | 1min_d2_smooth_1min | second_difference | 1min | 5019 | 698 | 1420 |