# Stage 3 Univariate MLP Comparison

## Question
Does the advanced two-hidden-layer MLP work better than the Stage 2 one-hidden-layer 3-minute baseline?

## Short Answer
Stage 3 MLP improves the baseline for at least one granularity, but the gains are small and not consistent.

## Best Stage 3 Result Per Granularity

| Granularity | Baseline R2 | Best Stage 3 R2 | R2 Delta | Baseline MAE | Best Stage 3 MAE | MAE Delta | Best Stage 3 Variant |
|---|---:|---:|---:|---:|---:|---:|---|
| 1min | 0.8099 | 0.8102 | +0.0002 | 0.3028 | 0.3027 | -0.0001 | neuralforecast_2_hidden_match_lookback |
| 30s | 0.8016 | 0.8113 | +0.0097 | 0.3110 | 0.3036 | -0.0073 | neuralforecast_2_hidden_match_lookback |
| raw | 0.8030 | 0.8023 | -0.0007 | 0.3095 | 0.3097 | +0.0002 | neuralforecast_2_hidden_match_lookback |

## Interpretation
- The comparison uses the same top preprocessing candidate per granularity and the same 3-minute forecast horizon.
- Stage 3 changes the MLP architecture, so the relevant comparison is against the Stage 2 3-minute baseline, not against one-step results.
- A useful improvement should increase R2 and reduce MAE/RMSE consistently. Small mixed changes should be treated as inconclusive.

## Supporting Artifacts
- `metrics/stage3_vs_baseline_r2.png`
- `metrics/stage3_vs_baseline_mae.png`
- `metrics/stage3_vs_baseline_rmse.png`
- `metrics/stage3_best_delta_vs_baseline.png`
- `error_distribution/stage3_best_vs_baseline_absolute_error_distribution.png`
- `forecast_windows/*_baseline_vs_stage3_forecast.png`
