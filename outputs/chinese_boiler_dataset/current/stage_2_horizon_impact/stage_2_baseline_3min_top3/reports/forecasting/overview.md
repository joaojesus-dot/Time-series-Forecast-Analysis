# Forecasting Evaluation

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Modeled granularities: `raw, 30s, 1min`.
- Forecast horizon: `3min`.
- Lookback window: `3min`.
- A `1step` horizon is evaluated across the whole test split; it is one row ahead per timestamp, not a single forecast point.
- Because row duration differs by granularity, raw, 30s, and 1min one-step metrics should not be treated as the same real-time horizon.

## Implemented Models
| Model Label | Status | Granularity Count | Report Path |
|---|---|---|---|
| Univariate MLP Target Lags | evaluated | 3 | models/univariate_mlp_target_lags.md |

## Reported Results By Split And Granularity
| Split | Granularity | Model Label | Mae | Rmse | R2 | Configuration |
|---|---|---|---|---|---|---|
| test | 1min | Univariate MLP Target Lags | 0.303 | 0.384 | 0.810 | first_difference / smooth=none / lr=0.0001 |
| test | 30s | Univariate MLP Target Lags | 0.311 | 0.393 | 0.802 | level / smooth=none / lr=0.0001 |
| test | raw | Univariate MLP Target Lags | 0.310 | 0.392 | 0.803 | first_difference / smooth=none / lr=0.0001 |