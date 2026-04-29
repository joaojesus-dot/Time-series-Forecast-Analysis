# Forecasting Evaluation

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Modeled granularities: `raw, 30s, 1min`.
- Forecast horizon: `1step`.
- Lookback window: `10min`.
- A `1step` horizon is evaluated across the whole test split; it is one row ahead per timestamp, not a single forecast point.
- Because row duration differs by granularity, raw, 30s, and 1min one-step metrics should not be treated as the same real-time horizon.

## Implemented Models
| Model Label | Status | Granularity Count | Report Path |
|---|---|---|---|
| Univariate MLP Target Lags | evaluated | 3 | models/univariate_mlp_target_lags.md |

## Reported Results By Split And Granularity
| Split | Granularity | Model Label | Mae | Rmse | R2 | Configuration |
|---|---|---|---|---|---|---|
| test | 1min | Univariate MLP Target Lags | 1.508 | 1.891 | 0.960 | first_difference / smooth=none / lr=0.0001 |
| test | 30s | Univariate MLP Target Lags | 0.876 | 1.107 | 0.986 | first_difference / smooth=none / lr=0.0001 |
| test | raw | Univariate MLP Target Lags | 0.104 | 0.132 | 1.000 | first_difference / smooth=none / lr=0.0001 |