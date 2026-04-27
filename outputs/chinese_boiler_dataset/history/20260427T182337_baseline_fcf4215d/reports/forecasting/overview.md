# Forecasting Evaluation

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Modeled granularities: `30s, 1min`.
- Forecast horizon: `10min`.
- Lookback window: `10min`.

## Implemented Models
| Model Label | Status | Granularity Count | Report Path |
|---|---|---|---|
| Univariate ARIMA | evaluated | 2 | models/univariate_arima.md |
| Univariate MLP Target Lags | evaluated | 2 | models/univariate_mlp_target_lags.md |

## Reported Results By Split And Granularity
| Split | Granularity | Model Label | Mae | Rmse | Configuration |
|---|---|---|---|---|---|
| test | 1min | Univariate MLP Target Lags | 2.137 | 2.693 | first_difference / smooth=1min / lr=0.01 |
| test | 30s | Univariate MLP Target Lags | 1.231 | 1.554 | first_difference / smooth=30s / lr=0.01 |
| validation | 1min | Univariate MLP Target Lags | 2.032 | 2.576 | first_difference / smooth=1min / lr=0.01 |
| validation | 30s | Univariate MLP Target Lags | 1.228 | 1.550 | first_difference / smooth=30s / lr=0.01 |