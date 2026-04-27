# Forecasting Overview

## Organizer View
- Target variable: `TE_8313B.AV_0#`.
- Protocol dataset prefix: `subset_B`.
- Forecast horizon: `10min`.
- Lookback window: `10min`.
- Active granularities: `raw, 30s, 1min, 5min`.
- This directory is structured for top-down reading: experiment plan, model cards, then cross-model comparison.
- Every model should be understandable from its own report without losing comparability with the other models.

## Active Model Catalog
| Analysis | Model Label | Status | Granularity Count | Supported Splits | Config Summary | Report Path |
|---|---|---|---|---|---|---|
| univariate | Univariate ARIMA | evaluated | 4 | test | enabled=True, method=CSS-ML, order=[1,0,1], season_length=1, seasonal_order=[0,0,0] | models/univariate_arima.md |
| univariate | Univariate MLP Target Lags | evaluated | 4 | validation,test | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False | models/univariate_mlp_target_lags.md |

## Best Current Results By Granularity
| Split | Granularity | Model Label | Mae | Rmse | Config Summary |
|---|---|---|---|---|---|
| test | 1min | Univariate MLP Target Lags | 5.614 | 7.165 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| test | 30s | Univariate MLP Target Lags | 5.803 | 7.461 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| test | 5min | Univariate MLP Target Lags | 5.060 | 6.653 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| test | raw | Univariate MLP Target Lags | 6.837 | 8.803 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| validation | 1min | Univariate MLP Target Lags | 5.918 | 7.345 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| validation | 30s | Univariate MLP Target Lags | 6.483 | 8.069 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| validation | 5min | Univariate MLP Target Lags | 5.475 | 6.947 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| validation | raw | Univariate MLP Target Lags | 7.479 | 9.424 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |