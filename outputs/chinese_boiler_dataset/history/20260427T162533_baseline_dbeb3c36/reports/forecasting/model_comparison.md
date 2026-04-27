# Model Comparison

## Purpose
- Compare model outputs under the shared forecasting protocol.
- Keep model family differences visible through configuration summaries instead of through inconsistent report formats.

## Comparable Metrics
| Split | Granularity | Mae Rank | Model Label | Mae | Rmse | Mape | Smape | Bias | Warning Count | Config Summary |
|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1.000 | Univariate MLP Target Lags | 5.614 | 7.165 | 0.721 | 0.722 | -1.393 | 0 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| test | 1min | 2.000 | Univariate ARIMA | 8.287 | 10.446 | 1.059 | 1.065 | -4.518 | 0 | enabled=True, method=CSS-ML, order=[1,0,1], season_length=1, seasonal_order=[0,0,0] |
| test | 30s | 1.000 | Univariate MLP Target Lags | 5.803 | 7.461 | 0.745 | 0.746 | -0.812 | 1 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| test | 30s | 2.000 | Univariate ARIMA | 8.280 | 10.435 | 1.058 | 1.064 | -4.501 | 0 | enabled=True, method=CSS-ML, order=[1,0,1], season_length=1, seasonal_order=[0,0,0] |
| test | 5min | 1.000 | Univariate MLP Target Lags | 5.060 | 6.653 | 0.649 | 0.650 | -1.059 | 0 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| test | 5min | 2.000 | Univariate ARIMA | 8.194 | 10.280 | 1.047 | 1.053 | -4.520 | 0 | enabled=True, method=CSS-ML, order=[1,0,1], season_length=1, seasonal_order=[0,0,0] |
| test | raw | 1.000 | Univariate MLP Target Lags | 6.837 | 8.803 | 0.879 | 0.879 | -0.408 | 0 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| test | raw | 2.000 | Univariate ARIMA | 7.910 | 9.757 | 1.013 | 1.017 | -3.254 | 0 | enabled=True, method=CSS-ML, order=[1,0,1], season_length=1, seasonal_order=[0,0,0] |
| validation | 1min | 1.000 | Univariate MLP Target Lags | 5.918 | 7.345 | 0.770 | 0.769 | 0.583 | 0 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| validation | 30s | 1.000 | Univariate MLP Target Lags | 6.483 | 8.069 | 0.844 | 0.843 | 1.133 | 1 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| validation | 5min | 1.000 | Univariate MLP Target Lags | 5.475 | 6.947 | 0.713 | 0.711 | 1.222 | 0 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |
| validation | raw | 1.000 | Univariate MLP Target Lags | 7.479 | 9.424 | 0.973 | 0.971 | 2.049 | 0 | activation=relu, enabled=True, hidden_layer_sizes=[64,32], max_iter=300, random_state=42, write_window_data=False |