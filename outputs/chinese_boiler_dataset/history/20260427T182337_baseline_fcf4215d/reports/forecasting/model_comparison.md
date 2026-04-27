# Model Comparison

## Comparable Metrics
| Split | Granularity | Mae Rank | Model Label | Mae | Rmse | Mape | Smape | Bias | Configuration |
|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1.000 | Univariate MLP Target Lags | 2.137 | 2.693 | 0.275 | 0.275 | 0.027 | first_difference / smooth=1min / lr=0.01 |
| test | 1min | 2.000 | Univariate ARIMA | 8.287 | 10.446 | 1.059 | 1.065 | -4.518 | ARIMA(1,0,1) |
| test | 30s | 1.000 | Univariate MLP Target Lags | 1.231 | 1.554 | 0.158 | 0.158 | 0.003 | first_difference / smooth=30s / lr=0.01 |
| test | 30s | 2.000 | Univariate ARIMA | 8.280 | 10.435 | 1.058 | 1.064 | -4.501 | ARIMA(1,0,1) |
| validation | 1min | 1.000 | Univariate MLP Target Lags | 2.032 | 2.576 | 0.264 | 0.264 | -0.010 | first_difference / smooth=1min / lr=0.01 |
| validation | 30s | 1.000 | Univariate MLP Target Lags | 1.228 | 1.550 | 0.160 | 0.160 | -0.009 | first_difference / smooth=30s / lr=0.01 |