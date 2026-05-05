# Model Comparison

## Comparable Metrics
| Split | Granularity | Mae Rank | Model Label | Mae | Rmse | Mape | Smape | Bias | R2 | Configuration |
|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1.000 | Univariate MLP Target Lags | 0.141 | 0.177 | 194.270 | 38.353 | 0.0004628 | 0.960 | first_difference / smooth=none / lr=0.0001 |
| test | 30s | 1.000 | Univariate MLP Target Lags | 0.100 | 0.126 | 74.357 | 31.048 | -0.0006596 | 0.980 | level / smooth=none / lr=0.0001 |
| test | raw | 1.000 | Univariate MLP Target Lags | 0.010 | 0.012 | 7.638 | 4.857 | -0.0001106 | 1.000 | first_difference / smooth=none / lr=0.0001 |