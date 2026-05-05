# Model Comparison

## Comparable Metrics
| Split | Granularity | Mae Rank | Model Label | Mae | Rmse | Mape | Smape | Bias | R2 | Configuration |
|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1.000 | Univariate MLP Target Lags | 0.303 | 0.384 | 293.529 | 62.922 | 0.003 | 0.810 | first_difference / smooth=none / lr=0.01 |
| test | 1min | 2.000 | Univariate MLP Target Lags | 0.306 | 0.388 | 282.191 | 62.638 | 0.004 | 0.806 | first_difference / smooth=none / lr=0.01 |
| test | 30s | 1.000 | Univariate MLP Target Lags | 0.304 | 0.383 | 153.591 | 65.568 | -0.033 | 0.811 | level / smooth=none / lr=0.01 |
| test | 30s | 2.000 | Univariate MLP Target Lags | 0.309 | 0.391 | 162.169 | 66.226 | -0.055 | 0.804 | level / smooth=none / lr=0.01 |
| test | raw | 1.000 | Univariate MLP Target Lags | 0.310 | 0.393 | 214.055 | 64.638 | -0.005 | 0.802 | first_difference / smooth=none / lr=0.0001 |
| test | raw | 2.000 | Univariate MLP Target Lags | 0.310 | 0.394 | 213.659 | 64.736 | -0.005 | 0.801 | first_difference / smooth=none / lr=0.0001 |