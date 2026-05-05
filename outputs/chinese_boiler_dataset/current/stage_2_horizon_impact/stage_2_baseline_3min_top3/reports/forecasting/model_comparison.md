# Model Comparison

## Comparable Metrics
| Split | Granularity | Mae Rank | Model Label | Mae | Rmse | Mape | Smape | Bias | R2 | Configuration |
|---|---|---|---|---|---|---|---|---|---|---|
| test | 1min | 1.000 | Univariate MLP Target Lags | 0.303 | 0.384 | 287.657 | 62.908 | 0.003 | 0.810 | first_difference / smooth=none / lr=0.0001 |
| test | 30s | 1.000 | Univariate MLP Target Lags | 0.311 | 0.393 | 146.659 | 66.945 | -0.040 | 0.802 | level / smooth=none / lr=0.0001 |
| test | raw | 1.000 | Univariate MLP Target Lags | 0.310 | 0.392 | 214.262 | 64.629 | -0.005 | 0.803 | first_difference / smooth=none / lr=0.0001 |