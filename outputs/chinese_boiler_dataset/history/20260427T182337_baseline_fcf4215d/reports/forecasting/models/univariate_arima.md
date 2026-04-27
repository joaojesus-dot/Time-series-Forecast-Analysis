# Univariate ARIMA

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Forecast horizon: `10min`.
- Lookback window: `10min`.

## Active Configuration
| Key | Value |
|---|---|
| enabled | True |
| order | [1, 0, 1] |
| season_length | 1 |
| seasonal_order | [0, 0, 0] |
| method | CSS-ML |

## Metrics
| Granularity | P | D | Q | Mae | Rmse | Mape | Smape | Bias |
|---|---|---|---|---|---|---|---|---|
| 1min | 1 | 0 | 1 | 8.287 | 10.446 | 1.059 | 1.065 | -4.518 |
| 30s | 1 | 0 | 1 | 8.280 | 10.435 | 1.058 | 1.064 | -4.501 |