# Univariate Forecasting Analysis

## Scope
- Target: `TE_8313B.AV_0#`.
- Active data source: `subset_B_raw` only.
- Current statistical baseline: fixed-order univariate ARIMA.
- Next model family prepared: MLP using scaled target lags only.

## Target Scaling
- Scaler: `standard` fitted on the train split only.
- Train mean: `773.075590`.
- Train std: `10.737163`.
- Scaled series artifact: `data/chinese_boiler_dataset/derived/raw_target_scaled.csv`.

## MLP Preparation
- Lookback: `60` raw steps (`0 days 00:05:00`).
- Horizon: `60` raw steps (`0 days 00:05:00`).
- Train windows available: `69001`.
- Test windows available: `17161`.
- Window arrays are intentionally not written yet; they should be generated in memory when the MLP trainer is added.

## Current Baseline
- Best raw ARIMA result: `ARIMA_p1_d0_q1` with MAE `7.910` and RMSE `9.757`.