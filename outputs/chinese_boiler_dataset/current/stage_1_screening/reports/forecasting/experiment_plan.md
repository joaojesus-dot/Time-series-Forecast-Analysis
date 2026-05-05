# Experiment Plan

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Raw input cadence: `5s`.
- Modeled aggregation levels: `raw`, `30s`, `1min`.

## Chronological Splits
- Train: `80%` of rows.
- Test: `20%` of rows.
- Test remains strictly after train in time order.

## Preparation Candidates
- Scaling method: `standard` fitted on train only.
- Differentiation orders tested: `0, 1, 2`.
- Default train-only smoothing windows tested: `none`.
- Per-granularity smoothing windows: `{"raw": ["none", "30s"], "30s": ["none", "1min"], "1min": ["none", "2min"]}`.
- Test forecasts are written for every candidate for manual visual and metric comparison.

## MLP Rules
- Lookback window: `10min`.
- Forecast horizon: `1step`.
- Horizon note: `1step` means one row ahead at each granularity, so raw, 30s, and 1min metrics are not equal real-time horizons.
- Input features: target lags only.
- Hidden architecture: single hidden layer.
- Engine: `neuralforecast`.
- Activation: `relu`.
- Optimizer: `adam`.
- Learning-rate grid: `[0.0001, 0.001, 0.005, 0.01]`.
- Minimum training steps: `5000`.
- Maximum training steps: `7500`.
- Accelerator: `gpu`.
- Windows batch size: `4096`.
- DataLoader workers: `0`.
- DataLoader pin memory: `True`.
- Comparison mode: `all_candidates_1step_review`.