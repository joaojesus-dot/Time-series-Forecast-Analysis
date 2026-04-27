# Experiment Plan

## Scope
- Target variable: `TE_8313B.AV_0#`.
- Raw input cadence: `5s`.
- Modeled aggregation levels: `30s`, `1min`.

## Chronological Splits
- Train: `70%` of rows.
- Validation: `10%` of rows.
- Test: `20%` of rows.
- Validation and test remain strictly after train in time order.

## Preparation Candidates
- Scaling method: `minmax` fitted on train only.
- Differentiation orders tested: `0, 1, 2`.
- Train-only smoothing windows tested: `none`, `30s`, `1min`.
- Candidate preparation is selected by validation MAE before inspecting test performance.

## MLP Rules
- Lookback window: `10min`.
- Forecast horizon: `10min`.
- Input features: target lags only.
- Hidden architecture: single hidden layer.
- Activation: `relu`.
- Solver: `sgd`.
- Learning-rate policy: `adaptive`.
- Initial learning-rate grid: `[0.001, 0.005, 0.01]`.
- Max iterations: `5000`.