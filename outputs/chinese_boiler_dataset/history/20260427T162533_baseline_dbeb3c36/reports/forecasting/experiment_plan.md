# Experiment Plan

## Goal
- Define one forecasting problem that every active model solves under the same protocol.
- Keep the protocol stable so model quality differences are attributable to the model and its configuration.
- Add new model families without redefining the target, split logic, or evaluation metrics.

## Fixed Problem Definition
- Target variable: `TE_8313B.AV_0#`.
- Dataset prefix: `subset_B`.
- Target granularities: `raw`, `30s`, `1min`, `5min`.
- Base raw cadence: `5s`.
- Forecast horizon: `10min` for every granularity.
- Lookback window: `10min` for every granularity.
- Step counts are derived from each granularity's timestamp gap.

## Chronological Split
- Model train: first `70%` of rows.
- Validation: next `10%` of rows.
- Test: final `20%` of rows.
- The final test period is never used for scaling, hyperparameter choice, or early stopping.
- Cross-model comparison should use the same target, horizon, granularity set, and test period.

## Scaling Rule
- Scaling method: `standard`.
- Fit split: `train`.
- Apply the fitted scaler to validation and test where scaling is needed by the model.
- Report metrics on the original target scale after inverse transform.

## Supervised Window Rule
- Each sample uses only lagged target values available before the forecast origin.
- The prediction target is the value exactly `10min` ahead.
- Windows crossing split boundaries are not allowed.

## Evaluation Metrics
- Primary metrics: MAE and RMSE.
- Secondary metrics: MAPE and sMAPE are allowed because the target temperature does not get close to zero.
- Bias should be tracked to detect systematic underprediction or overprediction.

## Organizer Rule
- Reports should describe the active model, its configuration, the comparable protocol, and the resulting metrics in a common structure.
- New models should add a new model card and new rows in the comparison tables instead of inventing a new report layout.

## Pipeline Diagram
```text
raw target series -> repair/preprocessing -> chronological split -> optional scaling
-> model-specific training inputs -> model fit -> inverse transform if needed -> shared metrics
```