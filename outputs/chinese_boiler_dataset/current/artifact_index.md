# Current Artifact Index

This folder contains the active, presentation-facing outputs for the industrial boiler temperature forecasting project. Older active artifacts from the previous flat tree were moved to `../legacy/pre_current_refactor/`; archived forecasting runs remain under `../history/`. Stage 0 and Stage 1 currently preserve evidence promoted from the latest reliable pre-refactor artifacts, because those stages were not rerun after the staged output-tree refactor.

## Stage 0 - Preprocessing

Path: `stage_0_preprocessing/`

- `reports/`: data preparation, target profiling, scaling, smoothing, differencing, stationarity, and correlation reports.
- `tables/pre_forecasting_summary/`: compact CSV tables used to support the preprocessing discussion.
- `plots/`: exploratory, reduced-subset, and pre-forecasting target diagnostic plots.
- Current source: promoted from `../legacy/pre_current_refactor/`.

## Stage 1 - MLP Screening At One Step

Path: `stage_1_screening/`

- Purpose: run the assignment-style MLP over all 72 preprocessing candidates at a `1step` horizon.
- `forecasting/`: full forecast CSVs and MLP training history when enabled.
- `reports/forecasting/`: experiment plan, candidate selection, model comparison, and MLP notes.
- `tables/forecasting_summary/`: screening metrics and selected-candidate tables.
- `plots/forecasting/`: ranking plots plus useful training loss and learning-curve diagnostics.
- Current source: promoted from archived run `../history/20260430T014551_standard-real-mlp-1step-review_dc82a37d/`.

## Stage 2 - Horizon Impact Baseline

Path: `stage_2_horizon_impact/`

- `stage_2_baseline_1step_top3/`: baseline MLP on the best candidate per granularity at `1step`.
- `stage_2_baseline_3min_top3/`: the same baseline MLP candidates at the project `3min` horizon.
- `comparison/`: offline comparison plots and CSVs for the same candidates across horizons.

## Stage 3 - Advanced 3-Minute Models

Path: `stage_3_advanced_models/`

- `univariate_mlp/`: first runnable Stage 3 model family. It compares two-hidden-layer NeuralForecast MLP variants at the 3-minute horizon. Because the parent folder already names the model family, its reports, forecast CSVs, and plots live directly under `univariate_mlp/reports/`, `univariate_mlp/forecasting/`, and `univariate_mlp/plots/`.
- `univariate_arima/`: future ARIMA workspace, blocked until model settings are reviewed.
- `univariate_exponential_smoothing/`: future smoothing-model workspace, blocked until model settings are reviewed.

## Legacy Follow-Up

Review `../legacy/pre_current_refactor/` after this migration. Files that still help the project story should be promoted into one of the staged folders above; files that do not help should be excluded from Git commits or removed intentionally.
