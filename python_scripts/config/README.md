# Forecasting Config Notes

This project forecasts an industrial boiler temperature sensor. The current project story is a 3-minute forecast; older exploratory outputs are kept under `outputs/chinese_boiler_dataset/legacy/pre_current_refactor/` until they are reviewed.

## Output Tree

- `outputs/chinese_boiler_dataset/current/`: active, presentation-facing artifacts.
- `outputs/chinese_boiler_dataset/current/stage_0_preprocessing/`: preprocessing reports, tables, and plots.
- `outputs/chinese_boiler_dataset/current/stage_1_screening/`: all-candidate one-step MLP screening outputs.
- `outputs/chinese_boiler_dataset/current/stage_2_horizon_impact/`: top-three baseline MLP runs at `1step` and `3min`, plus comparison plots.
- `outputs/chinese_boiler_dataset/current/stage_3_advanced_models/`: model-dependent Stage 3 outputs at the 3-minute horizon.
- `outputs/chinese_boiler_dataset/history/`: archived forecasting runs, kept untouched by the current-tree refactor.
- `outputs/chinese_boiler_dataset/legacy/pre_current_refactor/`: previous flat active folders moved aside for later review.

## Project Stages

- `stage_0_preprocessing`: prepares repaired data, derived granularity datasets, target profiling, scaling, smoothing, differencing, stationarity, and correlation summaries.
- `stage_1_mlp_screening_1step`: runs the assignment-style NeuralForecast MLP over all 72 preprocessing and learning-rate candidates at a `1step` horizon.
- `stage_2_baseline_1step_top3`: reruns the baseline MLP on the best preprocessing candidate per granularity at `1step`.
- `stage_2_baseline_3min_top3`: reruns the same top-three baseline MLP candidates at the project `3min` horizon.
- `stage_2_compare_horizon_impact`: offline plotting workflow comparing the archived `1step` and `3min` Stage 2 runs.
- `stage_3_univariate_mlp_3min`: runnable advanced univariate MLP stage at `3min`.
- `stage_3_univariate_arima_3min`: future ARIMA grid at `3min`; this is disabled until candidate and order settings are reviewed.
- `stage_3_univariate_exponential_smoothing_3min`: future exponential-smoothing variants at `3min`; this is disabled until candidate and smoothing settings are reviewed.

Stage 3 is model-dependent. Univariate MLP is the first runnable family; ARIMA and exponential smoothing remain placeholders.

## `boiler_preprocessing.json`

- `granularity_options`: maps each dataset label to its resampling cadence.
- `resampling_policy.timestamp_column`: timestamp column used during resampling.
- `resampling_policy.input_frequency`: raw sampling cadence of the source data.
- `resampling_policy.label`: timestamp label assigned to each resampled window.
- `resampling_policy.closed`: boundary convention used by `pandas.resample`.
- `resampling_policy.origin`: anchor used to align resampling windows.
- `resampling_policy.drop_partial_windows`: removes incomplete windows at the dataset edges.
- `resampling_policy.default_aggregation`: default reducer applied to numeric columns.
- `resampling_policy.column_aggregations`: per-column aggregation overrides.

## `boiler_forecasting.json`

- `active_stage`: default named stage used by `orchestrator.py`.
- `stages`: named runtime presets. A stage can set horizon, candidate list, model-family switches, and MLP/ARIMA/ETS parameters.
- `candidate_selections.candidate_per_granularity`: the top preprocessing candidate per granularity used for fair Stage 2 horizon comparison.
- `candidate_selections.stage_3_candidate_pool_pending_review`: wider top-tier candidate pool carried from exploratory work. It is intentionally marked pending review; before Stage 3 starts, this list should be re-analysed against the final 3-minute objective and either reduced, confirmed, or replaced.
- `candidate_selection_notes`: plain-language notes explaining why each candidate list exists and whether it is reviewed or pending review.
- `experimental_protocol.dataset_prefix`: dataset name prefix used to resolve derived input frames.
- `experimental_protocol.target_granularities`: raw, 30s, and 1min aggregated datasets used by forecasting models.
- `experimental_protocol.forecast_horizon`: future distance predicted by the active stage.
- `experimental_protocol.lookback_window`: history length used to build MLP lag windows. It is set to `3min`; with `match_lookback`, hidden width equals the resulting lag count per granularity.
- `experimental_protocol.differentiation_orders`: target difference orders evaluated during MLP preparation.
- `experimental_protocol.train_only_smoothing_windows_by_granularity`: smoothing windows applied only inside the chronological train split.
- `experimental_protocol.splits`: chronological train and test fractions.
- `experimental_protocol.scaling.method`: target scaling method used by forecasting models.
- `experimental_protocol.metrics`: metrics written to comparison tables.
- `experimental_protocol.supervised_window.input`: lag-window feature source for the MLP.
- `experimental_protocol.supervised_window.target`: target position used when building each supervised window.
- `experimental_protocol.supervised_window.allow_split_boundary_crossing`: keeps each supervised window inside one chronological split.
- `write_full_forecasts`: writes full per-timestamp forecast CSVs so plots can be regenerated without refitting.
- `write_forecast_plots`: enables forecast comparison plots.
- `run_tracking.enabled`: archives completed forecasting runs into `history/`.
- `run_tracking.tag`: short label appended to the archived run id.
- `run_tracking.history_root`: archive folder name under the dataset output folder.
- `run_tracking.copy_plots`: copies forecasting plots into archived run snapshots.

MLP preparation fits the target scaler on the raw chronological train split, applies it to raw train/test, and then aggregates each split before differencing and train-only smoothing. `univariate.forecast_output_scale` is `scaled`, so MLP, ARIMA, and exponential-smoothing metrics and plots are reported on the same scaled target level.

## Nixtla And StatsForecast Parameters Used

### `NeuralForecast`

Used as the container that fits and predicts the MLP model.

- `models`: contains the configured MLP instance.
- `freq`: timestamp frequency inferred from the active granularity. It tells NeuralForecast how far apart rows are.

### `MLP`

Used for the univariate neural baseline. The project feeds only lagged target values.

- `h`: forecast horizon in rows after converting `1step` or `3min` to the active granularity.
- `input_size`: number of lagged target values in the input window. With the current policy, this covers 3 minutes of history.
- `hidden_size`: hidden-layer width. `match_lookback` uses the same number as `input_size`; `fixed` uses `hidden_units`.
- `num_layers`: number of hidden layers.
- `learning_rate`: current learning rate from the active stage grid.
- `max_steps`: training-step cap.
- `batch_size`: number of series per batch. This project uses one univariate series.
- `windows_batch_size`: number of supervised windows sampled per training batch.
- `random_seed`: reproducibility seed.
- `accelerator`, `devices`: PyTorch Lightning hardware settings, usually GPU with one device when available.
- `dataloader_kwargs`: Windows-friendly loader settings, currently zero workers and optional pinned memory.
- `candidate_settings`: Stage-specific per-candidate MLP settings. Stage 3 MLP uses this to keep one learning rate per selected candidate and to give 30s a larger training-step cap.

### `cross_validation`

Used as the NeuralForecast fit/predict call for a chronological holdout block, not as teacher-style repeated cross-validation.

- `df`: train plus test series in NeuralForecast format (`unique_id`, `ds`, `y`).
- `n_windows`: one evaluation window.
- `val_size`: zero, so no separate validation fold is carved out.
- `test_size`: chronological test-block size.
- `step_size`: test-block size, so there is one final holdout forecast block.

### `StatsForecast`

Used as the container for ARIMA and exponential-smoothing models.

- `models`: list containing the configured ARIMA/ETS-family model.
- `freq`: active time frequency for the granularity.
- `n_jobs`: parallel job count used by StatsForecast.

### `ARIMA`

Reserved for the future Stage 3 ARIMA run.

- `order`: `(p, d, q)`. The current grid uses equal `p` and `q` values per granularity.
- `season_length`: one, because the current ARIMA setup is non-seasonal.
- `seasonal_order`: `(0, 0, 0)`, meaning no seasonal ARIMA component.
- `method`: fitting method passed through to StatsForecast.

### `SimpleExponentialSmoothingOptimized`

Reserved for the future Stage 3 simple exponential-smoothing run.

- `season_length`: one, because the configured variant is non-seasonal.

### `Holt`

Reserved for the future Stage 3 trend-aware smoothing run.

- `season_length`: one, keeping the model non-seasonal.

### `AutoETS`

Reserved for the future Stage 3 automatic ETS run.

- `season_length`: one, no seasonal cycle.
- `model`: currently `ZZN`, allowing automatic error/trend selection with no seasonality.

## Runtime Arguments

- `--skip-forecasting`: regenerates Stage 0 preprocessing artifacts without fitting forecasting models.
- `--forecasting-from-derived`: skips preprocessing and loads existing `data/chinese_boiler_dataset/derived/subset_B_*` CSV files for forecasting.
- `--forecasting-stage`: overrides `active_stage` for a single run.
- `--forecasting-models`: comma-separated univariate model families to run without editing JSON. Valid values are `mlp`, `arima`, and `ets`.
- `--forecasting-smoke-test`: runs one candidate and one learning rate for one training step.
- `--max-candidates`: limits MLP candidates for timing tests without editing JSON.
- `--max-learning-rates`: limits learning-rate options for timing tests without editing JSON.

## Offline Plot Scripts

- `python_scripts/plots.py --compare-archived-stages`: compares two archived Stage 2 MLP runs on the same preprocessing candidates. It writes metric bars and forecast/error overlays under `current/stage_2_horizon_impact/comparison/`.
- `python_scripts/offline_stage2_plots.py`: legacy helper for the old advanced univariate result set. Treat its outputs as legacy.

## Legacy Cleanup

After the current-tree migration, review `outputs/chinese_boiler_dataset/legacy/pre_current_refactor/`. Promote only artifacts that support the final story into the staged `current/` folders; exclude or remove the rest from Git commits.
