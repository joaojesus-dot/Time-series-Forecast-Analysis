# Forecasting Config Notes

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

- `experimental_protocol.dataset_prefix`: dataset name prefix used to resolve the derived input frames.
- `experimental_protocol.target_granularities`: aggregated datasets used by the forecasting models.
- `experimental_protocol.forecast_horizon`: future distance predicted by each model.
- `experimental_protocol.lookback_window`: history length used to build lag windows for MLP.
- `experimental_protocol.profiling_granularities`: datasets included in autocorrelation and stationarity profiling.
- `experimental_protocol.differentiation_orders`: difference orders evaluated during MLP preparation.
- `experimental_protocol.train_only_smoothing_windows`: default smoothing windows applied to the MLP train split only.
- `experimental_protocol.train_only_smoothing_windows_by_granularity`: per-granularity smoothing windows applied to the MLP train split only.
- `experimental_protocol.splits`: chronological train and test fractions.
- `experimental_protocol.scaling.method`: target scaling method fitted on the train split.
- `experimental_protocol.metrics`: metrics written to the comparison tables.
- `experimental_protocol.supervised_window.input`: lag-window feature source for the MLP.
- `experimental_protocol.supervised_window.target`: target position used when building each supervised window.
- `experimental_protocol.supervised_window.allow_split_boundary_crossing`: keeps each window inside one chronological split.
- `write_full_forecasts`: writes full per-timestamp forecast CSVs when enabled.
- `write_forecast_plots`: enables forecast comparison plots.
- `run_tracking.enabled`: archives each completed forecasting run into the history folder.
- `run_tracking.tag`: short label appended to the run id.
- `run_tracking.history_root`: folder that stores archived run snapshots.
- `run_tracking.copy_plots`: copies forecasting plots into the archived run snapshot.

## `univariate.arima`

- `enabled`: toggles univariate ARIMA execution.
- `order`: active `(p, d, q)` order.
- `season_length`: season length passed to StatsForecast ARIMA.
- `seasonal_order`: seasonal `(P, D, Q)` order passed to StatsForecast ARIMA.
- `method`: fitting method passed to StatsForecast ARIMA.

## `univariate.mlp`

- `enabled`: toggles univariate MLP execution.
- `engine`: forecasting backend; currently `neuralforecast`.
- `hidden_units_strategy`: chooses how the hidden-layer width is derived.
- `activation`: hidden-layer activation function; NeuralForecast MLP uses ReLU internally.
- `optimizer`: optimizer label used for reporting; NeuralForecast MLP defaults to Adam.
- `learning_rate_grid`: learning rates evaluated for each preparation candidate.
- `min_steps`: minimum training steps required by the assignment rule.
- `max_steps`: training step cap used after the minimum has been satisfied.
- `num_layers`: hidden-layer count; `1` follows the assignment's single-hidden-layer MLP.
- `batch_size`: series batch size passed to NeuralForecast.
- `windows_batch_size`: supervised-window batch size passed to NeuralForecast.
- `accelerator`: Lightning accelerator used by NeuralForecast, for example `gpu` or `cpu`.
- `devices`: number of accelerator devices requested.
- `random_state`: reproducibility seed passed to NeuralForecast MLP.
- `selection_mode`: label used in reports for the current candidate comparison strategy.
- `write_window_data`: reserved switch for writing raw window matrices in future revisions.

## `multivariate`

- `enabled`: toggles multivariate execution.
- `target_granularities`: aggregated datasets used by the multivariate models.
- `feature_set`: non-target feature set used by the multivariate models.
- `standardize_exogenous`: standardizes exogenous regressors before multivariate fitting.
- `max_train_rows`: row limit used to prevent memory-heavy multivariate ARIMA runs.
