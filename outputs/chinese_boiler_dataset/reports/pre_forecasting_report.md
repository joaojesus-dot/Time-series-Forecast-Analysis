# Pre-Forecasting Report

## Resampling
- Candidate B is generated at `raw`, `30s`, `1min`, and `5min` granularities.
- The raw cadence is 5 seconds.
- Resampled timestamps are causal window endpoints.
- Windows are `left`-closed and labeled on the `right` edge.
- Window origin is `start`.
- Incomplete edge windows are dropped: `True`.
- Default aggregation: `mean`.

## Configured Column Aggregation Overrides
- `TV_8329ZC.AV_0#`: `last`.
- `YJJWSLL.AV_0#`: `mean`.

## Generated Modeling Datasets
- `subset_B_raw`: 86,400 rows from `2022-03-27 14:28:54` to `2022-04-01 14:28:49`.
- `subset_B_30s`: 14,400 rows from `2022-03-27 14:29:24` to `2022-04-01 14:28:54`.
- `subset_B_1min`: 7,200 rows from `2022-03-27 14:29:54` to `2022-04-01 14:28:54`.
- `subset_B_5min`: 1,440 rows from `2022-03-27 14:33:54` to `2022-04-01 14:28:54`.

## Smoothing And Differencing
- Smoothing and differencing remain diagnostic transforms for now.
- The level target remains the default modeling target.
- Differencing and smoothing may become modeling inputs later if validation supports them.

## Transform Summary
- `subset_B_raw` `original` `none`: std `10.682`.
- `subset_B_raw` `trailing_rolling_mean` `1min`: std `10.655`.
- `subset_B_raw` `trailing_rolling_mean` `5min`: std `10.446`.
- `subset_B_raw` `first_difference` `none`: std `0.273`.
- `subset_B_raw` `second_difference` `none`: std `0.138`.
- `subset_B_30s` `original` `none`: std `10.673`.
- `subset_B_30s` `trailing_rolling_mean` `1min`: std `10.655`.
- `subset_B_30s` `trailing_rolling_mean` `5min`: std `10.446`.
- `subset_B_30s` `first_difference` `none`: std `1.276`.
- `subset_B_30s` `second_difference` `none`: std `1.307`.
- `subset_B_1min` `original` `none`: std `10.655`.
- `subset_B_1min` `trailing_rolling_mean` `5min`: std `10.446`.
- `subset_B_1min` `first_difference` `none`: std `2.016`.
- `subset_B_1min` `second_difference` `none`: std `2.310`.
- `subset_B_5min` `original` `none`: std `10.444`.
- `subset_B_5min` `first_difference` `none`: std `4.931`.
- `subset_B_5min` `second_difference` `none`: std `6.647`.