# Boiler Smoothing And Differencing Analysis

## Decision
- Smoothing is not selected as a primary modeling target transformation at this stage.
- Rolling smoothing only slightly reduces target spread and largely duplicates what temporal aggregation already provides.
- First differencing remains relevant for ARIMA testing because it converts the target level into a change series centered near zero.

## Summary Statistics
- `subset_B_raw` `original` `none`: std `10.682`.
- `subset_B_raw` `trailing_rolling_mean` `1min`: std `10.655`.
- `subset_B_raw` `trailing_rolling_mean` `5min`: std `10.446`.
- `subset_B_raw` `first_difference` `none`: std `0.273`.
- `subset_B_30s` `original` `none`: std `10.673`.
- `subset_B_30s` `trailing_rolling_mean` `1min`: std `10.655`.
- `subset_B_30s` `trailing_rolling_mean` `5min`: std `10.446`.
- `subset_B_30s` `first_difference` `none`: std `1.276`.
- `subset_B_1min` `original` `none`: std `10.655`.
- `subset_B_1min` `trailing_rolling_mean` `5min`: std `10.446`.
- `subset_B_1min` `first_difference` `none`: std `2.016`.
- `subset_B_5min` `original` `none`: std `10.444`.
- `subset_B_5min` `first_difference` `none`: std `4.931`.

## Modeling Implication
- ARIMA should test both `d=0` and `d=1` instead of assuming differencing is required.
- LSTM targets should remain in level form unless later validation results show a clear reason to predict differences.