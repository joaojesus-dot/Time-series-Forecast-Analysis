# Presentation Findings

## Dataset Selection
- The Chinese boiler dataset was selected because it has a regular 5-second cadence, clear physical structure, and only 30 missing cells, all repaired by `data_AutoReg.csv`.

## Forecasting Target
- The target is `TE_8313B.AV_0#`, interpreted as upper furnace chamber temperature.
- In the raw Candidate B dataset, the target mean is `774.06` and standard deviation is `10.68`.

## Feature Set
- Candidate B is the main reduced feature set because it balances physical reasoning with observed heatmap structure.
- Candidate C is retained as a control-aware comparison because boiler control signals may improve forecasting but change interpretation.

## Granularity
- Candidate B is generated at raw, 30-second, 1-minute, and 5-minute granularities.
- Aggregation should be compared by validation performance rather than chosen visually.

## Distribution And Outliers
- The largest outlier contributor in `subset_B_raw` is `AIR_8301A.AV_0#` with `2970` IQR events.
- Outlier clusters are multivariate, so automatic deletion is not justified.

## Smoothing And Differencing
- Raw target std is `10.68`; raw target with 5-minute trailing smoothing has std `10.45`.
- Smoothing does not add enough value to become the default target transformation.
- First differencing should be tested through ARIMA orders rather than selected manually at this stage.