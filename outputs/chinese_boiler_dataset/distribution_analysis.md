# Boiler Distribution Analysis

## Target Distribution
- `raw`: mean `774.06`, std `10.68`, IQR outliers `1.44%`.
- `30s`: mean `774.06`, std `10.67`, IQR outliers `1.44%`.
- `1min`: mean `774.06`, std `10.65`, IQR outliers `1.43%`.
- `5min`: mean `774.06`, std `10.45`, IQR outliers `1.39%`.

## Main Outlier Contributors In `subset_B_raw`
- `AIR_8301A.AV_0#`: `3.44%` IQR outliers.
- `PTCA_8322A.AV_0#`: `3.01%` IQR outliers.
- `YFJ3_ZD2.AV_0#`: `2.52%` IQR outliers.
- `YFJ3_AI.AV_0#`: `2.26%` IQR outliers.
- `ZZQBCHLL.AV_0#`: `1.98%` IQR outliers.
- `TE_8313B.AV_0#`: `1.44%` IQR outliers.

## Interpretation
- The target distribution is stable across granularities; aggregation changes the spread only slightly.
- Cross-variable distribution comparisons should use standardized plots because sensor units and magnitudes differ.
- Outlier counts are screening indicators, not treatment decisions.