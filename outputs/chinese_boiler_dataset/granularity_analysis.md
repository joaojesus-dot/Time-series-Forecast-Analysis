# Boiler Granularity Analysis

## Decision
- Candidate B is generated at `raw`, `30s`, `1min`, and `5min` granularities.
- The raw cadence is 5 seconds and is already known to be regular.
- Aggregation uses arithmetic mean because the selected variables are continuous process measurements.
- Missing-window analysis is not repeated here; data quality already confirmed a constant 5-second cadence.

## Generated Datasets
- `subset_B_raw`: 86,400 rows from `2022-03-27 14:28:54` to `2022-04-01 14:28:49`.
- `subset_B_30s`: 14,401 rows from `2022-03-27 14:28:30` to `2022-04-01 14:28:30`.
- `subset_B_1min`: 7,201 rows from `2022-03-27 14:28:00` to `2022-04-01 14:28:00`.
- `subset_B_5min`: 1,441 rows from `2022-03-27 14:25:00` to `2022-04-01 14:25:00`.

## Unit Assumptions
- Engineering units are not explicitly provided by the source metadata.
- Temperature is treated as Celsius, oxygen as `% O2`, fan current as amperes, and flow/pressure variables as source-native Chinese DCS units inferred from tag meaning and magnitude.
- Flow variables are plotted separately because primary fan flow, return-air flow, and steam flow have very different magnitudes.

## Generated Plots
- Raw Candidate B variables are plotted by physical segment for interpretation.
- The target is plotted across all granularities in one comparison figure.
- Candidate C is represented by one raw control-loop plot; extra Candidate C granularities are deferred until modeling.