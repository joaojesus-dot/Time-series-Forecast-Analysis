# Boiler Granularity Analysis

## Milestone 1: Modeling Scope
- Forecasting target: `TE_8313B.AV_0#`.
- Primary modeling dataset: `subset_B`, the heatmap-informed reduced physical dataset.
- Control-aware comparison dataset: `subset_C`, which adds explicit desuperheating control-loop variables.
- The repaired full dataset remains available for traceability, but the modeling path is based on the reduced subsets.

## Milestone 2: Temporal Granularity
- The known raw sampling interval is 5 seconds.
- Four temporal versions are generated for each reduced subset: `raw`, `30s`, `1min`, and `5min`.
- The dynamic dataset identifier follows the pattern `<subset>_<granularity>`, for example `subset_B_raw`, `subset_B_30s`, `subset_B_1min`, and `subset_B_5min`.
- Aggregation uses the arithmetic mean, which is appropriate at this stage because the selected variables are continuous process measurements.
- Missing-window analysis is intentionally not included here because the source cadence has already been confirmed as regular.
- Raw variance across all variables is not used as a summary metric because the sensors have different physical units and magnitudes.

## Plot Scale And Units
- The source metadata provides variable descriptions but does not explicitly provide engineering units.
- Units are therefore treated as assumptions based on tag meaning, magnitude, and common Chinese industrial boiler conventions.
- Temperature variables are interpreted as Celsius rather than Kelvin.
- Oxygen is interpreted as percent O2.
- Fan current is interpreted as amperes.
- Vibration is interpreted as a source-native vibration unit, likely mm/s or micrometers.
- Pressure and differential-pressure variables are interpreted as source-native Chinese DCS pressure units, likely Pa, kPa, or MPa depending on tag and magnitude.
- Flow variables are interpreted as source-native flow units. Primary and return-air flows are likely m3/h or Nm3/h, while compensated steam flow may be t/h.
- Plot y-axis labels are intentionally short; detailed unit assumptions are stored in `feature_selection_note.json`.
- Flow variables are plotted separately because their magnitudes are very different: primary fan outlet flow is around 46,000, return-air flow is around 700, and compensated steam flow is around 60.

## Generated Data Files
- Candidate B: `subset_B_raw.csv`, `subset_B_30s.csv`, `subset_B_1min.csv`, `subset_B_5min.csv`.
- Candidate C: `subset_C_raw.csv`, `subset_C_30s.csv`, `subset_C_1min.csv`, `subset_C_5min.csv`.
- Backward-compatible raw aliases are also written as `subset_B.csv` and `subset_C.csv`.
- A compact metadata summary is written to `granularity_summary.csv`.

## Generated Plots
- Time-series plots are generated for Candidate B at each granularity.
- The plots are grouped by reduced physical segment and separated when units or magnitudes would distort interpretation.
- Candidate B is plotted first because it is the main working dataset for the next KDD stage.