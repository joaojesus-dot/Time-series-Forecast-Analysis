# Boiler Exploration Findings

## KDD Focus
- The project is now boiler-only, moving from dataset screening to a domain-driven KDD process.
- The repaired working dataset is built from `data.csv` plus the 30 values filled by `data_AutoReg.csv`.
- Feature reduction is based on boiler physics, family heatmaps, and explicit separation of control-loop variables.

## Findings
- The repaired dataset was saved to `boiler_repaired.csv` for downstream traceability.
- Timestamp spacing is constant across the full series: the only observed gap is 5 seconds.
- Candidate B is the main reduced physical dataset for forecasting analysis.
- Candidate C is the control-aware comparison dataset.
- Feature-selection and family-reduction decisions are stored as JSON artifacts for traceability.