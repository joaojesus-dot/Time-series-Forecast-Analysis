# Boiler Dataset Overview

## Scope
- The project is focused on the Chinese boiler dataset.
- Repaired working dataset: `data/chinese_boiler_dataset/derived/boiler_repaired.csv`.
- Forecasting target: `TE_8313B.AV_0#`.
- Candidate B is the primary reduced forecasting dataset.
- Candidate C is retained as a control-aware comparison for later modeling.

## Feature Sets
- Candidate A features: `8`.
- Candidate B features: `13`.
- Candidate C features: `15`.
- Feature selection is derived directly from `python_scripts/config/boiler_family_reduction.json`.

## Raw Target
- Mean: `774.063`.
- Standard deviation: `10.682`.
- Minimum: `737.700`.
- Maximum: `818.080`.

## Source Data Quality
# Chinese Boiler Dataset

- Sources: `data.csv`, `data_AutoReg.csv`, `columns.csv`, `41597_2025_5096_Fig3_HTML.png`
- Rows: 86,400
- Timestamp column: `date`
- Time range: `2022-03-27 14:28:54` to `2022-04-01 14:28:49`
- Timestamp parse failures: 0
- Duplicate timestamps: 0
- Unique timestamp gaps: 1
- Timestamp gap distribution: 5s x 86399
- Missing cells in `data.csv`: 30
- Missing detail: `YJJWSLL.AV_0#`: 30
- Cells changed by `data_AutoReg.csv`: 30

## Notes
- The raw dataset has a constant 5-second cadence across the full range.
- The only missing values are 30 gaps in `YJJWSLL.AV_0#`.
- `data_AutoReg.csv` only repairs those gaps and preserves the rest of the schema, so it is a natural source for the repaired working dataset.