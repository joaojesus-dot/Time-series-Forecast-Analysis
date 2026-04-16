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