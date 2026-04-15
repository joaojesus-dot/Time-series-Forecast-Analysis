# Chinese Boiler Dataset

- Sources: `data.csv`, `data_AutoReg.csv`, `columns.csv`
- Rows: 86,400
- Timestamp column: `date`
- Time range: `2022-03-27 14:28:54` to `2022-04-01 14:28:49`
- Timestamp parse failures: 0
- Duplicate timestamps: 0
- Most common timestamp gap: 5 seconds
- Missing cells in `data.csv`: 30
- Missing detail: `YJJWSLL.AV_0#`: 30
- Cell differences between `data.csv` and `data_AutoReg.csv`: 30

## Notes
- The main file is strongly regular in time: a constant 5-second cadence across the full range.
- The only missing values are 30 gaps in `YJJWSLL.AV_0#`.
- `data_AutoReg.csv` keeps the same schema and fills exactly those 30 missing values, so it looks like a repaired/imputed variant of `data.csv`.