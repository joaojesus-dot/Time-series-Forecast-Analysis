# Garment Sewing Lines Dataset

- Source: `merged_filtered_monotonic.csv`
- Rows: 1,151
- Slot pairs: 11 target/output pairs
- Explicit timestamp column: none
- Missing cells: 0
- Duplicate rows: 0
- Rows with monotonic targets: 1,151 / 1,151
- Rows with monotonic outputs: 1,122 / 1,151
- Cells where output is above target: 4,913
- Output minus target range: -3360 to 6494

## Notes
- This is a wide slot-by-slot table, so time is implicit in the slot number rather than stored as a timestamp.
- Targets are fully monotonic; outputs regress in 29 rows, which is worth validating against the collection logic.
- Output can be both below and above target, so target/output drift should be treated as a real signal, not an error by default.