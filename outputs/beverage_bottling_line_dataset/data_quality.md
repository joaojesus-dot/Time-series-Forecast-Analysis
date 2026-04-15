# Beverage Bottling Line Dataset

- Source: `production_raw.xlsx`
- Sheets: `processed_hourly`, `daily_operation_summary`, `hourly_operation_breakdown`, `downtime_event_log`
- `processed_hourly` rows: 272, duplicate hour keys: 0, missing cells: 0
- `daily_operation_summary` rows: 59, duplicate (date, product) keys: 0, missing cells: 0
- `hourly_operation_breakdown` rows: 265, duplicate hour keys: 3, missing cells: 0
- `downtime_event_log` rows: 1,388, duplicate ids: 0, parse failures: 0
- Daily time-balance mismatches (`monitored != operation + pause`): 30
- Hourly time-balance mismatches: 0
- Downtime duration mismatches: 0

## Notes
- The workbook is structurally clean: no missing cells and unique keys for most sheets.
- `hourly_operation_breakdown` contains 3 repeated hour keys on 2022-09-20, so that sheet should be deduplicated or aggregated before modeling.
- `daily_operation_summary` has 30 rows where monitored time does not equal operation time plus pause time, so the daily totals need reconciliation before downstream use.