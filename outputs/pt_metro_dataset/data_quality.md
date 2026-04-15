# PT Metro Dataset

- Source: `MetroPT3(AirCompressor).csv`
- Rows: 1,516,948
- Timestamp column: `timestamp`
- Time range: `2020-02-01 00:00:00` to `2020-09-01 03:59:50`
- Timestamp parse failures: 0
- Duplicate timestamps: 0
- Missing cells: 0
- Most common timestamp gaps: 10s x 1337521, 9s x 128277, 12s x 38321
- `Unnamed: 0` step pattern: 10 x 1516947

## Notes
- The file is clean from a missing-value perspective.
- Sampling is mostly every 10 seconds, not every 1 second as the README text suggests.
- `Unnamed: 0` behaves like the original sample index and can be treated as metadata, not a feature.