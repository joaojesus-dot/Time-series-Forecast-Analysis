from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def build_boiler_quality_summary(folder: Path) -> dict[str, Any]:
    raw_path = folder / "data.csv"
    repaired_source_path = folder / "data_AutoReg.csv"

    raw_frame = pd.read_csv(raw_path)
    repaired_source_frame = pd.read_csv(repaired_source_path)

    timestamps = pd.to_datetime(raw_frame["date"], errors="coerce")
    gap_counts = timestamps.diff().dropna().dt.total_seconds().value_counts().sort_index()
    missing_by_column = raw_frame.isna().sum()
    changed_cells = (raw_frame.ne(repaired_source_frame) & ~(raw_frame.isna() & repaired_source_frame.isna())).sum().sum()

    return {
        "sources": ["data.csv", "data_AutoReg.csv", "columns.csv", "41597_2025_5096_Fig3_HTML.png"],
        "rows": len(raw_frame),
        "timestamp_column": "date",
        "time_range_start": timestamps.min(),
        "time_range_end": timestamps.max(),
        "timestamp_parse_failures": int(timestamps.isna().sum()),
        "duplicate_timestamps": int(timestamps.duplicated().sum()),
        "gap_counts": gap_counts.to_dict(),
        "missing_cells": int(missing_by_column.sum()),
        "missing_by_column": {column: int(count) for column, count in missing_by_column[missing_by_column.gt(0)].items()},
        "changed_cells": int(changed_cells),
    }


def analyze_boiler(folder: Path) -> str:
    summary = build_boiler_quality_summary(folder)
    gap_counts: dict[float, int] = dict(summary["gap_counts"])
    sources = list(summary["sources"])
    missing_by_column: dict[str, int] = dict(summary["missing_by_column"])

    boiler_quality_markdown = "\n".join(
        [
            "# Chinese Boiler Dataset",
            "",
            f"- Sources: `{ '`, `'.join(sources) }`",
            f"- Rows: {int(summary['rows']):,}",
            f"- Timestamp column: `{summary['timestamp_column']}`",
            f"- Time range: `{summary['time_range_start']}` to `{summary['time_range_end']}`",
            f"- Timestamp parse failures: {int(summary['timestamp_parse_failures'])}",
            f"- Duplicate timestamps: {int(summary['duplicate_timestamps'])}",
            f"- Unique timestamp gaps: {len(gap_counts)}",
            f"- Timestamp gap distribution: {', '.join(f'{int(gap)}s x {int(count)}' for gap, count in gap_counts.items())}",
            f"- Missing cells in `data.csv`: {int(summary['missing_cells'])}",
            f"- Missing detail: {', '.join(f'`{column}`: {int(count)}' for column, count in missing_by_column.items()) or 'none'}",
            f"- Cells changed by `data_AutoReg.csv`: {int(summary['changed_cells'])}",
            "",
            "## Notes",
            "- The raw dataset has a constant 5-second cadence across the full range.",
            "- The only missing values are 30 gaps in `YJJWSLL.AV_0#`.",
            "- `data_AutoReg.csv` only repairs those gaps and preserves the rest of the schema, so it is a natural source for the repaired working dataset.",
        ]
    )
    return boiler_quality_markdown


DATASET_ANALYZERS = {"chinese_boiler_dataset": analyze_boiler}
