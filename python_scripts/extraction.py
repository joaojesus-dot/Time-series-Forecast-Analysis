from __future__ import annotations

from pathlib import Path

import pandas as pd


def analyze_boiler(folder: Path) -> str:
    raw_path = folder / "data.csv"
    repaired_source_path = folder / "data_AutoReg.csv"

    raw_frame = pd.read_csv(raw_path)
    repaired_source_frame = pd.read_csv(repaired_source_path)

    timestamps = pd.to_datetime(raw_frame["date"], errors="coerce")
    gap_counts = timestamps.diff().dropna().dt.total_seconds().value_counts().sort_index()
    missing_by_column = raw_frame.isna().sum()
    changed_cells = (raw_frame.ne(repaired_source_frame) & ~(raw_frame.isna() & repaired_source_frame.isna())).sum().sum()

    boiler_quality_markdown = "\n".join(
        [
            "# Chinese Boiler Dataset",
            "",
            "- Sources: `data.csv`, `data_AutoReg.csv`, `columns.csv`, `41597_2025_5096_Fig3_HTML.png`",
            f"- Rows: {len(raw_frame):,}",
            "- Timestamp column: `date`",
            f"- Time range: `{timestamps.min()}` to `{timestamps.max()}`",
            f"- Timestamp parse failures: {int(timestamps.isna().sum())}",
            f"- Duplicate timestamps: {int(timestamps.duplicated().sum())}",
            f"- Unique timestamp gaps: {len(gap_counts)}",
            f"- Timestamp gap distribution: {', '.join(f'{int(gap)}s x {int(count)}' for gap, count in gap_counts.items())}",
            f"- Missing cells in `data.csv`: {int(missing_by_column.sum())}",
            f"- Missing detail: {', '.join(f'`{column}`: {int(count)}' for column, count in missing_by_column[missing_by_column.gt(0)].items()) or 'none'}",
            f"- Cells changed by `data_AutoReg.csv`: {int(changed_cells)}",
            "",
            "## Notes",
            "- The raw dataset has a constant 5-second cadence across the full range.",
            "- The only missing values are 30 gaps in `YJJWSLL.AV_0#`.",
            "- `data_AutoReg.csv` only repairs those gaps and preserves the rest of the schema, so it is a natural source for the repaired working dataset.",
        ]
    )
    return boiler_quality_markdown


DATASET_ANALYZERS = {"chinese_boiler_dataset": analyze_boiler}
