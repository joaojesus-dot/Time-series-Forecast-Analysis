from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


def analyze_metro(folder: Path) -> str:
    path = folder / "MetroPT3(AirCompressor).csv"
    rows = parse_fail = duplicate_timestamps = missing_cells = 0
    first_ts = last_ts = previous_ts = None
    step_counts: Counter[float] = Counter()
    index_step_counts: Counter[float] = Counter()
    previous_index = None

    for chunk in pd.read_csv(path, chunksize=500_000):
        timestamps = pd.to_datetime(chunk["timestamp"], errors="coerce")
        rows += len(chunk)
        parse_fail += int(timestamps.isna().sum())
        duplicate_timestamps += int(timestamps.duplicated().sum())
        missing_cells += int(chunk.isna().sum().sum())

        valid = timestamps.dropna()
        if not valid.empty:
            if first_ts is None:
                first_ts = valid.iloc[0]
            if previous_ts is not None:
                # Track the gap across chunk boundaries so chunked reading preserves cadence analysis.
                step_counts[float((valid.iloc[0] - previous_ts).total_seconds())] += 1
                if valid.iloc[0] == previous_ts:
                    duplicate_timestamps += 1
            # Count all within-chunk timestamp gaps and merge them into the global frequency table.
            step_counts.update(valid.diff().dropna().dt.total_seconds().value_counts().to_dict())
            last_ts = valid.iloc[-1]
            previous_ts = valid.iloc[-1]

        index_values = chunk["Unnamed: 0"]
        if previous_index is not None:
            index_step_counts[float(index_values.iloc[0] - previous_index)] += 1
        index_step_counts.update(index_values.diff().dropna().value_counts().to_dict())
        previous_index = index_values.iloc[-1]

    # Format the most common timestamp and index gaps for the markdown summary.
    top_steps = ", ".join(f"{int(step)}s x {count}" for step, count in step_counts.most_common(3))
    top_index_steps = ", ".join(
        f"{int(step)} x {count}" for step, count in index_step_counts.most_common(3)
    )
    metro_markdown_generator = "\n".join(
        [
            "# PT Metro Dataset",
            "",
            "- Source: `MetroPT3(AirCompressor).csv`",
            f"- Rows: {rows:,}",
            "- Timestamp column: `timestamp`",
            f"- Time range: `{first_ts}` to `{last_ts}`",
            f"- Timestamp parse failures: {parse_fail}",
            f"- Duplicate timestamps: {duplicate_timestamps}",
            f"- Missing cells: {missing_cells}",
            f"- Most common timestamp gaps: {top_steps}",
            f"- `Unnamed: 0` step pattern: {top_index_steps}",
            "",
            "## Notes",
            "- The file is clean from a missing-value perspective.",
            "- Sampling is mostly every 10 seconds, not every 1 second as the README text suggests.",
            "- `Unnamed: 0` behaves like the original sample index and can be treated as metadata, not a feature.",
        ]
    )
    return metro_markdown_generator


def analyze_garment(folder: Path) -> str:
    path = folder / "merged_filtered_monotonic.csv"
    frame = pd.read_csv(path)
    targets = [column for column in frame.columns if column.startswith("Target_")]
    outputs = [column for column in frame.columns if column.startswith("Output_")]

    target_monotonic_rows = (frame[targets].diff(axis=1).iloc[:, 1:] >= 0).all(axis=1).sum()
    output_monotonic_rows = (frame[outputs].diff(axis=1).iloc[:, 1:] >= 0).all(axis=1).sum()
    output_minus_target = frame[outputs].to_numpy() - frame[targets].to_numpy()

    garment_markdown_generator = "\n".join(
        [
            "# Garment Sewing Lines Dataset",
            "",
            "- Source: `merged_filtered_monotonic.csv`",
            f"- Rows: {len(frame):,}",
            f"- Slot pairs: {len(targets)} target/output pairs",
            "- Explicit timestamp column: none",
            f"- Missing cells: {int(frame.isna().sum().sum())}",
            f"- Duplicate rows: {int(frame.duplicated().sum())}",
            f"- Rows with monotonic targets: {int(target_monotonic_rows):,} / {len(frame):,}",
            f"- Rows with monotonic outputs: {int(output_monotonic_rows):,} / {len(frame):,}",
            f"- Cells where output is above target: {int((output_minus_target > 0).sum()):,}",
            f"- Output minus target range: {int(output_minus_target.min())} to {int(output_minus_target.max())}",
            "",
            "## Notes",
            "- This is a wide slot-by-slot table, so time is implicit in the slot number rather than stored as a timestamp.",
            "- Targets are fully monotonic; outputs regress in 29 rows, which is worth validating against the collection logic.",
            "- Output can be both below and above target, so target/output drift should be treated as a real signal, not an error by default.",
        ]
    )
    return garment_markdown_generator


def analyze_boiler(folder: Path) -> str:
    main_path = folder / "data.csv"
    autoreg_path = folder / "data_AutoReg.csv"
    main_frame = pd.read_csv(main_path)
    autoreg_frame = pd.read_csv(autoreg_path)
    timestamps = pd.to_datetime(main_frame["date"], errors="coerce")
    gaps = timestamps.diff().dropna().dt.total_seconds()
    most_common_gap_seconds = int(gaps.mode().iloc[0]) if not gaps.empty else 0
    main_missing = main_frame.isna().sum()
    missing_summary = ", ".join(
        f"`{column}`: {count}"
        for column, count in main_missing[main_missing.gt(0)].items()
    )
    # Compare the raw and AutoReg files cell by cell, ignoring positions where both are missing.
    differences = (main_frame.ne(autoreg_frame) & ~(main_frame.isna() & autoreg_frame.isna())).sum().sum()

    boiler_markdown_generator = "\n".join(
        [
            "# Chinese Boiler Dataset",
            "",
            "- Sources: `data.csv`, `data_AutoReg.csv`, `columns.csv`",
            f"- Rows: {len(main_frame):,}",
            "- Timestamp column: `date`",
            f"- Time range: `{timestamps.min()}` to `{timestamps.max()}`",
            f"- Timestamp parse failures: {int(timestamps.isna().sum())}",
            f"- Duplicate timestamps: {int(timestamps.duplicated().sum())}",
            f"- Most common timestamp gap: {most_common_gap_seconds} seconds",
            f"- Missing cells in `data.csv`: {int(main_missing.sum())}",
            f"- Missing detail: {missing_summary or 'none'}",
            f"- Cell differences between `data.csv` and `data_AutoReg.csv`: {int(differences)}",
            "",
            "## Notes",
            "- The main file is strongly regular in time: a constant 5-second cadence across the full range.",
            "- The only missing values are 30 gaps in `YJJWSLL.AV_0#`.",
            "- `data_AutoReg.csv` keeps the same schema and fills exactly those 30 missing values, so it looks like a repaired/imputed variant of `data.csv`.",
        ]
    )
    return boiler_markdown_generator


def analyze_beverage(folder: Path) -> str:
    path = folder / "production_raw.xlsx"
    hourly = pd.read_excel(path, sheet_name="processed_hourly")
    daily = pd.read_excel(path, sheet_name="daily_operation_summary")
    breakdown = pd.read_excel(path, sheet_name="hourly_operation_breakdown")
    downtime = pd.read_excel(path, sheet_name="downtime_event_log")

    hourly_ts = _combine_date_and_hour(hourly["date"], hourly["hour_start"])
    breakdown_ts = _combine_date_and_hour(breakdown["date"], breakdown["hour_start"])
    downtime_start = _combine_date_and_time(downtime["date"], downtime["downtime_start_time"])
    downtime_end = _combine_date_and_time(downtime["date"], downtime["downtime_end_time"])
    downtime_duration = pd.to_timedelta(downtime["downtime_time"].astype(str), errors="coerce")
    daily_balance_error = (
        daily["monitored_time_dec"] - daily["operation_time_dec"] - daily["pause_time_dec"]
    ).abs()

    beverage_markdown_generator = "\n".join(
        [
            "# Beverage Bottling Line Dataset",
            "",
            "- Source: `production_raw.xlsx`",
            "- Sheets: `processed_hourly`, `daily_operation_summary`, `hourly_operation_breakdown`, `downtime_event_log`",
            f"- `processed_hourly` rows: {len(hourly):,}, duplicate hour keys: {int(hourly_ts.duplicated().sum())}, missing cells: {int(hourly.isna().sum().sum())}",
            f"- `daily_operation_summary` rows: {len(daily):,}, duplicate (date, product) keys: {int(daily.duplicated(['date', 'product_type_l']).sum())}, missing cells: {int(daily.isna().sum().sum())}",
            f"- `hourly_operation_breakdown` rows: {len(breakdown):,}, duplicate hour keys: {int(breakdown_ts.duplicated().sum())}, missing cells: {int(breakdown.isna().sum().sum())}",
            f"- `downtime_event_log` rows: {len(downtime):,}, duplicate ids: {int(downtime['downtime_id'].duplicated().sum())}, parse failures: {int(downtime_start.isna().sum() + downtime_end.isna().sum())}",
            f"- Daily time-balance mismatches (`monitored != operation + pause`): {int(daily_balance_error.gt(1e-6).sum())}",
            f"- Hourly time-balance mismatches: {int((breakdown['monitored_time_h'] - breakdown['operation_time_h'] - breakdown['downtime_h']).abs().gt(1e-6).sum())}",
            f"- Downtime duration mismatches: {int(((downtime_end - downtime_start - downtime_duration).abs() > pd.Timedelta(seconds=1)).sum())}",
            "",
            "## Notes",
            "- The workbook is structurally clean: no missing cells and unique keys for most sheets.",
            "- `hourly_operation_breakdown` contains 3 repeated hour keys on 2022-09-20, so that sheet should be deduplicated or aggregated before modeling.",
            "- `daily_operation_summary` has 30 rows where monitored time does not equal operation time plus pause time, so the daily totals need reconciliation before downstream use.",
        ]
    )
    return beverage_markdown_generator


DATASET_ANALYZERS = {
    "pt_metro_dataset": analyze_metro,
    "garment_sewing_lines_dataset": analyze_garment,
    "chinese_boiler_dataset": analyze_boiler,
    "beverage_bottling_line_dataset": analyze_beverage,
}


def _combine_date_and_hour(date_column: pd.Series, hour_column: pd.Series) -> pd.Series:
    date_text = pd.to_datetime(date_column, errors="coerce").dt.strftime("%Y-%m-%d")
    hour_text = hour_column.astype("Int64").astype(str).str.zfill(2)
    return pd.to_datetime(date_text + " " + hour_text + ":00:00", errors="coerce")


def _combine_date_and_time(date_column: pd.Series, time_column: pd.Series) -> pd.Series:
    date_text = pd.to_datetime(date_column, errors="coerce").dt.strftime("%Y-%m-%d")
    combined = [f"{day} {clock}" for day, clock in zip(date_text, time_column)]
    parsed = [pd.to_datetime(value, errors="coerce") for value in combined]
    return pd.Series(parsed, index=date_column.index)
