from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Boiler data preparation
# ---------------------------------------------------------------------------
# The boiler source provides a raw file and an AutoReg-repaired file. We keep
# this repair step explicit because all later KDD artifacts must be traceable to
# the same cleaned working frame.

def load_repaired_boiler_frame(data_folder: Path) -> pd.DataFrame:
    raw = pd.read_csv(data_folder / "data.csv")
    autoreg = pd.read_csv(data_folder / "data_AutoReg.csv")
    return raw.combine_first(autoreg)


def build_feature_subset(
    dataset: pd.DataFrame,
    target_column: str,
    feature_columns: list[str],
    timestamp_column: str = "date",
) -> pd.DataFrame:
    return dataset[[timestamp_column, target_column, *feature_columns]].copy()


# ---------------------------------------------------------------------------
# Granularity handling
# ---------------------------------------------------------------------------
# The original cadence is known to be 5 seconds. Resampling only changes the
# output granularity and uses mean aggregation because these are continuous
# process measurements, not event counts.

def change_granularity(
    dataset: pd.DataFrame,
    output_granularity: str | None,
    timestamp_column: str = "date",
) -> pd.DataFrame:
    granular = dataset.copy()
    granular[timestamp_column] = pd.to_datetime(granular[timestamp_column], errors="coerce")
    granular = granular.dropna(subset=[timestamp_column]).sort_values(timestamp_column)

    if output_granularity is None:
        return granular.reset_index(drop=True)

    return (
        granular.set_index(timestamp_column)
        .resample(output_granularity)
        .mean(numeric_only=True)
        .reset_index()
    )


def build_granularity_versions(
    dataset: pd.DataFrame,
    granularity_options: dict[str, str | None],
    dataset_name: str,
    timestamp_column: str = "date",
) -> dict[str, pd.DataFrame]:
    return {
        f"{dataset_name}_{granularity_key}": change_granularity(dataset, output_granularity, timestamp_column)
        for granularity_key, output_granularity in granularity_options.items()
    }


# ---------------------------------------------------------------------------
# Target transformations
# ---------------------------------------------------------------------------
# These transformations are diagnostics for the modeling stage. They are kept as
# series records so plotting and statistical summaries can reuse the same values
# without recalculating rolling means or first differences.

def infer_seconds_per_step(dataset: pd.DataFrame, timestamp_column: str = "date") -> float:
    timestamps = pd.to_datetime(dataset[timestamp_column], errors="coerce")
    diffs = timestamps.diff().dropna().dt.total_seconds()
    return float(diffs.median()) if not diffs.empty else 5.0


def build_target_transform_series(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    smoothing_windows: dict[str, int],
    timestamp_column: str = "date",
) -> list[dict[str, Any]]:
    transform_series = []

    for id_key, dataset in datasets.items():
        frame = dataset.copy()
        frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
        granularity = id_key.rsplit("_", 1)[-1]
        target = frame[target_column]
        seconds_per_step = infer_seconds_per_step(frame, timestamp_column)

        transform_series.append(
            _transform_record(id_key, granularity, "original", "none", 1, frame[timestamp_column], target)
        )

        for window_label, window_seconds in smoothing_windows.items():
            window_steps = max(1, round(window_seconds / seconds_per_step))
            if window_steps < 2:
                continue

            smoothed = target.rolling(window=window_steps, min_periods=window_steps).mean()
            transform_series.append(
                _transform_record(
                    id_key,
                    granularity,
                    "trailing_rolling_mean",
                    window_label,
                    window_steps,
                    frame[timestamp_column],
                    smoothed,
                )
            )

        transform_series.append(
            _transform_record(id_key, granularity, "first_difference", "none", 1, frame[timestamp_column], target.diff())
        )

    return transform_series


def _transform_record(
    id_key: str,
    granularity: str,
    transform: str,
    window_label: str,
    window_steps: int,
    dates: pd.Series,
    values: pd.Series,
) -> dict[str, Any]:
    return {
        "id_key": id_key,
        "granularity": granularity,
        "transform": transform,
        "window": window_label,
        "window_steps": window_steps,
        "dates": dates,
        "values": values,
    }
