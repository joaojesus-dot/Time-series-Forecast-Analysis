from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


SUPPORTED_AGGREGATIONS = {
    "mean": "mean",
    "median": "median",
    "last": "last",
    "first": "first",
    "min": "min",
    "max": "max",
    "std": "std",
}


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
    resampling_policy: dict[str, Any] | None = None,
) -> pd.DataFrame:
    granular = dataset.copy()
    granular[timestamp_column] = pd.to_datetime(granular[timestamp_column], errors="coerce")
    granular = granular.dropna(subset=[timestamp_column]).sort_values(timestamp_column)

    if output_granularity is None:
        return granular.reset_index(drop=True)

    policy = normalize_resampling_policy(resampling_policy, timestamp_column)
    value_columns = [column for column in granular.columns if column != timestamp_column]
    aggregation_map = build_aggregation_map(value_columns, policy)
    indexed = granular.set_index(timestamp_column)
    resampler = indexed.resample(
        output_granularity,
        label=policy["label"],
        closed=policy["closed"],
        origin=policy["origin"],
    )
    resampled = resampler.agg(aggregation_map)

    if policy["drop_partial_windows"]:
        expected_count = expected_rows_per_window(output_granularity, policy["input_frequency"])
        window_counts = resampler.size()
        resampled = resampled.loc[window_counts[window_counts == expected_count].index]

    return resampled.dropna(how="all").reset_index()


def build_granularity_versions(
    dataset: pd.DataFrame,
    granularity_options: dict[str, str | None],
    dataset_name: str,
    timestamp_column: str = "date",
    resampling_policy: dict[str, Any] | None = None,
) -> dict[str, pd.DataFrame]:
    return {
        f"{dataset_name}_{granularity_key}": change_granularity(
            dataset,
            output_granularity,
            timestamp_column,
            resampling_policy,
        )
        for granularity_key, output_granularity in granularity_options.items()
    }


def normalize_resampling_policy(
    resampling_policy: dict[str, Any] | None,
    timestamp_column: str,
) -> dict[str, Any]:
    policy = {
        "timestamp_column": timestamp_column,
        "input_frequency": "5s",
        "label": "right",
        "closed": "left",
        "origin": "start",
        "drop_partial_windows": True,
        "default_aggregation": "mean",
        "column_aggregations": {},
    }
    if resampling_policy:
        policy.update(resampling_policy)

    validate_aggregation_name(str(policy["default_aggregation"]))
    for aggregation_name in policy["column_aggregations"].values():
        validate_aggregation_name(str(aggregation_name))

    return policy


def build_aggregation_map(value_columns: list[str], policy: dict[str, Any]) -> dict[str, str]:
    default_aggregation = SUPPORTED_AGGREGATIONS[str(policy["default_aggregation"])]
    column_aggregations = {
        column: SUPPORTED_AGGREGATIONS[str(aggregation_name)]
        for column, aggregation_name in policy["column_aggregations"].items()
        if column in value_columns
    }
    return {column: column_aggregations.get(column, default_aggregation) for column in value_columns}


def validate_aggregation_name(aggregation_name: str) -> None:
    if aggregation_name not in SUPPORTED_AGGREGATIONS:
        allowed = ", ".join(sorted(SUPPORTED_AGGREGATIONS))
        raise ValueError(f"Unsupported aggregation '{aggregation_name}'. Allowed values: {allowed}.")


def expected_rows_per_window(output_granularity: str, input_frequency: str) -> int:
    output_delta = pd.Timedelta(output_granularity)
    input_delta = pd.Timedelta(input_frequency)
    if input_delta <= pd.Timedelta(0):
        raise ValueError("input_frequency must be positive.")
    if output_delta < input_delta:
        raise ValueError("output_granularity must be greater than or equal to input_frequency.")
    ratio = output_delta / input_delta
    if ratio != int(ratio):
        raise ValueError(
            f"output_granularity '{output_granularity}' must be an integer multiple of input_frequency "
            f"'{input_frequency}'."
        )
    return int(ratio)


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
