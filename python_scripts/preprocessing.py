from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import pandas as pd


class Scaler(TypedDict, total=False):
    mean: float
    std: float
    min: float
    max: float
    range: float
    method: str
    scale: float


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


def build_feature_selection(family_reduction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Derive forecasting subsets from the family-reduction config."""
    target_column = find_target_column(family_reduction_rows)
    return {
        "target_column": target_column,
        "candidate_a_features": variables_for_candidate(family_reduction_rows, "A", target_column),
        "candidate_b_features": variables_for_candidate(family_reduction_rows, "B", target_column),
        "candidate_c_features": variables_for_candidate(family_reduction_rows, "C", target_column),
    }


def find_target_column(family_reduction_rows: list[dict[str, Any]]) -> str:
    """Read the single target variable marked in the family-reduction config."""
    for row in family_reduction_rows:
        for representative in row.get("representative_variables", []):
            if representative.get("role") == "target":
                return str(representative["name"])
    raise ValueError("No target representative found in family reduction config.")


def variables_for_candidate(
    family_reduction_rows: list[dict[str, Any]],
    candidate: str,
    target_column: str,
) -> list[str]:
    """Return non-target representative variables assigned to one candidate set."""
    variables = []
    for row in family_reduction_rows:
        for representative in row.get("representative_variables", []):
            name = str(representative["name"])
            if name == target_column:
                continue
            if candidate in representative.get("candidates", []):
                variables.append(name)
    return variables


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


def steps_for_duration(duration: str, frequency: str) -> int:
    """Convert a real-time duration into row steps for one sampling frequency."""
    duration_delta = pd.Timedelta(duration)
    frequency_delta = pd.Timedelta(frequency)
    if frequency_delta <= pd.Timedelta(0):
        raise ValueError("frequency must be positive.")
    if duration_delta < frequency_delta:
        raise ValueError(f"duration '{duration}' must be greater than or equal to frequency '{frequency}'.")
    ratio = duration_delta / frequency_delta
    if ratio != int(ratio):
        raise ValueError(f"duration '{duration}' must be an integer multiple of frequency '{frequency}'.")
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
        transform_series.append(
            _transform_record(
                id_key,
                granularity,
                "second_difference",
                "none",
                1,
                frame[timestamp_column],
                target.diff().diff(),
            )
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


# ---------------------------------------------------------------------------
# Forecasting frame preparation
# ---------------------------------------------------------------------------
# Forecasting models need strict timestamp ordering, chronological splits, and
# scalers fitted only on training data. These are data-preparation concerns, so
# the model-specific modules reuse the helpers here instead of reimplementing
# them.

def build_forecasting_frame(
    dataset: pd.DataFrame,
    id_key: str,
    target_column: str,
    timestamp_column: str = "date",
    exogenous_columns: list[str] | None = None,
) -> pd.DataFrame:
    selected_columns = [timestamp_column, target_column] + list(exogenous_columns or [])
    frame = dataset[selected_columns].copy()
    frame[timestamp_column] = pd.to_datetime(frame[timestamp_column], errors="coerce")
    frame[target_column] = pd.to_numeric(frame[target_column], errors="coerce")
    for column in exogenous_columns or []:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=selected_columns).sort_values(timestamp_column)
    frame = frame.rename(columns={timestamp_column: "ds", target_column: "y"}).assign(unique_id=id_key)
    return frame[["unique_id", "ds", "y"] + list(exogenous_columns or [])]


def split_train_test(frame: pd.DataFrame, train_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    split_index = int(len(frame) * train_fraction)
    if split_index <= 0 or split_index >= len(frame):
        raise ValueError("train_fraction creates an empty train or test set.")
    return frame.iloc[:split_index].copy(), frame.iloc[split_index:].copy()


def split_train_validation_test(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronologically split rows into model-train, validation, and final test sets."""
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    if validation_fraction < 0 or validation_fraction >= train_fraction:
        raise ValueError("validation_fraction must be non-negative and smaller than train_fraction.")

    train_end = int(len(frame) * (train_fraction - validation_fraction))
    validation_end = int(len(frame) * train_fraction)
    if train_end <= 0 or validation_end <= train_end or validation_end >= len(frame):
        raise ValueError("Configured split fractions create an empty train, validation, or test set.")
    return (
        frame.iloc[:train_end].copy(),
        frame.iloc[train_end:validation_end].copy(),
        frame.iloc[validation_end:].copy(),
    )


def infer_exogenous_columns(dataset: pd.DataFrame, target_column: str, timestamp_column: str = "date") -> list[str]:
    excluded = {target_column, timestamp_column}
    return [
        column
        for column in dataset.columns
        if column not in excluded and pd.api.types.is_numeric_dtype(dataset[column])
    ]


def fit_standard_scaler(series: pd.Series) -> Scaler:
    mean = float(series.mean())
    std = float(series.std(ddof=0))
    if pd.isna(std) or std == 0:
        std = 1.0
    return {"mean": mean, "std": std}


def fit_minmax_scaler(series: pd.Series) -> Scaler:
    minimum = float(series.min())
    maximum = float(series.max())
    value_range = maximum - minimum
    if pd.isna(value_range) or value_range == 0:
        value_range = 1.0
    return {"min": minimum, "max": maximum, "range": value_range}


def fit_linear_scaler(series: pd.Series, method: str) -> Scaler:
    if method == "standard":
        scaler = fit_standard_scaler(series)
        scaler["method"] = "standard"
        scaler["scale"] = scaler["std"]
        return scaler
    if method == "minmax":
        scaler = fit_minmax_scaler(series)
        scaler["method"] = "minmax"
        scaler["scale"] = scaler["range"]
        return scaler
    raise ValueError(f"Unsupported scaling method '{method}'.")


def apply_standard_scaler(series: pd.Series, scaler: Scaler) -> pd.Series:
    return (series - _require_float(scaler, "mean")) / _require_float(scaler, "std")


def apply_minmax_scaler(series: pd.Series, scaler: Scaler) -> pd.Series:
    return (series - _require_float(scaler, "min")) / _require_float(scaler, "range")


def apply_linear_scaler(series: pd.Series, scaler: Scaler) -> pd.Series:
    method = str(scaler["method"])
    if method == "standard":
        return apply_standard_scaler(series, scaler)
    if method == "minmax":
        return apply_minmax_scaler(series, scaler)
    raise ValueError(f"Unsupported scaling method '{method}'.")


def _require_float(scaler: Scaler, key: str) -> float:
    value = scaler.get(key)
    if not isinstance(value, (int, float)):
        raise TypeError(f"Scaler entry '{key}' must be numeric.")
    return float(value)


def standardize_columns(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame = train_frame.copy()
    test_frame = test_frame.copy()
    for column in columns:
        scaler = fit_standard_scaler(train_frame[column])
        train_frame[column] = apply_standard_scaler(train_frame[column], scaler)
        test_frame[column] = apply_standard_scaler(test_frame[column], scaler)
    return train_frame, test_frame


def build_scaled_target_frame(
    frame: pd.DataFrame,
    train_fraction: float,
    validation_fraction: float,
    granularity: str,
    target_column: str,
    frequency: str,
    scaling_method: str = "standard",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frame, validation_frame, test_frame = split_train_validation_test(frame, train_fraction, validation_fraction)
    scaler = fit_linear_scaler(train_frame["y"], scaling_method)

    scaled_frame = pd.concat(
        [
            train_frame.assign(split="train"),
            validation_frame.assign(split="validation"),
            test_frame.assign(split="test"),
        ],
        ignore_index=True,
    )
    scaled_frame["y_scaled"] = apply_linear_scaler(scaled_frame["y"], scaler)
    scaling_summary = pd.DataFrame(
        [
            {
                "granularity": granularity,
                "frequency": frequency,
                "target_column": target_column,
                "scaler": scaling_method,
                "fit_split": "train",
                "train_rows": len(train_frame),
                "validation_rows": len(validation_frame),
                "test_rows": len(test_frame),
                "train_mean": scaler.get("mean", np.nan),
                "train_std": scaler.get("std", np.nan),
                "train_min": scaler.get("min"),
                "train_max": scaler.get("max"),
                "train_scale": scaler["scale"],
                "train_start": train_frame["ds"].iloc[0],
                "train_end": train_frame["ds"].iloc[-1],
                "validation_start": validation_frame["ds"].iloc[0],
                "validation_end": validation_frame["ds"].iloc[-1],
                "test_start": test_frame["ds"].iloc[0],
                "test_end": test_frame["ds"].iloc[-1],
            }
        ]
    )
    return scaled_frame, scaling_summary


def statsforecast_frequency(
    granularity: str,
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
) -> str:
    if granularity == "raw":
        return str(resampling_policy["input_frequency"])
    frequency = granularity_options.get(granularity)
    if frequency is None:
        raise ValueError(f"Could not infer frequency for granularity '{granularity}'.")
    return str(frequency)
