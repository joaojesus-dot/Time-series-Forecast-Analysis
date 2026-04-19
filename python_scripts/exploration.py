from __future__ import annotations

from typing import Any

import pandas as pd


# ---------------------------------------------------------------------------
# Dataset-level summaries
# ---------------------------------------------------------------------------
# Exploration functions return dataframes only. They do not write files or
# create plots, which keeps this module reusable and easy to review.

def build_granularity_summary(
    datasets: dict[str, pd.DataFrame],
    dataset_name: str,
    timestamp_column: str = "date",
) -> pd.DataFrame:
    rows = []
    for id_key, dataset in datasets.items():
        rows.append(
            {
                "id_key": id_key,
                "candidate": dataset_name,
                "granularity": id_key.replace(f"{dataset_name}_", ""),
                "rows": len(dataset),
                "columns": len(dataset.columns),
                "start_date": dataset[timestamp_column].min(),
                "end_date": dataset[timestamp_column].max(),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Distribution statistics
# ---------------------------------------------------------------------------
# The IQR bounds are stored together with the descriptive statistics so outlier
# detection can reuse exactly the same thresholds instead of recalculating them.

def summarize_variables(
    datasets: dict[str, pd.DataFrame],
    timestamp_column: str = "date",
) -> pd.DataFrame:
    rows = []
    for id_key, dataset in datasets.items():
        granularity = id_key.rsplit("_", 1)[-1]
        for column in [col for col in dataset.columns if col != timestamp_column]:
            rows.append(summarize_series(id_key, granularity, column, dataset[column]))
    return pd.DataFrame(rows)


def summarize_series(
    id_key: str,
    granularity: str,
    variable: str,
    values: pd.Series,
) -> dict[str, object]:
    clean_values = values.dropna()
    q1 = clean_values.quantile(0.25)
    q3 = clean_values.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    outlier_count = int(((clean_values < lower_bound) | (clean_values > upper_bound)).sum())

    return {
        "id_key": id_key,
        "granularity": granularity,
        "variable": variable,
        "count": int(clean_values.count()),
        "mean": clean_values.mean(),
        "std": clean_values.std(),
        "min": clean_values.min(),
        "q1": q1,
        "median": clean_values.median(),
        "q3": q3,
        "max": clean_values.max(),
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "skewness": clean_values.skew(),
        "iqr_outlier_count": outlier_count,
        "iqr_outlier_percent": 100 * outlier_count / len(clean_values) if len(clean_values) else 0,
    }


# ---------------------------------------------------------------------------
# Outlier analysis
# ---------------------------------------------------------------------------
# Outliers are calculated per variable using each variable's own IQR bounds.
# This avoids applying one raw threshold across sensors with different units and
# magnitudes. Z-score scaling can make variables comparable for plots or model
# inputs, but a global z-score threshold would still assume similar distribution
# shapes and tail behavior across all sensors, which is not guaranteed here.


def locate_iqr_outliers(
    dataset: pd.DataFrame,
    distribution_summary: pd.DataFrame,
    id_key: str,
    timestamp_column: str = "date",
) -> pd.DataFrame:
    event_frames = []
    bounds = distribution_summary[distribution_summary["id_key"] == id_key].set_index("variable")

    for column in [col for col in dataset.columns if col != timestamp_column]:
        if column not in bounds.index:
            continue

        lower_bound = bounds.at[column, "lower_bound"]
        upper_bound = bounds.at[column, "upper_bound"]
        mask = (dataset[column] < lower_bound) | (dataset[column] > upper_bound)

        if not mask.any():
            continue

        events = dataset.loc[mask, [timestamp_column, column]].copy()
        events = events.rename(columns={timestamp_column: "date", column: "value"})
        events["variable"] = column
        events["direction"] = events["value"].apply(lambda value: "low" if value < lower_bound else "high")
        event_frames.append(events)

    if not event_frames:
        return pd.DataFrame(columns=["date", "variable", "value", "direction"])

    events = pd.concat(event_frames, ignore_index=True)
    events["date"] = pd.to_datetime(events["date"])
    return events.sort_values(["date", "variable"]).reset_index(drop=True)


def summarize_outliers(
    events: pd.DataFrame,
    window: str = "5min",
    top_windows: int = 20,
    top_simultaneous: int = 50,
) -> dict[str, pd.DataFrame]:
    return {
        "variable_counts": build_outlier_variable_counts(events),
        "top_windows": build_top_outlier_windows(events, window, top_windows),
        "simultaneous": build_top_simultaneous_outliers(events, top_simultaneous),
    }


def build_outlier_variable_counts(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["variable", "high", "low", "total"])

    counts = events.groupby(["variable", "direction"]).size().unstack(fill_value=0).reset_index()
    for direction in ["high", "low"]:
        if direction not in counts.columns:
            counts[direction] = 0

    counts["total"] = counts["high"] + counts["low"]
    return counts.sort_values("total", ascending=False)


def build_top_outlier_windows(events: pd.DataFrame, window: str = "5min", top_n: int = 20) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["window_start", "event_count", "variable_count"])

    windowed = events.copy()
    windowed["window_start"] = windowed["date"].dt.floor(window)
    return (
        windowed.groupby("window_start")
        .agg(event_count=("variable", "size"), variable_count=("variable", "nunique"))
        .reset_index()
        .sort_values("event_count", ascending=False)
        .head(top_n)
    )


def build_top_simultaneous_outliers(events: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(columns=["date", "variable_count", "variables"])

    simultaneous = (
        events.groupby("date")
        .agg(
            variable_count=("variable", "nunique"),
            variables=("variable", lambda values: "; ".join(sorted(set(values)))),
        )
        .reset_index()
    )
    return simultaneous[simultaneous["variable_count"] >= 2].sort_values("variable_count", ascending=False).head(top_n)


# ---------------------------------------------------------------------------
# Transform summaries
# ---------------------------------------------------------------------------
# Smoothing and differencing summaries reuse summarize_series, making the target
# transformation diagnostics comparable with the raw distribution analysis.

def summarize_target_transforms(
    transform_series: list[dict[str, Any]],
    target_column: str,
) -> pd.DataFrame:
    rows = []
    for item in transform_series:
        summary = summarize_series(
            id_key=str(item["id_key"]),
            granularity=str(item["granularity"]),
            variable=target_column,
            values=item["values"],
        )
        summary["target"] = target_column
        summary["transform"] = item["transform"]
        summary["window"] = item["window"]
        summary["window_steps"] = item["window_steps"]
        rows.append(summary)
    return pd.DataFrame(rows)
