from __future__ import annotations

from typing import Any

import numpy as np
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
# These summaries are descriptive only. Outlier treatment is intentionally not
# part of the active pipeline until we choose a standard-deviation/z-score rule.

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
        "skewness": clean_values.skew(),
    }


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


# ---------------------------------------------------------------------------
# Forecast evaluation summaries
# ---------------------------------------------------------------------------
# Forecasting modules own model fitting, but metric calculation and window-count
# summaries are analysis outputs, so they live with the rest of exploration.

def forecast_metrics(actual: pd.Series, forecast: pd.Series) -> dict[str, float]:
    errors = forecast - actual
    abs_errors = errors.abs()
    nonzero_actual = actual.replace(0, np.nan)
    denominator = (actual.abs() + forecast.abs()).replace(0, np.nan)
    return {
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float((abs_errors / nonzero_actual.abs()).mean() * 100),
        "smape": float((2 * abs_errors / denominator).mean() * 100),
        "bias": float(errors.mean()),
    }


def build_forecast_metric_row(
    forecast_frame: pd.DataFrame,
    exogenous_columns: list[str] | None = None,
) -> dict[str, Any]:
    row = {
        "id_key": forecast_frame["id_key"].iloc[0],
        "granularity": forecast_frame["granularity"].iloc[0],
        "model": forecast_frame["model"].iloc[0],
        "p": int(forecast_frame["p"].iloc[0]),
        "d": int(forecast_frame["d"].iloc[0]),
        "q": int(forecast_frame["q"].iloc[0]),
        "train_rows": int(forecast_frame["train_rows"].iloc[0]),
        "test_rows": int(forecast_frame["test_rows"].iloc[0]),
        "warning_count": int(forecast_frame.attrs.get("warning_count", 0)),
        "warnings": "; ".join(forecast_frame.attrs.get("warnings", [])),
        **forecast_metrics(forecast_frame["y"], forecast_frame["forecast"]),
    }
    if exogenous_columns is not None:
        row["exogenous_count"] = len(exogenous_columns)
        row["exogenous_columns"] = "; ".join(exogenous_columns)
    return row


def build_mlp_window_summary(
    scaled_frame: pd.DataFrame,
    granularity: str,
    lookback_steps: int,
    horizon_steps: int,
    input_frequency: str,
) -> pd.DataFrame:
    train_rows = int((scaled_frame["split"] == "train").sum())
    test_rows = int((scaled_frame["split"] == "test").sum())
    train_windows = max(0, train_rows - lookback_steps - horizon_steps + 1)
    test_windows = max(0, test_rows - lookback_steps - horizon_steps + 1)
    return pd.DataFrame(
        [
            {
                "granularity": granularity,
                "input_frequency": input_frequency,
                "model_family": "MLP",
                "status": "prepared_not_trained",
                "feature_source": "scaled target lags only",
                "target_source": "scaled future target",
                "lookback_steps": lookback_steps,
                "horizon_steps": horizon_steps,
                "lookback_duration": describe_duration(lookback_steps, input_frequency),
                "horizon_duration": describe_duration(horizon_steps, input_frequency),
                "train_rows": train_rows,
                "test_rows": test_rows,
                "train_windows": train_windows,
                "test_windows": test_windows,
                "write_window_data": False,
            }
        ]
    )


def describe_duration(steps: int, frequency: str) -> str:
    try:
        seconds = pd.to_timedelta(frequency).total_seconds() * steps
    except ValueError:
        return f"{steps} x {frequency}"
    return str(pd.to_timedelta(seconds, unit="s"))
