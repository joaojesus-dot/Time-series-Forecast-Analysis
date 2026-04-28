from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tsa.stattools import acf, adfuller, kpss


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
    total_sum_squares = np.square(actual - actual.mean()).sum()
    residual_sum_squares = np.square(errors).sum()
    r2 = np.nan if total_sum_squares == 0 else 1 - residual_sum_squares / total_sum_squares
    return {
        "mae": float(abs_errors.mean()),
        "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        "mape": float((abs_errors / nonzero_actual.abs()).mean() * 100),
        "smape": float((2 * abs_errors / denominator).mean() * 100),
        "bias": float(errors.mean()),
        "r2": float(r2),
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
    lookback_duration: str,
    horizon_duration: str,
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
                "status": "windows_prepared",
                "feature_source": "scaled target lags only",
                "target_source": "scaled future target",
                "lookback_steps": lookback_steps,
                "horizon_steps": horizon_steps,
                "lookback_duration": lookback_duration,
                "horizon_duration": horizon_duration,
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


# ---------------------------------------------------------------------------
# Target profiling
# ---------------------------------------------------------------------------
# The assignment asks for evidence about the target's granularity,
# autocorrelation, and stationarity. These helpers keep that analysis explicit
# and reproducible instead of leaving it only as narrative text.

def build_target_autocorrelation_summary(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    granularity_options: dict[str, str | None],
    input_frequency: str,
    lag_durations: list[str] | None = None,
) -> pd.DataFrame:
    """Measure the target ACF at the requested real-time lags for each dataset."""
    lag_durations = lag_durations or ["30s", "1min", "5min", "10min"]
    rows: list[dict[str, Any]] = []

    for id_key, dataset in datasets.items():
        granularity = id_key.rsplit("_", 1)[-1]
        frequency = granularity_options.get(granularity) or input_frequency
        series = dataset[target_column].dropna().reset_index(drop=True)
        if len(series) < 3:
            continue

        lag_steps: dict[str, int] = {}
        for duration in lag_durations:
            try:
                step = max(1, int(round(pd.Timedelta(duration) / pd.Timedelta(str(frequency)))))
            except ValueError:
                continue
            if step < len(series):
                lag_steps[duration] = step

        max_lag = max(lag_steps.values(), default=1)
        acf_values = acf(series, nlags=max_lag, fft=True, missing="drop")
        row: dict[str, Any] = {
            "id_key": id_key,
            "granularity": granularity,
            "frequency": str(frequency),
            "rows": len(series),
        }
        for duration, step in lag_steps.items():
            row[f"acf_{duration}"] = float(acf_values[step])
        rows.append(row)

    return pd.DataFrame(rows)


def build_stationarity_summary(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for id_key, dataset in datasets.items():
        granularity = id_key.rsplit("_", 1)[-1]
        base_series = dataset[target_column].dropna().reset_index(drop=True)
        transforms = {
            "original": base_series,
            "first_difference": base_series.diff().dropna().reset_index(drop=True),
            "second_difference": base_series.diff().diff().dropna().reset_index(drop=True),
        }
        for transform_name, series in transforms.items():
            if len(series) < 20:
                continue
            adf_result = safe_adfuller(series)
            kpss_result = safe_kpss(series)
            adf_pvalue = float(adf_result.get("adf_pvalue", np.nan))
            kpss_pvalue = float(kpss_result.get("kpss_pvalue", np.nan))
            rows.append(
                {
                    "id_key": id_key,
                    "granularity": granularity,
                    "transform": transform_name,
                    "rows": len(series),
                    **adf_result,
                    **kpss_result,
                    "stationary_recommended": bool(
                        adf_pvalue <= 0.05 and kpss_pvalue >= 0.05
                    ),
                }
            )
    return pd.DataFrame(rows)


def build_target_correlation_summary(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    timestamp_column: str = "date",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for id_key, dataset in datasets.items():
        granularity = id_key.rsplit("_", 1)[-1]
        numeric_frame = dataset.drop(columns=[timestamp_column], errors="ignore").select_dtypes(include=["number"])
        if target_column not in numeric_frame.columns:
            continue
        target_correlations = numeric_frame.corr(numeric_only=True)[target_column].drop(labels=[target_column])
        sorted_correlations = target_correlations.sort_values(
            key=lambda values: values.abs(),
            ascending=False,
        )
        for variable, correlation in sorted_correlations.items():
            rows.append(
                {
                    "id_key": id_key,
                    "granularity": granularity,
                    "target_column": target_column,
                    "variable": variable,
                    "correlation": float(correlation),
                    "abs_correlation": float(abs(correlation)),
                }
            )
    return pd.DataFrame(rows)


def safe_adfuller(series: pd.Series) -> dict[str, Any]:
    try:
        result = adfuller(series, autolag="AIC")
        return {
            "adf_statistic": float(result[0]),
            "adf_pvalue": float(result[1]),
            "adf_used_lag": int(result[2]),
            "adf_nobs": int(result[3]),
        }
    except Exception as error:  # pragma: no cover - defensive summary path
        return {
            "adf_statistic": np.nan,
            "adf_pvalue": np.nan,
            "adf_used_lag": -1,
            "adf_nobs": 0,
            "adf_error": str(error),
        }


def safe_kpss(series: pd.Series) -> dict[str, Any]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", InterpolationWarning)
            result = kpss(series, regression="c", nlags="auto")
        return {
            "kpss_statistic": float(result[0]),
            "kpss_pvalue": float(result[1]),
            "kpss_used_lag": int(result[2]),
        }
    except Exception as error:  # pragma: no cover - defensive summary path
        return {
            "kpss_statistic": np.nan,
            "kpss_pvalue": np.nan,
            "kpss_used_lag": -1,
            "kpss_error": str(error),
        }
