from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, cast

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ARIMA

try:
    from exploration import build_forecast_metric_row
    from preprocessing import (
        build_forecasting_frame,
        infer_exogenous_columns,
        split_train_test,
        standardize_columns,
        statsforecast_frequency,
    )
except ImportError:  # pragma: no cover - package import path
    from .exploration import build_forecast_metric_row
    from .preprocessing import (
        build_forecasting_frame,
        infer_exogenous_columns,
        split_train_test,
        standardize_columns,
        statsforecast_frequency,
    )


def run_multivariate_analysis(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    forecasting_policy: dict[str, Any],
    analysis_policy: dict[str, Any],
    output_dir: Path,
    reports_dir: Path,
    plots_dir: Path,
    timestamp_column: str,
) -> dict[str, pd.DataFrame]:
    """Run optional ARIMAX comparisons with Candidate B non-target variables."""
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    return run_arimax_forecasts(
        datasets=datasets,
        target_column=target_column,
        target_granularities=analysis_policy.get("target_granularities", ["30s", "1min"]),
        granularity_options=granularity_options,
        resampling_policy=resampling_policy,
        forecasting_policy=forecasting_policy,
        analysis_policy=analysis_policy,
        output_dir=output_dir,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
        timestamp_column=timestamp_column,
    )


def run_arimax_forecasts(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    target_granularities: list[str],
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    forecasting_policy: dict[str, Any],
    analysis_policy: dict[str, Any],
    output_dir: Path,
    reports_dir: Path,
    plots_dir: Path,
    timestamp_column: str,
) -> dict[str, pd.DataFrame]:
    """Evaluate the configured single ARIMA order with exogenous regressors."""
    forecast_frames = []
    metric_rows = []
    skipped_rows = []
    arima_config = forecasting_policy.get("univariate", {}).get("arima", forecasting_policy.get("arima", {}))
    p_value, d_value, q_value = parse_arima_order(arima_config)
    train_fraction = protocol_train_test_boundary(forecasting_policy)
    max_train_rows = int(analysis_policy.get("max_train_rows", 12_000))
    write_full_forecasts = bool(forecasting_policy.get("write_full_forecasts", False))
    for granularity in target_granularities:
        id_key = f"subset_B_{granularity}"
        if id_key not in datasets:
            continue

        exogenous_columns = infer_exogenous_columns(datasets[id_key], target_column, timestamp_column)
        series_frame = build_forecasting_frame(
            datasets[id_key],
            id_key,
            target_column,
            timestamp_column,
            exogenous_columns=exogenous_columns,
        )
        train_frame, test_frame = split_train_test(series_frame, train_fraction)
        if len(train_frame) > max_train_rows:
            skipped_rows.append(
                {
                    "id_key": id_key,
                    "granularity": granularity,
                    "train_rows": len(train_frame),
                    "max_train_rows": max_train_rows,
                    "reason": "Skipped because statsforecast ARIMA with exogenous regressors performs a dense SVD that is not memory-safe at this row count.",
                }
            )
            continue

        if bool(analysis_policy.get("standardize_exogenous", True)):
            train_frame, test_frame = standardize_columns(train_frame, test_frame, exogenous_columns)
        frequency = statsforecast_frequency(granularity, granularity_options, resampling_policy)
        model_alias = f"ARIMAX_p{p_value}_d{d_value}_q{q_value}"
        forecast_frame = forecast_arimax(
            train_frame=train_frame,
            test_frame=test_frame,
            frequency=frequency,
            p_value=p_value,
            d_value=d_value,
            q_value=q_value,
            model_alias=model_alias,
            arima_config=arima_config,
            exogenous_columns=exogenous_columns,
        )
        forecast_frame = add_forecast_metadata(
            forecast_frame,
            id_key=id_key,
            granularity=granularity,
            model_alias=model_alias,
            p_value=p_value,
            d_value=d_value,
            q_value=q_value,
            train_rows=len(train_frame),
            test_rows=len(test_frame),
            exogenous_count=len(exogenous_columns),
        )
        forecast_frames.append(forecast_frame)
        metric_rows.append(build_forecast_metric_row(forecast_frame, exogenous_columns=exogenous_columns))

    forecasts = (
        pd.concat(forecast_frames, ignore_index=True)
        if forecast_frames
        else pd.DataFrame(columns=["ds", "y", "forecast"])
    )
    metrics = (
        pd.DataFrame(metric_rows).sort_values(["granularity", "d"])
        if metric_rows
        else pd.DataFrame(columns=["granularity", "model", "mae", "rmse"])
    )
    if write_full_forecasts:
        write_csv_output(forecasts, output_dir / "arimax_forecasts.csv")
    return {"forecasts": forecasts, "metrics": metrics, "skipped": pd.DataFrame(skipped_rows)}


def forecast_arimax(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    frequency: str,
    p_value: int,
    d_value: int,
    q_value: int,
    model_alias: str,
    arima_config: dict[str, Any],
    exogenous_columns: list[str],
) -> pd.DataFrame:
    """Fit one fixed-order ARIMAX using known test-period exogenous values."""
    seasonal_order = tuple(
        int(value) for value in cast(list[int] | tuple[int, int, int], arima_config.get("seasonal_order", [0, 0, 0]))
    )
    model = ARIMA(
        order=(p_value, d_value, q_value),
        season_length=int(arima_config.get("season_length", 1)),
        seasonal_order=seasonal_order,
        include_mean=d_value == 0,
        include_drift=d_value == 1,
        method=str(arima_config.get("method", "CSS-ML")),
        alias=model_alias,
    )
    forecast_engine = StatsForecast(models=[model], freq=frequency, n_jobs=1)
    future_exogenous = test_frame[["unique_id", "ds"] + exogenous_columns]
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        forecast = forecast_engine.forecast(df=train_frame, h=len(test_frame), X_df=future_exogenous)

    forecast = forecast[["ds", model_alias]].rename(columns={model_alias: "forecast"})
    evaluated = test_frame[["ds", "y"]].merge(forecast, on="ds", how="inner")
    if len(evaluated) != len(test_frame):
        raise ValueError(
            f"Forecast/test timestamp alignment failed for {model_alias}: "
            f"{len(evaluated)} aligned rows out of {len(test_frame)} test rows."
        )
    evaluated.attrs["warning_count"] = len(caught_warnings)
    evaluated.attrs["warnings"] = sorted({str(item.message) for item in caught_warnings})
    return evaluated


def parse_arima_order(arima_config: dict[str, Any]) -> tuple[int, int, int]:
    """Return the active ARIMA `(p, d, q)` order from forecasting config."""
    order = cast(list[int] | tuple[int, int, int], arima_config.get("order", [1, 0, 1]))
    if len(order) != 3:
        raise ValueError("arima.order must contain exactly three values: [p, d, q].")
    return int(order[0]), int(order[1]), int(order[2])


def protocol_train_test_boundary(forecasting_policy: dict[str, Any]) -> float:
    """Return the chronological train/test boundary used for holdout evaluation."""
    protocol = forecasting_policy.get("experimental_protocol", {})
    splits = protocol.get("splits", {})
    if splits:
        return float(splits["train"])
    return float(forecasting_policy.get("train_fraction", 0.8))


def add_forecast_metadata(
    forecast_frame: pd.DataFrame,
    id_key: str,
    granularity: str,
    model_alias: str,
    p_value: int,
    d_value: int,
    q_value: int,
    train_rows: int,
    test_rows: int,
    exogenous_count: int,
) -> pd.DataFrame:
    forecast_frame["id_key"] = id_key
    forecast_frame["granularity"] = granularity
    forecast_frame["model"] = model_alias
    forecast_frame["p"] = p_value
    forecast_frame["d"] = d_value
    forecast_frame["q"] = q_value
    forecast_frame["train_rows"] = train_rows
    forecast_frame["test_rows"] = test_rows
    forecast_frame["exogenous_count"] = exogenous_count
    return forecast_frame


def write_csv_output(frame: pd.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(output_path)
