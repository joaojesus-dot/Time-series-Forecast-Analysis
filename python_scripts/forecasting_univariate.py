from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import ARIMA

try:
    from exploration import build_forecast_metric_row, build_mlp_window_summary
    from plots import write_forecast_metrics_plot, write_forecast_plot
    from preprocessing import (
        build_forecasting_frame,
        build_scaled_target_frame,
        split_train_test,
        statsforecast_frequency,
    )
    from reports import write_arima_report, write_univariate_forecasting_report
except ImportError:  # pragma: no cover - package import path
    from .exploration import build_forecast_metric_row, build_mlp_window_summary
    from .plots import write_forecast_metrics_plot, write_forecast_plot
    from .preprocessing import (
        build_forecasting_frame,
        build_scaled_target_frame,
        split_train_test,
        statsforecast_frequency,
    )
    from .reports import write_arima_report, write_univariate_forecasting_report


def run_univariate_analysis(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    forecasting_policy: dict[str, Any],
    analysis_policy: dict[str, Any],
    output_dir: Path,
    reports_dir: Path,
    plots_dir: Path,
    derived_data_dir: Path,
    timestamp_column: str,
) -> dict[str, pd.DataFrame]:
    """Run raw-target univariate forecasting artifacts for ARIMA and MLP prep."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    derived_data_dir.mkdir(parents=True, exist_ok=True)

    target_granularities = [str(analysis_policy.get("target_granularity", "raw"))]
    arima_results = run_univariate_arima_forecasts(
        datasets=datasets,
        target_column=target_column,
        target_granularities=target_granularities,
        granularity_options=granularity_options,
        resampling_policy=resampling_policy,
        forecasting_policy=forecasting_policy,
        output_dir=output_dir,
        reports_dir=reports_dir,
        plots_dir=plots_dir,
        timestamp_column=timestamp_column,
    )
    scaled_outputs = build_univariate_scaled_target_outputs(
        datasets=datasets,
        target_column=target_column,
        granularity="raw",
        train_fraction=float(forecasting_policy["train_fraction"]),
        data_dir=derived_data_dir,
        analysis_policy=analysis_policy,
        timestamp_column=timestamp_column,
    )
    write_univariate_forecasting_report(
        output_path=reports_dir / "univariate_forecasting.md",
        target_column=target_column,
        arima_metrics=arima_results["metrics"],
        scaling_summary=scaled_outputs["scaling_summary"],
        mlp_summary=scaled_outputs["mlp_summary"],
    )
    return {**arima_results, **scaled_outputs}


def run_univariate_arima_forecasts(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    target_granularities: list[str],
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    forecasting_policy: dict[str, Any],
    output_dir: Path,
    reports_dir: Path,
    plots_dir: Path,
    timestamp_column: str,
) -> dict[str, pd.DataFrame]:
    """Evaluate the configured single ARIMA order on selected target granularities."""
    forecast_frames = []
    metric_rows = []
    arima_config = forecasting_policy["arima"]
    p_value, d_value, q_value = parse_arima_order(arima_config)
    write_full_forecasts = bool(forecasting_policy.get("write_full_forecasts", False))
    write_forecast_plots = bool(forecasting_policy.get("write_forecast_plots", True))

    for granularity in target_granularities:
        id_key = f"subset_B_{granularity}"
        if id_key not in datasets:
            continue

        series_frame = build_forecasting_frame(datasets[id_key], id_key, target_column, timestamp_column)
        train_frame, test_frame = split_train_test(series_frame, float(forecasting_policy["train_fraction"]))
        frequency = statsforecast_frequency(granularity, granularity_options, resampling_policy)
        granularity_forecasts = []

        model_alias = f"ARIMA_p{p_value}_d{d_value}_q{q_value}"
        forecast_frame = forecast_arima(
            train_frame=train_frame,
            test_frame=test_frame,
            frequency=frequency,
            p_value=p_value,
            d_value=d_value,
            q_value=q_value,
            model_alias=model_alias,
            arima_config=arima_config,
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
        )
        forecast_frames.append(forecast_frame)
        granularity_forecasts.append(forecast_frame)
        metric_rows.append(build_forecast_metric_row(forecast_frame))

        if write_forecast_plots:
            write_forecast_plot(id_key, granularity, train_frame, test_frame, granularity_forecasts, plots_dir, "ARIMA")

    forecasts = pd.concat(forecast_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["granularity", "d"])
    if write_full_forecasts:
        write_csv_output(forecasts, output_dir / "arima_forecasts.csv")
    write_forecast_metrics_plot(metrics, plots_dir / "arima_metric_comparison.png")
    write_arima_report(metrics, reports_dir / "arima_forecasting.md", (p_value, d_value, q_value), target_granularities)
    return {"forecasts": forecasts, "metrics": metrics}


def build_univariate_scaled_target_outputs(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    granularity: str,
    train_fraction: float,
    data_dir: Path,
    analysis_policy: dict[str, Any],
    timestamp_column: str,
) -> dict[str, pd.DataFrame]:
    """Write train-fitted z-score target artifacts for the future MLP baseline."""
    id_key = f"subset_B_{granularity}"
    if id_key not in datasets:
        raise KeyError(f"Missing dataset '{id_key}' for univariate scaling.")

    frame = build_forecasting_frame(datasets[id_key], id_key, target_column, timestamp_column)
    scaled_frame, scaling_summary = build_scaled_target_frame(frame, train_fraction, granularity, target_column)

    mlp_config = analysis_policy.get("mlp", {})
    mlp_summary = build_mlp_window_summary(
        scaled_frame,
        granularity=granularity,
        lookback_steps=int(mlp_config.get("lookback_steps", 60)),
        horizon_steps=int(mlp_config.get("horizon_steps", 60)),
        input_frequency=str(mlp_config.get("input_frequency", "5s")),
    )

    write_csv_output(scaled_frame[["unique_id", "ds", "split", "y", "y_scaled"]], data_dir / "raw_target_scaled.csv")
    return {
        "scaled_target": scaled_frame,
        "scaling_summary": scaling_summary,
        "mlp_summary": mlp_summary,
    }


def forecast_arima(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    frequency: str,
    p_value: int,
    d_value: int,
    q_value: int,
    model_alias: str,
    arima_config: dict[str, Any],
) -> pd.DataFrame:
    """Fit one fixed-order univariate ARIMA and align predictions to the holdout timestamps."""
    model = ARIMA(
        order=(p_value, d_value, q_value),
        season_length=int(arima_config.get("season_length", 1)),
        seasonal_order=tuple(int(value) for value in arima_config.get("seasonal_order", [0, 0, 0])),
        include_mean=d_value == 0,
        include_drift=d_value == 1,
        method=str(arima_config.get("method", "CSS-ML")),
        alias=model_alias,
    )
    forecast_engine = StatsForecast(models=[model], freq=frequency, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        forecast = forecast_engine.forecast(df=train_frame, h=len(test_frame))

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
    order = arima_config.get("order", [1, 0, 1])
    if len(order) != 3:
        raise ValueError("arima.order must contain exactly three values: [p, d, q].")
    return tuple(int(value) for value in order)


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
) -> pd.DataFrame:
    forecast_frame["id_key"] = id_key
    forecast_frame["granularity"] = granularity
    forecast_frame["model"] = model_alias
    forecast_frame["p"] = p_value
    forecast_frame["d"] = d_value
    forecast_frame["q"] = q_value
    forecast_frame["train_rows"] = train_rows
    forecast_frame["test_rows"] = test_rows
    return forecast_frame


def write_csv_output(frame: pd.DataFrame, output_path: Path) -> None:
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(output_path)
