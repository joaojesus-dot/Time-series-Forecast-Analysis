from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP as NeuralForecastMLP
from statsforecast import StatsForecast
from statsforecast.models import ARIMA

try:
    from exploration import build_forecast_metric_row, forecast_metrics
    from plots import write_test_comparison_metric_plots, write_test_comparison_plots, write_univariate_comparison_plots
    from preprocessing import (
        build_forecasting_frame,
        build_scaled_target_frame,
        split_train_test,
        statsforecast_frequency,
        steps_for_duration,
    )
except ImportError:  # pragma: no cover - package import path
    from .exploration import build_forecast_metric_row, forecast_metrics
    from .plots import write_test_comparison_metric_plots, write_test_comparison_plots, write_univariate_comparison_plots
    from .preprocessing import (
        build_forecasting_frame,
        build_scaled_target_frame,
        split_train_test,
        statsforecast_frequency,
        steps_for_duration,
    )


class WindowBundle(TypedDict):
    x: pd.DataFrame
    y_model: pd.Series
    target_dates: pd.Series
    y_actual: pd.Series
    prev_y_1: pd.Series
    prev_y_2: pd.Series


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
    """Run ARIMA and MLP evaluations for the configured univariate datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    derived_data_dir.mkdir(parents=True, exist_ok=True)

    protocol = forecasting_policy["experimental_protocol"]
    target_granularities = list(protocol["target_granularities"])

    arima_config = analysis_policy.get("arima", {})
    if bool(arima_config.get("enabled", False)):
        arima_results = run_univariate_arima_forecasts(
            datasets=datasets,
            target_column=target_column,
            target_granularities=target_granularities,
            granularity_options=granularity_options,
            resampling_policy=resampling_policy,
            forecasting_policy=forecasting_policy,
            protocol=protocol,
            output_dir=output_dir,
            reports_dir=reports_dir,
            plots_dir=plots_dir,
            timestamp_column=timestamp_column,
        )
    else:
        arima_results = {
            "forecasts": pd.DataFrame(columns=["ds", "y", "forecast", "granularity"]),
            "metrics": pd.DataFrame(columns=["granularity", "model", "mae", "rmse", "r2"]),
        }

    candidate_specs, scaling_summary, mlp_window_summary = build_mlp_preparation_candidates(
        datasets=datasets,
        target_column=target_column,
        target_granularities=target_granularities,
        granularity_options=granularity_options,
        resampling_policy=resampling_policy,
        protocol=protocol,
        timestamp_column=timestamp_column,
    )

    mlp_results = run_mlp_test_comparison(
        candidate_specs=candidate_specs,
        scaling_summary=scaling_summary,
        analysis_policy=analysis_policy,
        output_dir=output_dir,
    )

    write_test_comparison_metric_plots(
        mlp_results["mlp_test_comparison"],
        plots_dir / "test_comparison_metrics",
    )
    write_test_comparison_plots(
        mlp_results["mlp_forecasts"],
        plots_dir / "test_comparison",
    )
    write_univariate_comparison_plots(
        arima_metrics=arima_results["metrics"],
        mlp_metrics=mlp_results["mlp_metrics"],
        arima_forecasts=arima_results["forecasts"],
        mlp_forecasts=mlp_results["mlp_forecasts"],
        output_dir=plots_dir,
    )
    return {
        **arima_results,
        "scaling_summary": scaling_summary,
        "mlp_window_summary": mlp_window_summary,
        **mlp_results,
    }


def run_univariate_arima_forecasts(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    target_granularities: list[str],
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    forecasting_policy: dict[str, Any],
    protocol: dict[str, Any],
    output_dir: Path,
    reports_dir: Path,
    plots_dir: Path,
    timestamp_column: str,
) -> dict[str, pd.DataFrame]:
    """Fit the configured ARIMA order on each univariate modeling granularity."""
    del reports_dir, plots_dir
    forecast_frames = []
    metric_rows = []
    arima_config = forecasting_policy.get("univariate", {}).get("arima", forecasting_policy.get("arima", {}))
    p_value, d_value, q_value = parse_arima_order(arima_config)
    train_fraction = protocol_train_test_boundary(protocol)
    write_full_forecasts = bool(forecasting_policy.get("write_full_forecasts", False))

    for granularity in target_granularities:
        id_key = f"{protocol['dataset_prefix']}_{granularity}"
        if id_key not in datasets:
            continue

        series_frame = build_forecasting_frame(datasets[id_key], id_key, target_column, timestamp_column)
        train_frame, test_frame = split_train_test(series_frame, train_fraction)
        frequency = statsforecast_frequency(granularity, granularity_options, resampling_policy)
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
        metric_rows.append(build_forecast_metric_row(forecast_frame))

    if not forecast_frames:
        return {
            "forecasts": pd.DataFrame(columns=["ds", "y", "forecast"]),
            "metrics": pd.DataFrame(columns=["granularity", "model", "mae", "rmse"]),
        }

    forecasts = pd.concat(forecast_frames, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values(["granularity", "d"])
    if write_full_forecasts:
        write_csv_output(forecasts, output_dir / "arima_forecasts.csv")
    return {"forecasts": forecasts, "metrics": metrics}


def build_mlp_preparation_candidates(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    target_granularities: list[str],
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    protocol: dict[str, Any],
    timestamp_column: str,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Build every configured MLP preparation candidate and its window counts."""
    candidate_specs: list[dict[str, Any]] = []
    scaling_summaries = []
    window_rows = []

    splits = protocol["splits"]
    train_fraction = float(splits["train"])
    horizon_duration = str(protocol["forecast_horizon"])
    lookback_duration = str(protocol["lookback_window"])
    difference_orders = list(protocol.get("differentiation_orders", [0, 1, 2]))
    scaling_method = str(protocol.get("scaling", {}).get("method", "standard"))

    for granularity in target_granularities:
        id_key = f"{protocol['dataset_prefix']}_{granularity}"
        if id_key not in datasets:
            continue

        frequency = statsforecast_frequency(granularity, granularity_options, resampling_policy)
        lookback_steps = steps_for_duration(lookback_duration, frequency)
        horizon_steps = steps_for_duration(horizon_duration, frequency)
        frame = build_forecasting_frame(datasets[id_key], id_key, target_column, timestamp_column)
        scaled_frame, scaling_summary = build_scaled_target_frame(
            frame,
            train_fraction,
            granularity,
            target_column,
            frequency,
            scaling_method=scaling_method,
        )
        scaling_summaries.append(scaling_summary)
        scaler_row = scaling_summary.iloc[0].to_dict()

        for difference_order in difference_orders:
            transformed_frame = build_transformed_mlp_frame(scaled_frame, difference_order)
            for smoothing_window in smoothing_windows_for_granularity(protocol, granularity):
                if smoothing_window != "none":
                    try:
                        steps_for_duration(smoothing_window, frequency)
                    except ValueError:
                        continue
                prepared_frame = apply_train_only_smoothing(transformed_frame, smoothing_window, frequency)
                candidate_label = build_candidate_label(granularity, difference_order, smoothing_window)
                train_windows = build_prepared_windows(prepared_frame, "train", lookback_steps, horizon_steps)
                test_windows = build_prepared_windows(prepared_frame, "test", lookback_steps, horizon_steps)
                candidate_specs.append(
                    {
                        "candidate_label": candidate_label,
                        "granularity": granularity,
                        "frequency": frequency,
                        "difference_order": difference_order,
                        "transform_name": transform_name_for_order(difference_order),
                        "training_smoothing_window": smoothing_window,
                        "lookback_steps": lookback_steps,
                        "horizon_steps": horizon_steps,
                        "lookback_duration": lookback_duration,
                        "horizon_duration": horizon_duration,
                        "prepared_frame": prepared_frame,
                        "scaler_row": scaler_row,
                        "train_windows": train_windows,
                        "test_windows": test_windows,
                    }
                )
                window_rows.append(
                    {
                        "granularity": granularity,
                        "candidate_label": candidate_label,
                        "status": "windows_prepared",
                        "difference_order": difference_order,
                        "transform_name": transform_name_for_order(difference_order),
                        "training_smoothing_window": smoothing_window,
                        "lookback_duration": lookback_duration,
                        "lookback_steps": lookback_steps,
                        "horizon_duration": horizon_duration,
                        "horizon_steps": horizon_steps,
                        "train_windows": len(train_windows["x"]),
                        "test_windows": len(test_windows["x"]),
                    }
                )

    scaling_summary_frame = pd.concat(scaling_summaries, ignore_index=True) if scaling_summaries else pd.DataFrame()
    mlp_window_summary = pd.DataFrame(window_rows)
    return candidate_specs, scaling_summary_frame, mlp_window_summary


def run_mlp_test_comparison(
    candidate_specs: list[dict[str, Any]],
    scaling_summary: pd.DataFrame,
    analysis_policy: dict[str, Any],
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Forecast the chronological test split for every configured MLP preparation candidate."""
    del scaling_summary
    mlp_config = analysis_policy.get("mlp", {})
    if not bool(mlp_config.get("enabled", False)) or not candidate_specs:
        return {
            "mlp_metrics": pd.DataFrame(),
            "mlp_forecasts": pd.DataFrame(),
            "mlp_test_comparison": pd.DataFrame(),
            "mlp_parameter_effects": pd.DataFrame(),
        }

    metric_rows = []
    test_forecasts = []
    learning_rates = mlp_config.get("learning_rate_grid", mlp_config.get("initial_learning_rate_grid", [0.001]))

    for candidate in candidate_specs:
        train_windows = candidate["train_windows"]
        test_windows = candidate["test_windows"]
        if train_windows["x"].empty or test_windows["x"].empty:
            continue

        hidden_units = select_hidden_units(candidate["lookback_steps"], mlp_config)
        for learning_rate_init in learning_rates:
            test_forecast, warning_messages = forecast_neural_mlp_candidate(
                candidate=candidate,
                hidden_units=hidden_units,
                learning_rate_init=float(learning_rate_init),
                mlp_config=mlp_config,
            )
            if test_forecast.empty:
                continue
            test_metrics = forecast_metrics(test_forecast["y"], test_forecast["forecast"])
            test_forecast["model"] = "NeuralForecast_MLP_target_lags"
            test_forecast["granularity"] = candidate["granularity"]
            test_forecast["split"] = "test"
            test_forecast["candidate_label"] = candidate["candidate_label"]
            test_forecast["difference_order"] = candidate["difference_order"]
            test_forecast["transform_name"] = candidate["transform_name"]
            test_forecast["training_smoothing_window"] = candidate["training_smoothing_window"]
            test_forecast["learning_rate_init"] = float(learning_rate_init)
            test_forecasts.append(test_forecast)
            metric_rows.append(
                {
                    "model": "NeuralForecast_MLP_target_lags",
                    "split": "test",
                    "candidate_label": candidate["candidate_label"],
                    "granularity": candidate["granularity"],
                    "difference_order": candidate["difference_order"],
                    "transform_name": candidate["transform_name"],
                    "training_smoothing_window": candidate["training_smoothing_window"],
                    "lookback_steps": candidate["lookback_steps"],
                    "horizon_steps": candidate["horizon_steps"],
                    "lookback_duration": candidate["lookback_duration"],
                    "horizon_duration": candidate["horizon_duration"],
                    "hidden_units": hidden_units,
                    "learning_rate_init": float(learning_rate_init),
                    "engine": str(mlp_config.get("engine", "neuralforecast")),
                    "optimizer": str(mlp_config.get("optimizer", "adam")),
                    "num_layers": int(mlp_config.get("num_layers", 1)),
                    "min_steps": int(mlp_config.get("min_steps", 5000)),
                    "max_steps": int(mlp_config.get("max_steps", 7500)),
                    "accelerator": str(mlp_config.get("accelerator", "auto")),
                    "warning_count": len(warning_messages),
                    "warnings": "; ".join(warning_messages),
                    **test_metrics,
                }
            )

    test_metrics_frame = pd.DataFrame(metric_rows)
    if test_metrics_frame.empty:
        return {
            "mlp_metrics": pd.DataFrame(),
            "mlp_forecasts": pd.DataFrame(),
            "mlp_test_comparison": pd.DataFrame(),
            "mlp_parameter_effects": pd.DataFrame(),
        }

    test_metrics_frame = test_metrics_frame.sort_values(
        ["granularity", "difference_order", "training_smoothing_window", "learning_rate_init"]
    ).reset_index(drop=True)
    test_metrics_frame["selection_mode"] = str(mlp_config.get("selection_mode", "test_comparison"))

    return {
        "mlp_metrics": test_metrics_frame,
        "mlp_forecasts": pd.concat(test_forecasts, ignore_index=True),
        "mlp_test_comparison": test_metrics_frame.copy(),
        "mlp_parameter_effects": test_metrics_frame.copy(),
    }


def build_transformed_mlp_frame(scaled_frame: pd.DataFrame, difference_order: int) -> pd.DataFrame:
    transformed = scaled_frame.copy()
    transformed["model_series"] = np.nan
    for split in ["train", "test"]:
        mask = transformed["split"] == split
        series = transformed.loc[mask, "y_scaled"].reset_index(drop=True)
        if difference_order == 0:
            transformed_series = series
        elif difference_order == 1:
            transformed_series = series.diff()
        elif difference_order == 2:
            transformed_series = series.diff().diff()
        else:
            raise ValueError(f"Unsupported difference order '{difference_order}'.")
        transformed.loc[mask, "model_series"] = transformed_series.to_numpy()
    return transformed


def apply_train_only_smoothing(
    transformed_frame: pd.DataFrame,
    smoothing_window: str,
    frequency: str,
) -> pd.DataFrame:
    """Apply the configured smoothing window to the train split model series only."""
    prepared = transformed_frame.copy()
    if smoothing_window == "none":
        return prepared

    window_steps = steps_for_duration(smoothing_window, frequency)
    train_mask = prepared["split"] == "train"
    train_series = prepared.loc[train_mask, "model_series"].reset_index(drop=True)
    prepared.loc[train_mask, "model_series"] = (
        train_series.rolling(window=window_steps, min_periods=window_steps).mean().to_numpy()
    )
    return prepared


def build_prepared_windows(
    prepared_frame: pd.DataFrame,
    split: str,
    lookback_steps: int,
    horizon_steps: int,
) -> WindowBundle:
    """Convert one prepared split into lag windows and aligned forecast targets."""
    split_frame = prepared_frame[prepared_frame["split"] == split].reset_index(drop=True)
    row_count = len(split_frame) - lookback_steps - horizon_steps + 1
    if row_count <= 0:
        return {
            "x": pd.DataFrame(),
            "y_model": pd.Series(dtype=float),
            "target_dates": pd.Series(dtype="datetime64[ns]"),
            "y_actual": pd.Series(dtype=float),
            "prev_y_1": pd.Series(dtype=float),
            "prev_y_2": pd.Series(dtype=float),
        }

    x_rows = []
    y_model_rows = []
    target_dates = []
    y_actual_rows = []
    prev_y_1 = []
    prev_y_2 = []

    series = split_frame["model_series"].reset_index(drop=True)
    for start in range(row_count):
        lag_slice = series.iloc[start : start + lookback_steps]
        target_position = start + lookback_steps + horizon_steps - 1
        target_value = series.iloc[target_position]
        if lag_slice.isna().any() or pd.isna(target_value):
            continue
        if target_position - 1 < 0:
            continue
        x_rows.append(lag_slice.to_list())
        y_model_rows.append(float(target_value))
        target_dates.append(split_frame.at[target_position, "ds"])
        y_actual_rows.append(_frame_float(split_frame, target_position, "y"))
        prev_y_1.append(_frame_float(split_frame, target_position - 1, "y"))
        prev_y_2.append(_frame_float(split_frame, target_position - 2, "y") if target_position - 2 >= 0 else np.nan)

    feature_names = [f"lag_{lookback_steps - index}" for index in range(lookback_steps)]
    return {
        "x": pd.DataFrame(x_rows, columns=feature_names),
        "y_model": pd.Series(y_model_rows),
        "target_dates": pd.Series(target_dates),
        "y_actual": pd.Series(y_actual_rows),
        "prev_y_1": pd.Series(prev_y_1),
        "prev_y_2": pd.Series(prev_y_2),
    }


def forecast_neural_mlp_candidate(
    candidate: dict[str, Any],
    hidden_units: int,
    learning_rate_init: float,
    mlp_config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str]]:
    """Fit one NeuralForecast MLP on train data and score the chronological test window."""
    prepared = candidate["prepared_frame"].dropna(subset=["model_series"]).copy()
    if prepared.empty:
        return pd.DataFrame(columns=["ds", "y", "forecast"]), []

    test_size = int((prepared["split"] == "test").sum())
    if test_size <= 0:
        return pd.DataFrame(columns=["ds", "y", "forecast"]), []

    model_alias = f"MLP_lr_{format_learning_rate_alias(learning_rate_init)}"
    forecast_frame = prepared[["unique_id", "ds", "model_series"]].rename(columns={"model_series": "y"})
    neural_model = NeuralForecastMLP(
        h=int(candidate["horizon_steps"]),
        input_size=int(candidate["lookback_steps"]),
        num_layers=int(mlp_config.get("num_layers", 1)),
        hidden_size=int(hidden_units),
        max_steps=int(mlp_config.get("max_steps", 7500)),
        learning_rate=float(learning_rate_init),
        batch_size=int(mlp_config.get("batch_size", 1)),
        windows_batch_size=int(mlp_config.get("windows_batch_size", 1024)),
        inference_windows_batch_size=int(mlp_config.get("inference_windows_batch_size", -1)),
        scaler_type=str(mlp_config.get("scaler_type", "identity")),
        random_seed=int(mlp_config.get("random_state", 42)),
        alias=model_alias,
        min_steps=int(mlp_config.get("min_steps", 5000)),
        accelerator=str(mlp_config.get("accelerator", "auto")),
        devices=int(mlp_config.get("devices", 1)),
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
    )
    engine = NeuralForecast(models=[neural_model], freq=str(candidate["frequency"]))
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        cross_validation = engine.cross_validation(
            df=forecast_frame,
            n_windows=None,
            val_size=0,
            test_size=test_size,
            step_size=1,
            refit=False,
            verbose=False,
        )
    warnings_seen = sorted({str(item.message) for item in caught_warnings})
    return build_neural_mlp_test_forecast(
        cross_validation=cross_validation,
        model_alias=model_alias,
        candidate=candidate,
    ), warnings_seen


def build_neural_mlp_test_forecast(
    cross_validation: pd.DataFrame,
    model_alias: str,
    candidate: dict[str, Any],
) -> pd.DataFrame:
    test_windows = candidate["test_windows"]
    target_dates = test_windows["target_dates"]
    if not isinstance(target_dates, pd.Series):
        raise TypeError("Prepared windows must store target dates in a pandas Series.")

    selected = cross_validation[cross_validation["ds"].isin(set(target_dates))].sort_values("ds").copy()
    if selected.empty:
        return pd.DataFrame(columns=["ds", "y", "forecast"])

    split_windows = build_inverse_windows_for_dates(candidate["prepared_frame"], selected["ds"])
    predicted_model_values = selected[model_alias].reset_index(drop=True)
    forecast = inverse_prepared_predictions(
        predicted_model_values,
        split_windows,
        candidate["scaler_row"],
        int(candidate["difference_order"]),
    )
    target_dates = split_windows["target_dates"]
    actual_values = split_windows["y_actual"]
    if not isinstance(target_dates, pd.Series) or not isinstance(actual_values, pd.Series):
        raise TypeError("Prepared windows must store target dates and actual values in pandas Series objects.")
    return pd.DataFrame(
        {
            "ds": selected["ds"].reset_index(drop=True),
            "y": actual_values.reset_index(drop=True),
            "forecast": forecast.reset_index(drop=True),
        }
    )


def build_inverse_windows_for_dates(prepared_frame: pd.DataFrame, target_dates: pd.Series) -> WindowBundle:
    split_frame = prepared_frame[prepared_frame["split"] == "test"].reset_index(drop=True)
    date_to_position = {value: index for index, value in enumerate(split_frame["ds"])}
    y_actual = []
    prev_y_1 = []
    prev_y_2 = []

    for target_date in target_dates.reset_index(drop=True):
        position = date_to_position[target_date]
        y_actual.append(_frame_float(split_frame, position, "y"))
        prev_y_1.append(_frame_float(split_frame, position - 1, "y") if position - 1 >= 0 else np.nan)
        prev_y_2.append(_frame_float(split_frame, position - 2, "y") if position - 2 >= 0 else np.nan)

    return {
        "x": pd.DataFrame(),
        "y_model": pd.Series(dtype=float),
        "target_dates": target_dates.reset_index(drop=True),
        "y_actual": pd.Series(y_actual),
        "prev_y_1": pd.Series(prev_y_1),
        "prev_y_2": pd.Series(prev_y_2),
    }


def inverse_prepared_predictions(
    predicted_model_values: pd.Series,
    split_windows: WindowBundle,
    scaler_row: dict[str, Any],
    difference_order: int,
) -> pd.Series:
    scale = float(scaler_row["train_scale"])
    previous_y_1 = split_windows["prev_y_1"]
    previous_y_2 = split_windows["prev_y_2"]
    if not isinstance(previous_y_1, pd.Series) or not isinstance(previous_y_2, pd.Series):
        raise TypeError("Prepared windows must store previous target values in pandas Series objects.")
    if difference_order == 0:
        return inverse_scale_values(predicted_model_values, scaler_row)
    if difference_order == 1:
        delta_original = predicted_model_values * scale
        return previous_y_1.reset_index(drop=True) + delta_original.reset_index(drop=True)
    if difference_order == 2:
        second_difference_original = predicted_model_values * scale
        return (
            second_difference_original.reset_index(drop=True)
            + 2 * previous_y_1.reset_index(drop=True)
            - previous_y_2.reset_index(drop=True)
        )
    raise ValueError(f"Unsupported difference order '{difference_order}'.")


def inverse_scale_values(values: pd.Series, scaler: dict[str, Any]) -> pd.Series:
    if str(scaler["scaler"]) == "standard":
        return values * float(scaler["train_std"]) + float(scaler["train_mean"])
    if str(scaler["scaler"]) == "minmax":
        return values * float(scaler["train_scale"]) + float(scaler["train_min"])
    raise ValueError(f"Unsupported scaler '{scaler['scaler']}'.")


def select_hidden_units(lookback_steps: int, mlp_config: dict[str, Any]) -> int:
    strategy = str(mlp_config.get("hidden_units_strategy", "match_lookback"))
    if strategy == "match_lookback":
        return int(lookback_steps)
    return int(mlp_config.get("hidden_units", lookback_steps))


def format_learning_rate_alias(learning_rate_init: float) -> str:
    return f"{learning_rate_init:g}".replace("-", "m").replace(".", "p")


def build_candidate_label(granularity: str, difference_order: int, smoothing_window: str) -> str:
    return f"{granularity}_d{difference_order}_smooth_{smoothing_window}"


def smoothing_windows_for_granularity(protocol: dict[str, Any], granularity: str) -> list[str]:
    windows_by_granularity = protocol.get("train_only_smoothing_windows_by_granularity", {})
    if isinstance(windows_by_granularity, dict) and granularity in windows_by_granularity:
        return [str(value) for value in windows_by_granularity[granularity]]
    return [str(value) for value in protocol.get("train_only_smoothing_windows", ["none"])]


def transform_name_for_order(difference_order: int) -> str:
    names = {0: "level", 1: "first_difference", 2: "second_difference"}
    return names[int(difference_order)]


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
    """Fit one fixed-order univariate ARIMA and align predictions to holdout timestamps."""
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
    order = cast(list[int] | tuple[int, int, int], arima_config.get("order", [1, 0, 1]))
    if len(order) != 3:
        raise ValueError("arima.order must contain exactly three values: [p, d, q].")
    return int(order[0]), int(order[1]), int(order[2])


def protocol_train_test_boundary(protocol: dict[str, Any]) -> float:
    splits = protocol["splits"]
    return float(splits["train"])


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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(output_path)


def _frame_float(frame: pd.DataFrame, row_index: int, column: str) -> float:
    value = frame.at[row_index, column]
    if pd.isna(value):
        return float("nan")
    return float(value)
