from __future__ import annotations

import warnings
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any, TypedDict, cast

import numpy as np
import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import MLP as NeuralForecastMLP
from statsforecast import StatsForecast
from statsforecast.models import ARIMA, AutoETS, Holt, SimpleExponentialSmoothingOptimized

try:
    from exploration import build_forecast_metric_row, forecast_metrics
    from plots import (
        write_mlp_training_diagnostic_plots,
        write_test_comparison_metric_plots,
        write_test_comparison_plots,
        write_univariate_comparison_plots,
    )
    from preprocessing import (
        Scaler,
        apply_linear_scaler,
        build_forecasting_frame,
        change_granularity,
        fit_linear_scaler,
        split_train_test,
        statsforecast_frequency,
        steps_for_duration,
    )
except ImportError:  # pragma: no cover - package import path
    from .exploration import build_forecast_metric_row, forecast_metrics
    from .plots import (
        write_mlp_training_diagnostic_plots,
        write_test_comparison_metric_plots,
        write_test_comparison_plots,
        write_univariate_comparison_plots,
    )
    from .preprocessing import (
        Scaler,
        apply_linear_scaler,
        build_forecasting_frame,
        change_granularity,
        fit_linear_scaler,
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
    forecast_output_scale = str(analysis_policy.get("forecast_output_scale", "original"))

    arima_config = analysis_policy.get("arima", {})
    if bool(arima_config.get("enabled", False)):
        print("[Univariate] Starting ARIMA grid.", flush=True)
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
        print(
            f"[Univariate] Finished ARIMA grid: {len(arima_results['arima_metrics'])} scored fits.",
            flush=True,
        )
    else:
        arima_results = {
            "arima_forecasts": pd.DataFrame(columns=["ds", "y", "forecast", "granularity"]),
            "arima_metrics": pd.DataFrame(columns=["granularity", "model", "mae", "rmse", "r2"]),
        }

    exponential_smoothing_config = analysis_policy.get("exponential_smoothing", {})
    if bool(exponential_smoothing_config.get("enabled", False)):
        print("[Univariate] Starting exponential-smoothing grid.", flush=True)
        exponential_smoothing_results = run_univariate_exponential_smoothing_forecasts(
            datasets=datasets,
            target_column=target_column,
            target_granularities=target_granularities,
            granularity_options=granularity_options,
            resampling_policy=resampling_policy,
            forecasting_policy=forecasting_policy,
            protocol=protocol,
            output_dir=output_dir,
            timestamp_column=timestamp_column,
        )
        print(
            "[Univariate] Finished exponential-smoothing grid: "
            f"{len(exponential_smoothing_results['exponential_smoothing_metrics'])} scored fits.",
            flush=True,
        )
    else:
        exponential_smoothing_results = {
            "exponential_smoothing_forecasts": pd.DataFrame(columns=["ds", "y", "forecast", "granularity"]),
            "exponential_smoothing_metrics": pd.DataFrame(
                columns=["granularity", "candidate_label", "model", "mae", "rmse", "r2"]
            ),
        }

    print("[Univariate] Preparing MLP candidate windows.", flush=True)
    candidate_specs, scaling_summary, mlp_window_summary = build_mlp_preparation_candidates(
        datasets=datasets,
        target_column=target_column,
        target_granularities=target_granularities,
        granularity_options=granularity_options,
        resampling_policy=resampling_policy,
        protocol=protocol,
        timestamp_column=timestamp_column,
    )
    print(f"[Univariate] Prepared {len(candidate_specs)} MLP candidate grids.", flush=True)

    print("[Univariate] Starting MLP grid.", flush=True)
    analysis_policy.setdefault("mlp", {}).setdefault("forecast_output_scale", forecast_output_scale)
    mlp_results = run_mlp_test_comparison(
        candidate_specs=candidate_specs,
        scaling_summary=scaling_summary,
        analysis_policy=analysis_policy,
        write_full_forecasts=bool(forecasting_policy.get("write_full_forecasts", False)),
        output_dir=output_dir,
    )
    print(
        f"[Univariate] Finished MLP grid: {len(mlp_results['mlp_test_comparison'])} scored fits.",
        flush=True,
    )

    print("[Univariate] Writing univariate plots.", flush=True)
    write_test_comparison_metric_plots(
        mlp_results["mlp_test_comparison"],
        plots_dir / "test_comparison_metrics",
    )
    write_test_comparison_plots(
        mlp_results["mlp_forecasts"],
        plots_dir / "test_comparison",
    )
    write_mlp_training_diagnostic_plots(
        mlp_results["mlp_training_history"],
        mlp_results["mlp_test_comparison"],
        plots_dir / "training_diagnostics",
    )
    write_univariate_comparison_plots(
        arima_metrics=arima_results["arima_metrics"],
        mlp_test_comparison=mlp_results["mlp_test_comparison"],
        arima_forecasts=arima_results["arima_forecasts"],
        mlp_forecasts=mlp_results["mlp_forecasts"],
        exponential_smoothing_metrics=exponential_smoothing_results["exponential_smoothing_metrics"],
        exponential_smoothing_forecasts=exponential_smoothing_results["exponential_smoothing_forecasts"],
        output_dir=plots_dir,
    )
    print("[Univariate] Finished univariate analysis.", flush=True)
    return {
        **arima_results,
        **exponential_smoothing_results,
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
    """Fit the configured ARIMA order grid on each univariate modeling granularity."""
    del reports_dir, plots_dir
    forecast_frames = []
    metric_rows = []
    arima_config = forecasting_policy.get("univariate", {}).get("arima", forecasting_policy.get("arima", {}))
    output_scale = str(forecasting_policy.get("univariate", {}).get("forecast_output_scale", "original"))
    scaling_method = str(protocol.get("scaling", {}).get("method", "standard"))
    train_fraction = protocol_train_test_boundary(protocol)
    lookback_duration = str(protocol.get("lookback_window", "10min"))
    write_full_forecasts = bool(forecasting_policy.get("write_full_forecasts", False))
    arima_plan = build_arima_fit_plan(
        target_granularities,
        granularity_options,
        resampling_policy,
        protocol,
        arima_config,
        lookback_duration,
    )
    total_models = len(arima_plan)
    completed_models = 0
    run_started = perf_counter()
    if total_models:
        print(f"[ARIMA] Starting fixed-order grid: {total_models} model fits.", flush=True)

    for granularity in target_granularities:
        id_key = f"{protocol['dataset_prefix']}_{granularity}"
        if id_key not in datasets:
            continue

        series_frame = build_forecasting_frame(datasets[id_key], id_key, target_column, timestamp_column)
        train_frame, test_frame = split_train_test(series_frame, train_fraction)
        frequency = statsforecast_frequency(granularity, granularity_options, resampling_policy)
        lookback_steps = steps_for_duration(lookback_duration, frequency)
        for p_value, d_value, q_value in arima_orders_for_granularity(arima_config, granularity, lookback_steps):
            model_alias = f"ARIMA_p{p_value}_d{d_value}_q{q_value}"
            candidate_label = f"{granularity}_arima_d{d_value}_pq{p_value}"
            print(
                f"[ARIMA] Fitting {completed_models + 1}/{total_models}: "
                f"granularity={granularity} order=({p_value},{d_value},{q_value}) "
                f"train_rows={len(train_frame)} test_rows={len(test_frame)}",
                flush=True,
            )
            try:
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
            except Exception as exc:  # pragma: no cover - defensive for long model grids
                completed_models += 1
                elapsed_seconds = perf_counter() - run_started
                average_seconds = elapsed_seconds / completed_models
                remaining_seconds = max(total_models - completed_models, 0) * average_seconds
                print(
                    f"[ARIMA] Failed {completed_models}/{total_models}: "
                    f"{model_alias} for {granularity}: {type(exc).__name__}: {exc} | "
                    f"elapsed={format_duration(elapsed_seconds)} avg/model={format_duration(average_seconds)} "
                    f"ETA={format_duration(remaining_seconds)}",
                    flush=True,
                )
                metric_rows.append(
                    {
                        "granularity": granularity,
                        "candidate_label": candidate_label,
                        "model": model_alias,
                        "p": p_value,
                        "d": d_value,
                        "q": q_value,
                        "train_rows": len(train_frame),
                        "test_rows": len(test_frame),
                        "mae": np.nan,
                        "rmse": np.nan,
                        "mape": np.nan,
                        "smape": np.nan,
                        "bias": np.nan,
                        "r2": np.nan,
                        "warning_count": 1,
                        "warnings": str(exc),
                        "forecast_output_scale": output_scale,
                    }
                )
                continue
            forecast_frame = apply_direct_forecast_output_scale(
                forecast_frame,
                train_frame=train_frame,
                output_scale=output_scale,
                scaling_method=scaling_method,
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
                candidate_label=candidate_label,
            )
            forecast_frames.append(forecast_frame)
            metric_row = build_forecast_metric_row(forecast_frame)
            metric_row["candidate_label"] = candidate_label
            metric_row["warning_count"] = int(forecast_frame.attrs.get("warning_count", 0))
            metric_row["warnings"] = "; ".join(forecast_frame.attrs.get("warnings", []))
            metric_row["forecast_output_scale"] = output_scale
            metric_rows.append(metric_row)
            completed_models += 1
            elapsed_seconds = perf_counter() - run_started
            average_seconds = elapsed_seconds / completed_models
            remaining_seconds = max(total_models - completed_models, 0) * average_seconds
            print(
                f"[ARIMA] Finished {completed_models}/{total_models}: "
                f"{model_alias} MAE={metric_row['mae']:.4f} RMSE={metric_row['rmse']:.4f} "
                f"R2={metric_row.get('r2', np.nan):.4f} | "
                f"elapsed={format_duration(elapsed_seconds)} avg/model={format_duration(average_seconds)} "
                f"ETA={format_duration(remaining_seconds)}",
                flush=True,
            )

    if not metric_rows:
        return {
            "arima_forecasts": pd.DataFrame(columns=["ds", "y", "forecast"]),
            "arima_metrics": pd.DataFrame(columns=["granularity", "candidate_label", "model", "mae", "rmse", "r2"]),
        }

    forecasts = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows).sort_values(["granularity", "d", "p"]).reset_index(drop=True)
    if write_full_forecasts:
        write_csv_output(forecasts, output_dir / "arima_forecasts.csv")
    print(
        f"[ARIMA] Completed fixed-order grid: {len(metrics)} scored fits "
        f"in {format_duration(perf_counter() - run_started)}.",
        flush=True,
    )
    return {"arima_forecasts": forecasts, "arima_metrics": metrics}


def run_univariate_exponential_smoothing_forecasts(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    target_granularities: list[str],
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    forecasting_policy: dict[str, Any],
    protocol: dict[str, Any],
    output_dir: Path,
    timestamp_column: str,
) -> dict[str, pd.DataFrame]:
    """Fit the configured non-seasonal exponential-smoothing variants per granularity."""
    forecast_frames = []
    metric_rows = []
    smoothing_config = forecasting_policy.get("univariate", {}).get("exponential_smoothing", {})
    output_scale = str(forecasting_policy.get("univariate", {}).get("forecast_output_scale", "original"))
    scaling_method = str(protocol.get("scaling", {}).get("method", "standard"))
    train_fraction = protocol_train_test_boundary(protocol)
    write_full_forecasts = bool(forecasting_policy.get("write_full_forecasts", False))
    variants = exponential_smoothing_variants(smoothing_config)
    total_models = sum(1 for granularity in target_granularities if f"{protocol['dataset_prefix']}_{granularity}" in datasets)
    total_models *= len(variants)
    completed_models = 0
    run_started = perf_counter()
    if total_models:
        print(f"[ETS] Starting exponential-smoothing grid: {total_models} model fits.", flush=True)

    for granularity in target_granularities:
        id_key = f"{protocol['dataset_prefix']}_{granularity}"
        if id_key not in datasets:
            continue

        series_frame = build_forecasting_frame(datasets[id_key], id_key, target_column, timestamp_column)
        train_frame, test_frame = split_train_test(series_frame, train_fraction)
        frequency = statsforecast_frequency(granularity, granularity_options, resampling_policy)
        for variant in variants:
            model_alias = str(variant["name"])
            candidate_label = f"{granularity}_ets_{model_alias}"
            print(
                f"[ETS] Fitting {completed_models + 1}/{total_models}: "
                f"granularity={granularity} model={model_alias} "
                f"train_rows={len(train_frame)} test_rows={len(test_frame)}",
                flush=True,
            )
            try:
                forecast_frame = forecast_exponential_smoothing(
                    train_frame=train_frame,
                    test_frame=test_frame,
                    frequency=frequency,
                    model_alias=model_alias,
                    smoothing_config=smoothing_config,
                    variant=variant,
                )
            except Exception as exc:  # pragma: no cover - defensive for optional model family
                completed_models += 1
                elapsed_seconds = perf_counter() - run_started
                average_seconds = elapsed_seconds / completed_models
                remaining_seconds = max(total_models - completed_models, 0) * average_seconds
                print(
                    f"[ETS] Failed {completed_models}/{total_models}: "
                    f"{model_alias} for {granularity}: {type(exc).__name__}: {exc} | "
                    f"elapsed={format_duration(elapsed_seconds)} avg/model={format_duration(average_seconds)} "
                    f"ETA={format_duration(remaining_seconds)}",
                    flush=True,
                )
                metric_rows.append(
                    {
                        "granularity": granularity,
                        "candidate_label": candidate_label,
                        "model": model_alias,
                        "train_rows": len(train_frame),
                        "test_rows": len(test_frame),
                        "mae": np.nan,
                        "rmse": np.nan,
                        "mape": np.nan,
                        "smape": np.nan,
                        "bias": np.nan,
                        "r2": np.nan,
                        "warning_count": 1,
                        "warnings": str(exc),
                        "forecast_output_scale": output_scale,
                    }
                )
                continue
            forecast_frame = apply_direct_forecast_output_scale(
                forecast_frame,
                train_frame=train_frame,
                output_scale=output_scale,
                scaling_method=scaling_method,
            )
            forecast_frame["id_key"] = id_key
            forecast_frame["granularity"] = granularity
            forecast_frame["model"] = model_alias
            forecast_frame["candidate_label"] = candidate_label
            forecast_frame["train_rows"] = len(train_frame)
            forecast_frame["test_rows"] = len(test_frame)
            forecast_frame["forecast_output_scale"] = output_scale
            forecast_frames.append(forecast_frame)
            metric_row = build_generic_forecast_metric_row(forecast_frame)
            metric_row["forecast_output_scale"] = output_scale
            metric_rows.append(metric_row)
            completed_models += 1
            elapsed_seconds = perf_counter() - run_started
            average_seconds = elapsed_seconds / completed_models
            remaining_seconds = max(total_models - completed_models, 0) * average_seconds
            print(
                f"[ETS] Finished {completed_models}/{total_models}: "
                f"{model_alias} MAE={metric_row['mae']:.4f} RMSE={metric_row['rmse']:.4f} "
                f"R2={metric_row.get('r2', np.nan):.4f} | "
                f"elapsed={format_duration(elapsed_seconds)} avg/model={format_duration(average_seconds)} "
                f"ETA={format_duration(remaining_seconds)}",
                flush=True,
            )

    if not metric_rows:
        return {
            "exponential_smoothing_forecasts": pd.DataFrame(columns=["ds", "y", "forecast"]),
            "exponential_smoothing_metrics": pd.DataFrame(
                columns=["granularity", "candidate_label", "model", "mae", "rmse", "r2"]
            ),
        }

    forecasts = pd.concat(forecast_frames, ignore_index=True) if forecast_frames else pd.DataFrame()
    metrics = pd.DataFrame(metric_rows).sort_values(["granularity", "model"]).reset_index(drop=True)
    if write_full_forecasts:
        write_csv_output(forecasts, output_dir / "exponential_smoothing_forecasts.csv")
    print(
        f"[ETS] Completed exponential-smoothing grid: {len(metrics)} scored fits "
        f"in {format_duration(perf_counter() - run_started)}.",
        flush=True,
    )
    return {"exponential_smoothing_forecasts": forecasts, "exponential_smoothing_metrics": metrics}


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

    raw_id_key = f"{protocol['dataset_prefix']}_raw"
    if raw_id_key not in datasets:
        return [], pd.DataFrame(), pd.DataFrame()

    raw_frame = build_forecasting_frame(datasets[raw_id_key], raw_id_key, target_column, timestamp_column)
    raw_train_frame, raw_test_frame = split_train_test(raw_frame, train_fraction)
    scaler = fit_linear_scaler(raw_train_frame["y"], scaling_method)
    raw_train_frame = add_scaled_target(raw_train_frame, scaler)
    raw_test_frame = add_scaled_target(raw_test_frame, scaler)

    for granularity in target_granularities:
        id_key = f"{protocol['dataset_prefix']}_{granularity}"
        frequency = statsforecast_frequency(granularity, granularity_options, resampling_policy)
        lookback_steps = steps_for_duration(lookback_duration, frequency)
        horizon_steps = steps_for_duration(horizon_duration, frequency)
        split_frame = build_split_first_granularity_frame(
            raw_train_frame=raw_train_frame,
            raw_test_frame=raw_test_frame,
            id_key=id_key,
            granularity=granularity,
            target_column=target_column,
            timestamp_column=timestamp_column,
            granularity_options=granularity_options,
            resampling_policy=resampling_policy,
        )

        for difference_order in difference_orders:
            transformed_frame = build_transformed_mlp_frame(split_frame, difference_order)
            for smoothing_window in smoothing_windows_for_granularity(protocol, granularity):
                if smoothing_window != "none":
                    try:
                        steps_for_duration(smoothing_window, frequency)
                    except ValueError:
                        continue
                prepared_frame = apply_train_only_smoothing(transformed_frame, smoothing_window, frequency)
                scaling_summary = build_pre_aggregation_scaling_summary(
                    prepared_frame,
                    scaler,
                    granularity,
                    target_column,
                    frequency,
                    scaling_method,
                    difference_order,
                    smoothing_window,
                )
                scaling_summaries.append(scaling_summary)
                scaler_row = scaling_summary.iloc[0].to_dict()
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
    write_full_forecasts: bool,
    output_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Forecast the chronological test split for every configured MLP preparation candidate."""
    del scaling_summary
    mlp_config = analysis_policy.get("mlp", {})
    if not bool(mlp_config.get("enabled", False)) or not candidate_specs:
        return {
            "mlp_forecasts": pd.DataFrame(),
            "mlp_test_comparison": pd.DataFrame(),
            "mlp_training_history": pd.DataFrame(),
            "mlp_training_summary": pd.DataFrame(),
        }

    metric_rows = []
    test_forecasts = []
    training_history_frames = []
    candidate_specs = select_configured_candidate_specs(candidate_specs, mlp_config)
    learning_rates = mlp_config.get("learning_rate_grid", mlp_config.get("initial_learning_rate_grid", [0.001]))
    model_variants = configured_mlp_model_variants(mlp_config)
    candidate_specs = limit_sequence(candidate_specs, mlp_config.get("candidate_limit"))
    learning_rates = limit_sequence([float(value) for value in learning_rates], mlp_config.get("learning_rate_limit"))
    total_models = sum(
        1
        for candidate in candidate_specs
        if not candidate["train_windows"]["x"].empty and not candidate["test_windows"]["x"].empty
        for _ in learning_rates_for_candidate(candidate, learning_rates, mlp_config)
        for _ in model_variants
    )
    completed_models = 0
    run_started = perf_counter()
    if total_models:
        print(
            "[MLP] Starting candidate review: "
            f"{len(candidate_specs)} candidates, {len(learning_rates)} learning rates, "
            f"{len(model_variants)} model variants, {total_models} model fits. "
            f"accelerator={mlp_config.get('accelerator', 'auto')} "
            f"windows_batch_size={mlp_config.get('windows_batch_size', 1024)}",
            flush=True,
        )

    for candidate_index, candidate in enumerate(candidate_specs, start=1):
        train_windows = candidate["train_windows"]
        test_windows = candidate["test_windows"]
        if train_windows["x"].empty or test_windows["x"].empty:
            print(
                f"[MLP] Skipping candidate {candidate_index}/{len(candidate_specs)}: "
                f"{candidate['candidate_label']} has empty train/test windows.",
                flush=True,
            )
            continue

        print(
            f"[MLP] Candidate {candidate_index}/{len(candidate_specs)}: "
            f"{candidate['candidate_label']} | train_windows={len(train_windows['x'])} "
            f"test_windows={len(test_windows['x'])}",
            flush=True,
        )
        candidate_settings = mlp_candidate_settings(candidate, mlp_config)
        candidate_learning_rates = learning_rates_for_candidate(candidate, learning_rates, mlp_config)
        for variant in model_variants:
            variant_config = mlp_config_for_candidate(
                mlp_config_for_variant(mlp_config, variant),
                candidate_settings,
            )
            hidden_layer_sizes = select_hidden_layer_sizes(candidate["lookback_steps"], variant_config)
            hidden_units = int(hidden_layer_sizes[0])
            model_name = mlp_model_name(variant_config)
            for learning_rate_index, learning_rate_init in enumerate(candidate_learning_rates, start=1):
                model_started = perf_counter()
                print(
                    f"[MLP] Training {completed_models + 1}/{total_models}: "
                    f"candidate={candidate['candidate_label']} "
                    f"variant={variant_config['name']} "
                    f"hidden_layers={hidden_layer_sizes} "
                    f"lr={float(learning_rate_init):g} "
                    f"({learning_rate_index}/{len(candidate_learning_rates)} for candidate)",
                    flush=True,
                )
                test_forecast, warning_messages, training_history = forecast_mlp_candidate(
                    candidate=candidate,
                    hidden_layer_sizes=hidden_layer_sizes,
                    learning_rate_init=float(learning_rate_init),
                    mlp_config=variant_config,
                )
                if not training_history.empty:
                    training_history["model"] = model_name
                    training_history["model_variant"] = str(variant_config["name"])
                    training_history["engine"] = str(variant_config.get("engine", "neuralforecast"))
                    training_history["granularity"] = candidate["granularity"]
                    training_history["candidate_label"] = candidate["candidate_label"]
                    training_history["difference_order"] = candidate["difference_order"]
                    training_history["transform_name"] = candidate["transform_name"]
                    training_history["training_smoothing_window"] = candidate["training_smoothing_window"]
                    training_history["learning_rate_init"] = float(learning_rate_init)
                    training_history["hidden_layer_sizes"] = repr(hidden_layer_sizes)
                    training_history["num_layers"] = int(variant_config.get("num_layers", len(hidden_layer_sizes)))
                    training_history["selected_role"] = candidate.get("selected_role", "")
                    training_history_frames.append(training_history)
                if test_forecast.empty:
                    completed_models += 1
                    print(
                        f"[MLP] Finished {completed_models}/{total_models}: no aligned forecasts produced.",
                        flush=True,
                    )
                    continue
                test_metrics = forecast_metrics(test_forecast["y"], test_forecast["forecast"])
                model_seconds = perf_counter() - model_started
                completed_models += 1
                elapsed_seconds = perf_counter() - run_started
                average_seconds = elapsed_seconds / completed_models
                remaining_seconds = max(total_models - completed_models, 0) * average_seconds
                print(
                    f"[MLP] Finished {completed_models}/{total_models} in {format_duration(model_seconds)} | "
                    f"MAE={test_metrics['mae']:.4f} RMSE={test_metrics['rmse']:.4f} R2={test_metrics['r2']:.4f} | "
                    f"avg/model={format_duration(average_seconds)} ETA={format_duration(remaining_seconds)} | "
                    f"warnings={len(warning_messages)}",
                    flush=True,
                )
                test_forecast["model"] = model_name
                test_forecast["model_variant"] = str(variant_config["name"])
                test_forecast["engine"] = str(variant_config.get("engine", "neuralforecast"))
                test_forecast["granularity"] = candidate["granularity"]
                test_forecast["split"] = "test"
                test_forecast["candidate_label"] = candidate["candidate_label"]
                test_forecast["difference_order"] = candidate["difference_order"]
                test_forecast["transform_name"] = candidate["transform_name"]
                test_forecast["training_smoothing_window"] = candidate["training_smoothing_window"]
                test_forecast["learning_rate_init"] = float(learning_rate_init)
                test_forecast["hidden_layer_sizes"] = repr(hidden_layer_sizes)
                test_forecast["selected_role"] = candidate.get("selected_role", "")
                test_forecast["forecast_output_scale"] = str(variant_config.get("forecast_output_scale", "original"))
                test_forecasts.append(test_forecast)
                metric_rows.append(
                    {
                        "model": model_name,
                        "model_variant": str(variant_config["name"]),
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
                        "second_hidden_units": hidden_layer_sizes[1] if len(hidden_layer_sizes) > 1 else np.nan,
                        "hidden_layer_sizes": repr(hidden_layer_sizes),
                        "forecast_output_scale": str(variant_config.get("forecast_output_scale", "original")),
                        "learning_rate_init": float(learning_rate_init),
                        "selected_role": candidate.get("selected_role", ""),
                        "engine": str(variant_config.get("engine", "neuralforecast")),
                        "optimizer": str(variant_config.get("optimizer", variant_config.get("solver", "adam"))),
                        "num_layers": int(variant_config.get("num_layers", len(hidden_layer_sizes))),
                        "min_steps": int(variant_config.get("min_steps", 5000)),
                        "max_steps": int(variant_config.get("max_steps", 7500)),
                        "accelerator": str(variant_config.get("accelerator", "auto")),
                        "windows_batch_size": int(variant_config.get("windows_batch_size", 1024)),
                        "dataloader_num_workers": int(variant_config.get("dataloader_num_workers", 0)),
                        "training_seconds": model_seconds,
                        "warning_count": len(warning_messages),
                        "warnings": "; ".join(warning_messages),
                        **test_metrics,
                    }
                )

    test_metrics_frame = pd.DataFrame(metric_rows)
    if test_metrics_frame.empty:
        return {
            "mlp_forecasts": pd.DataFrame(),
            "mlp_test_comparison": pd.DataFrame(),
            "mlp_training_history": (
                pd.concat(training_history_frames, ignore_index=True)
                if training_history_frames
                else pd.DataFrame()
            ),
            "mlp_training_summary": pd.DataFrame(),
        }

    test_metrics_frame = test_metrics_frame.sort_values(
        ["granularity", "difference_order", "training_smoothing_window", "model_variant", "learning_rate_init"]
    ).reset_index(drop=True)
    test_metrics_frame["selection_mode"] = str(mlp_config.get("selection_mode", "test_comparison"))
    training_history_frame = (
        pd.concat(training_history_frames, ignore_index=True)
        if training_history_frames
        else pd.DataFrame()
    )
    if not training_history_frame.empty:
        write_csv_output(training_history_frame, output_dir / "training_results" / "mlp_training_history.csv")
    training_summary_frame = build_training_summary_frame(training_history_frame, test_metrics_frame)
    print(
        f"[MLP] Completed candidate review: {len(test_metrics_frame)} scored model fits "
        f"in {format_duration(perf_counter() - run_started)}.",
        flush=True,
    )

    forecast_frame = pd.concat(test_forecasts, ignore_index=True)
    if write_full_forecasts:
        write_csv_output(forecast_frame, output_dir / "mlp_forecasts.csv")

    return {
        "mlp_forecasts": forecast_frame,
        "mlp_test_comparison": test_metrics_frame.copy(),
        "mlp_training_history": training_history_frame,
        "mlp_training_summary": training_summary_frame,
    }


def select_configured_candidate_specs(
    candidate_specs: list[dict[str, Any]],
    mlp_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Reduce preparation candidates to the selected set when configured."""
    if not bool(mlp_config.get("use_selected_candidate_combinations", False)):
        return candidate_specs

    selected_config = mlp_config.get("selected_candidate_combinations", [])
    if not isinstance(selected_config, list) or not selected_config:
        raise ValueError("use_selected_candidate_combinations is true but no selected candidates are configured.")

    selected_by_key: dict[tuple[str, int, str], dict[str, Any]] = {}
    for selected in selected_config:
        if not isinstance(selected, dict):
            raise TypeError("Each selected candidate combination must be an object.")
        selected_by_key[candidate_selection_key(selected)] = selected

    selected_specs: list[dict[str, Any]] = []
    for candidate in candidate_specs:
        selected = selected_by_key.get(candidate_selection_key(candidate))
        if selected is None:
            continue
        configured_candidate = dict(candidate)
        if "learning_rate_init" in selected:
            configured_candidate["selected_learning_rate_init"] = float(selected["learning_rate_init"])
        configured_candidate["selected_role"] = str(selected.get("role", ""))
        selected_specs.append(configured_candidate)

    matched_keys = {candidate_selection_key(candidate) for candidate in selected_specs}
    missing = set(selected_by_key) - matched_keys
    if missing:
        missing_labels = ", ".join(
            f"{granularity}/d{difference_order}/{smoothing_window}"
            for granularity, difference_order, smoothing_window in sorted(missing)
        )
        raise ValueError(f"Selected candidate combinations were not available in the generated grid: {missing_labels}")

    print(
        f"[MLP] Using {len(selected_specs)} selected candidate combinations from configuration.",
        flush=True,
    )
    return selected_specs


def candidate_selection_key(candidate: dict[str, Any]) -> tuple[str, int, str]:
    return (
        str(candidate["granularity"]),
        int(candidate["difference_order"]),
        str(candidate["training_smoothing_window"]),
    )


def learning_rates_for_candidate(
    candidate: dict[str, Any],
    default_learning_rates: list[float],
    mlp_config: dict[str, Any],
) -> list[float]:
    candidate_settings = mlp_candidate_settings(candidate, mlp_config)
    if isinstance(candidate_settings.get("learning_rate_grid"), list):
        return limit_sequence(
            [float(value) for value in candidate_settings["learning_rate_grid"]],
            mlp_config.get("learning_rate_limit"),
        )
    if "learning_rate_init" in candidate_settings:
        return [float(candidate_settings["learning_rate_init"])]
    if bool(mlp_config.get("use_selected_learning_rates", True)) and "selected_learning_rate_init" in candidate:
        return [float(candidate["selected_learning_rate_init"])]
    return default_learning_rates


def mlp_candidate_settings(candidate: dict[str, Any], mlp_config: dict[str, Any]) -> dict[str, Any]:
    configured = mlp_config.get("candidate_settings", {})
    if not isinstance(configured, dict):
        return {}
    by_label = configured.get(str(candidate.get("candidate_label", "")), {})
    return dict(by_label) if isinstance(by_label, dict) else {}


def configured_mlp_model_variants(mlp_config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = mlp_config.get("model_variants", [])
    if not isinstance(variants, list) or not variants:
        return [
            {
                "name": "neuralforecast_single_hidden",
                "engine": str(mlp_config.get("engine", "neuralforecast")),
                "num_layers": int(mlp_config.get("num_layers", 1)),
                "hidden_units_strategy": str(mlp_config.get("hidden_units_strategy", "match_lookback")),
            }
        ]
    return [dict(variant) for variant in variants if isinstance(variant, dict)]


def mlp_config_for_variant(mlp_config: dict[str, Any], variant: dict[str, Any]) -> dict[str, Any]:
    variant_config = dict(mlp_config)
    variant_config.update(variant)
    variant_config["name"] = str(variant.get("name", variant_config.get("engine", "mlp")))
    return variant_config


def mlp_config_for_candidate(mlp_config: dict[str, Any], candidate_settings: dict[str, Any]) -> dict[str, Any]:
    candidate_config = dict(mlp_config)
    for key, value in candidate_settings.items():
        if key in {"learning_rate_grid", "learning_rate_init"}:
            continue
        candidate_config[key] = value
    return candidate_config


def mlp_model_name(mlp_config: dict[str, Any]) -> str:
    engine = str(mlp_config.get("engine", "neuralforecast"))
    variant = str(mlp_config.get("name", engine))
    if engine == "neuralforecast":
        return f"NeuralForecast_MLP_{variant}"
    return f"MLP_{variant}"


def add_scaled_target(frame: pd.DataFrame, scaler: Scaler) -> pd.DataFrame:
    scaled = frame.copy()
    scaled["y_scaled"] = apply_linear_scaler(scaled["y"], scaler)
    return scaled


def build_split_first_granularity_frame(
    raw_train_frame: pd.DataFrame,
    raw_test_frame: pd.DataFrame,
    id_key: str,
    granularity: str,
    target_column: str,
    timestamp_column: str,
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
) -> pd.DataFrame:
    """Apply aggregation after the chronological split, matching the assignment diagram."""
    fixed_policy = resampling_policy_with_fixed_origin(resampling_policy, raw_train_frame["ds"].iloc[0])
    train_frame = aggregate_forecasting_split(
        raw_train_frame,
        id_key,
        granularity,
        target_column,
        timestamp_column,
        granularity_options,
        fixed_policy,
    ).assign(split="train")
    test_frame = aggregate_forecasting_split(
        raw_test_frame,
        id_key,
        granularity,
        target_column,
        timestamp_column,
        granularity_options,
        fixed_policy,
    ).assign(split="test")
    return pd.concat([train_frame, test_frame], ignore_index=True)


def resampling_policy_with_fixed_origin(resampling_policy: dict[str, Any], origin: Any) -> dict[str, Any]:
    """Keep split-first resampling on one grid instead of re-anchoring each split."""
    fixed_policy = dict(resampling_policy)
    fixed_policy["origin"] = pd.Timestamp(origin)
    return fixed_policy


def aggregate_forecasting_split(
    split_frame: pd.DataFrame,
    id_key: str,
    granularity: str,
    target_column: str,
    timestamp_column: str,
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
) -> pd.DataFrame:
    output_granularity = None if granularity == "raw" else granularity_options.get(granularity)
    source_frame = split_frame.rename(
        columns={
            "ds": timestamp_column,
            "y": target_column,
            "y_scaled": f"{target_column}_scaled",
        }
    )
    source_frame = source_frame[[timestamp_column, target_column, f"{target_column}_scaled"]].copy()
    granular_frame = change_granularity(
        source_frame,
        output_granularity,
        timestamp_column=timestamp_column,
        resampling_policy=resampling_policy,
    )
    forecast_frame = build_forecasting_frame(granular_frame, id_key, target_column, timestamp_column)
    scaled_frame = build_forecasting_frame(
        granular_frame,
        id_key,
        f"{target_column}_scaled",
        timestamp_column,
    )[["ds", "y"]].rename(columns={"y": "y_scaled"})
    return forecast_frame.merge(scaled_frame, on="ds", how="inner")


def build_pre_aggregation_scaling_summary(
    prepared_frame: pd.DataFrame,
    scaler: Scaler,
    granularity: str,
    target_column: str,
    frequency: str,
    scaling_method: str,
    difference_order: int,
    smoothing_window: str,
) -> pd.DataFrame:
    """Describe the raw-train scaler applied before aggregation."""
    train_mask = prepared_frame["split"] == "train"
    test_mask = prepared_frame["split"] == "test"
    train_values = prepared_frame.loc[train_mask, "model_series"].dropna()
    return pd.DataFrame(
        [
            {
                "granularity": granularity,
                "frequency": frequency,
                "target_column": target_column,
                "scaler": scaling_method,
                "fit_split": "train",
                "scaled_series": "raw_target_before_aggregation",
                "difference_order": int(difference_order),
                "transform_name": transform_name_for_order(difference_order),
                "training_smoothing_window": smoothing_window,
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_non_null_rows": int(train_values.size),
                "train_mean": scaler.get("mean", np.nan),
                "train_std": scaler.get("std", np.nan),
                "train_min": scaler.get("min"),
                "train_max": scaler.get("max"),
                "train_scale": float(scaler.get("scale", np.nan)),
                "train_start": prepared_frame.loc[train_mask, "ds"].iloc[0],
                "train_end": prepared_frame.loc[train_mask, "ds"].iloc[-1],
                "test_start": prepared_frame.loc[test_mask, "ds"].iloc[0],
                "test_end": prepared_frame.loc[test_mask, "ds"].iloc[-1],
            }
        ]
    )


def build_transformed_mlp_frame(frame: pd.DataFrame, difference_order: int) -> pd.DataFrame:
    transformed = frame.copy()
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
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    """Fit one NeuralForecast MLP on train data and score the chronological test window."""
    prepared = candidate["prepared_frame"].dropna(subset=["model_series"]).copy()
    if prepared.empty:
        return pd.DataFrame(columns=["ds", "y", "forecast"]), [], pd.DataFrame()

    test_size = int((prepared["split"] == "test").sum())
    if test_size <= 0:
        return pd.DataFrame(columns=["ds", "y", "forecast"]), [], pd.DataFrame()

    model_alias = f"MLP_lr_{format_learning_rate_alias(learning_rate_init)}"
    forecast_frame = prepared[["unique_id", "ds", "model_series"]].rename(columns={"model_series": "y"})
    dataloader_kwargs = {
        "num_workers": int(mlp_config.get("dataloader_num_workers", 0)),
        "pin_memory": bool(mlp_config.get("dataloader_pin_memory", False)),
    }
    model_kwargs: dict[str, Any] = {
        "h": int(candidate["horizon_steps"]),
        "input_size": int(candidate["lookback_steps"]),
        "num_layers": int(mlp_config.get("num_layers", 1)),
        "hidden_size": int(hidden_units),
        "max_steps": int(mlp_config.get("max_steps", 7500)),
        "learning_rate": float(learning_rate_init),
        "batch_size": int(mlp_config.get("batch_size", 1)),
        "windows_batch_size": int(mlp_config.get("windows_batch_size", 1024)),
        "inference_windows_batch_size": int(mlp_config.get("inference_windows_batch_size", -1)),
        "scaler_type": str(mlp_config.get("scaler_type", "identity")),
        "random_seed": int(mlp_config.get("random_state", 42)),
        "alias": model_alias,
        "min_steps": int(mlp_config.get("min_steps", 5000)),
        "accelerator": str(mlp_config.get("accelerator", "auto")),
        "devices": int(mlp_config.get("devices", 1)),
        "dataloader_kwargs": dataloader_kwargs,
        "logger": False,
        "enable_checkpointing": False,
        "enable_model_summary": False,
        "enable_progress_bar": False,
    }
    if "precision" in mlp_config:
        model_kwargs["precision"] = mlp_config["precision"]
    neural_model = NeuralForecastMLP(**model_kwargs)
    engine = NeuralForecast(models=[neural_model], freq=str(candidate["frequency"]))
    print(
        f"[MLP] Fitting/cross-validating {candidate['candidate_label']} "
        f"lr={learning_rate_init:g} test_size={test_size} horizon_steps={candidate['horizon_steps']}",
        flush=True,
    )
    cross_validation_frame = pd.DataFrame()
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            cross_validation = engine.cross_validation(
                df=forecast_frame,
                n_windows=None,
                val_size=0,
                test_size=test_size,
                step_size=1,
                refit=False,
                verbose=False,
            )
            cross_validation_frame = cast(pd.DataFrame, cross_validation)
            print(
                f"[MLP] Cross-validation returned {len(cross_validation_frame.index)} forecast rows for "
                f"{candidate['candidate_label']} lr={learning_rate_init:g}. Aligning forecasts...",
                flush=True,
            )
        except PermissionError:
            if int(mlp_config.get("dataloader_num_workers", 0)) <= 0:
                raise
            fallback_config = dict(mlp_config)
            fallback_config["dataloader_num_workers"] = 0
            print(
                "[MLP] DataLoader workers failed with a Windows permission error; "
                "retrying this model with dataloader_num_workers=0.",
                flush=True,
            )
            fallback_forecast, fallback_warnings, fallback_training_history = forecast_neural_mlp_candidate(
                candidate=candidate,
                hidden_units=hidden_units,
                learning_rate_init=learning_rate_init,
                mlp_config=fallback_config,
            )
            return fallback_forecast, [
                "DataLoader worker fallback used after Windows PermissionError.",
                *fallback_warnings,
            ], fallback_training_history
    warnings_seen = sorted({str(item.message) for item in caught_warnings})
    fitted_model = engine.models[0] if getattr(engine, "models", None) else neural_model
    training_history = build_training_history_frame(fitted_model)
    aligned_forecast = build_neural_mlp_test_forecast(
        cross_validation=cross_validation_frame,
        model_alias=model_alias,
        candidate=candidate,
        output_scale=str(mlp_config.get("forecast_output_scale", "original")),
    )
    print(
        f"[MLP] Aligned {len(aligned_forecast)} forecast rows for "
        f"{candidate['candidate_label']} lr={learning_rate_init:g}.",
        flush=True,
    )
    return aligned_forecast, warnings_seen, training_history


def forecast_mlp_candidate(
    candidate: dict[str, Any],
    hidden_layer_sizes: tuple[int, ...],
    learning_rate_init: float,
    mlp_config: dict[str, Any],
) -> tuple[pd.DataFrame, list[str], pd.DataFrame]:
    engine = str(mlp_config.get("engine", "neuralforecast"))
    if engine == "neuralforecast":
        return forecast_neural_mlp_candidate(
            candidate=candidate,
            hidden_units=int(hidden_layer_sizes[0]),
            learning_rate_init=learning_rate_init,
            mlp_config=mlp_config,
        )
    raise ValueError(f"Unsupported MLP engine '{engine}'.")


def build_training_history_frame(neural_model: Any) -> pd.DataFrame:
    """Return per-step NeuralForecast training loss captured during fitting."""
    train_trajectories = getattr(neural_model, "train_trajectories", [])
    rows = [{"step": int(step), "train_loss": float(loss)} for step, loss in train_trajectories if pd.notna(loss)]
    return build_loss_history_frame(rows)


def build_loss_history_frame(loss_history: Any) -> pd.DataFrame:
    if isinstance(loss_history, list) and loss_history and isinstance(loss_history[0], dict):
        rows = [
            {"step": int(row["step"]), "train_loss": float(row["train_loss"])}
            for row in loss_history
            if pd.notna(row.get("train_loss"))
        ]
    else:
        rows = [
            {"step": int(index), "train_loss": float(loss)}
            for index, loss in enumerate(list(loss_history), start=1)
            if pd.notna(loss)
        ]
    if not rows:
        return pd.DataFrame(columns=["step", "train_loss", "train_loss_rolling", "best_train_loss"])
    frame = pd.DataFrame(rows).drop_duplicates(subset=["step"], keep="last").sort_values("step").reset_index(drop=True)
    rolling_window = max(1, min(50, len(frame.index) // 20 or 1))
    frame["train_loss_rolling"] = frame["train_loss"].rolling(window=rolling_window, min_periods=1).mean()
    frame["best_train_loss"] = frame["train_loss"].cummin()
    return frame


def build_training_summary_frame(training_history: pd.DataFrame, test_metrics: pd.DataFrame) -> pd.DataFrame:
    if training_history.empty:
        return pd.DataFrame()
    group_columns = [
        column
        for column in ["granularity", "candidate_label", "model_variant", "learning_rate_init"]
        if column in training_history.columns
    ]
    summary = (
        training_history.sort_values("step")
        .groupby(group_columns, as_index=False)
        .tail(1)[group_columns + ["step", "train_loss", "best_train_loss"]]
        .rename(columns={"step": "training_steps", "train_loss": "final_train_loss"})
        .reset_index(drop=True)
    )
    if not test_metrics.empty:
        metric_columns = [
            column
            for column in ["mae", "rmse", "mape", "smape", "bias", "r2", "training_seconds"]
            if column in test_metrics.columns
        ]
        merge_columns = [column for column in group_columns if column in test_metrics.columns]
        summary = summary.merge(
            test_metrics[merge_columns + metric_columns],
            on=merge_columns,
            how="left",
        )
    return summary.sort_values(group_columns).reset_index(drop=True)


def build_neural_mlp_test_forecast(
    cross_validation: pd.DataFrame,
    model_alias: str,
    candidate: dict[str, Any],
    output_scale: str = "original",
) -> pd.DataFrame:
    test_windows = cast(WindowBundle, candidate["test_windows"])
    target_dates = test_windows["target_dates"]
    if not isinstance(target_dates, pd.Series):
        raise TypeError("Prepared windows must store target dates in a pandas Series.")
    target_date_set = set(pd.to_datetime(target_dates).to_list())

    if "cutoff" not in cross_validation.columns:
        selected = cross_validation[cross_validation["ds"].isin(target_date_set)].sort_values("ds").copy()
        return build_neural_mlp_forecast_from_final_rows(selected, model_alias, candidate, output_scale=output_scale)

    forecast_path = cross_validation.copy()
    forecast_path["ds"] = pd.to_datetime(forecast_path["ds"])
    forecast_path["cutoff"] = pd.to_datetime(forecast_path["cutoff"])
    horizon_offset = pd.to_timedelta(str(candidate["frequency"])) * int(candidate["horizon_steps"])
    forecast_path["horizon_offset"] = forecast_path["ds"] - forecast_path["cutoff"]
    selected = forecast_path[
        (forecast_path["horizon_offset"] == horizon_offset)
        & forecast_path["ds"].isin(target_date_set)
    ].sort_values(["cutoff", "ds"])
    if selected.empty:
        return pd.DataFrame(columns=["ds", "y", "forecast"])
    if int(candidate["horizon_steps"]) == 1:
        return build_neural_mlp_forecast_from_final_rows(selected, model_alias, candidate, output_scale=output_scale)

    records = []
    path_by_cutoff: dict[pd.Timestamp, pd.DataFrame] = {
        pd.Timestamp(cutoff): group.sort_values("ds")
        for cutoff, group in forecast_path.groupby("cutoff", sort=False)
    }
    for row in selected.itertuples(index=False):
        cutoff = pd.Timestamp(getattr(row, "cutoff"))
        target_date = pd.Timestamp(getattr(row, "ds"))
        cutoff_path = path_by_cutoff.get(cutoff)
        if cutoff_path is None:
            continue
        path_rows = cutoff_path[(cutoff_path["ds"] > cutoff) & (cutoff_path["ds"] <= target_date)]
        if len(path_rows) < int(candidate["horizon_steps"]):
            continue
        forecast_value, actual_value = reconstruct_neuralforecast_path(
            path_rows[model_alias].reset_index(drop=True),
            cutoff,
            target_date,
            candidate,
            output_scale=output_scale,
        )
        records.append({"ds": target_date, "y": actual_value, "forecast": forecast_value})

    return pd.DataFrame(records, columns=["ds", "y", "forecast"])


def build_neural_mlp_forecast_from_final_rows(
    selected: pd.DataFrame,
    model_alias: str,
    candidate: dict[str, Any],
    output_scale: str = "original",
) -> pd.DataFrame:
    if selected.empty:
        return pd.DataFrame(columns=["ds", "y", "forecast"])
    prepared_frame = cast(pd.DataFrame, candidate["prepared_frame"])
    split_windows = build_inverse_windows_for_dates(prepared_frame, selected["ds"])
    forecast = reconstruct_prepared_predictions(
        selected[model_alias].reset_index(drop=True),
        split_windows,
        cast(dict[str, Any], candidate["scaler_row"]),
        int(candidate["difference_order"]),
        output_scale=output_scale,
    )
    target_dates = split_windows["target_dates"]
    actual_values = actual_values_for_output_scale(
        split_windows,
        cast(dict[str, Any], candidate["scaler_row"]),
        output_scale,
    )
    if not isinstance(target_dates, pd.Series) or not isinstance(actual_values, pd.Series):
        raise TypeError("Prepared windows must store target dates and actual values in pandas Series objects.")
    return pd.DataFrame(
        {
            "ds": selected["ds"].reset_index(drop=True),
            "y": actual_values.reset_index(drop=True),
            "forecast": forecast.reset_index(drop=True),
        }
    )


def reconstruct_neuralforecast_path(
    predicted_model_values: pd.Series,
    cutoff: pd.Timestamp,
    target_date: pd.Timestamp,
    candidate: dict[str, Any],
    output_scale: str = "original",
) -> tuple[float, float]:
    prepared_frame = cast(pd.DataFrame, candidate["prepared_frame"]).reset_index(drop=True)
    scaler_row = cast(dict[str, Any], candidate["scaler_row"])
    date_to_position = {value: index for index, value in enumerate(pd.to_datetime(prepared_frame["ds"]))}
    cutoff_position = date_to_position[pd.Timestamp(cutoff)]
    target_position = date_to_position[pd.Timestamp(target_date)]
    actual_original = _frame_float(prepared_frame, target_position, "y")
    actual_value = output_value_for_scale(actual_original, scaler_row, output_scale)
    difference_order = int(candidate["difference_order"])

    if difference_order == 0:
        forecast_path = predicted_model_values.reset_index(drop=True)
        if output_scale == "original":
            forecast_path = inverse_scale_values(forecast_path, scaler_row)
        return float(forecast_path.iloc[-1]), actual_value

    if difference_order == 1:
        current_value = scale_original_value(_frame_float(prepared_frame, cutoff_position, "y"), scaler_row)
        forecast_differences = predicted_model_values.reset_index(drop=True)
        for predicted_difference in forecast_differences:
            current_value += float(predicted_difference)
        return output_reconstructed_scaled_value(current_value, scaler_row, output_scale), actual_value

    if difference_order == 2:
        if cutoff_position - 1 < 0:
            raise ValueError("Second-difference inverse forecast requires one observed value before the cutoff.")
        previous_2 = scale_original_value(_frame_float(prepared_frame, cutoff_position - 1, "y"), scaler_row)
        previous_1 = scale_original_value(_frame_float(prepared_frame, cutoff_position, "y"), scaler_row)
        forecast_second_differences = predicted_model_values.reset_index(drop=True)
        for predicted_second_difference in forecast_second_differences:
            next_value = float(predicted_second_difference) + 2 * previous_1 - previous_2
            previous_2, previous_1 = previous_1, next_value
        return output_reconstructed_scaled_value(previous_1, scaler_row, output_scale), actual_value

    raise ValueError(f"Unsupported difference order '{difference_order}'.")


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


def reconstruct_prepared_predictions(
    predicted_model_values: pd.Series,
    split_windows: WindowBundle,
    scaler_row: dict[str, Any],
    difference_order: int,
    output_scale: str = "original",
) -> pd.Series:
    previous_y_1 = split_windows["prev_y_1"]
    previous_y_2 = split_windows["prev_y_2"]
    if not isinstance(previous_y_1, pd.Series) or not isinstance(previous_y_2, pd.Series):
        raise TypeError("Prepared windows must store previous target values in pandas Series objects.")
    if difference_order == 0:
        forecast_scaled = predicted_model_values.reset_index(drop=True)
        return scaled_or_original_values(forecast_scaled, scaler_row, output_scale)
    if difference_order == 1:
        previous_scaled = scale_original_values(previous_y_1.reset_index(drop=True), scaler_row)
        forecast_scaled = previous_scaled + predicted_model_values.reset_index(drop=True)
        return scaled_or_original_values(forecast_scaled, scaler_row, output_scale)
    if difference_order == 2:
        previous_scaled_1 = scale_original_values(previous_y_1.reset_index(drop=True), scaler_row)
        previous_scaled_2 = scale_original_values(previous_y_2.reset_index(drop=True), scaler_row)
        forecast_scaled = predicted_model_values.reset_index(drop=True) + 2 * previous_scaled_1 - previous_scaled_2
        return scaled_or_original_values(forecast_scaled, scaler_row, output_scale)
    raise ValueError(f"Unsupported difference order '{difference_order}'.")


def inverse_prepared_predictions(
    predicted_model_values: pd.Series,
    split_windows: WindowBundle,
    scaler_row: dict[str, Any],
    difference_order: int,
    output_scale: str = "original",
) -> pd.Series:
    return reconstruct_prepared_predictions(
        predicted_model_values,
        split_windows,
        scaler_row,
        difference_order,
        output_scale,
    )


def actual_values_for_output_scale(
    split_windows: WindowBundle,
    scaler_row: dict[str, Any],
    output_scale: str,
) -> pd.Series:
    actual_values = split_windows["y_actual"]
    if not isinstance(actual_values, pd.Series):
        raise TypeError("Prepared windows must store actual values in a pandas Series object.")
    if output_scale == "scaled":
        return scale_original_values(actual_values.reset_index(drop=True), scaler_row)
    if output_scale == "original":
        return actual_values.reset_index(drop=True)
    raise ValueError(f"Unsupported forecast_output_scale '{output_scale}'.")


def output_value_for_scale(value: float, scaler_row: dict[str, Any], output_scale: str) -> float:
    if output_scale == "scaled":
        return scale_original_value(value, scaler_row)
    if output_scale == "original":
        return value
    raise ValueError(f"Unsupported forecast_output_scale '{output_scale}'.")


def output_reconstructed_scaled_value(value: float, scaler_row: dict[str, Any], output_scale: str) -> float:
    if output_scale == "scaled":
        return float(value)
    if output_scale == "original":
        return inverse_scale_value(value, scaler_row)
    raise ValueError(f"Unsupported forecast_output_scale '{output_scale}'.")


def scaled_or_original_values(values: pd.Series, scaler_row: dict[str, Any], output_scale: str) -> pd.Series:
    if output_scale == "scaled":
        return values.reset_index(drop=True)
    if output_scale == "original":
        return inverse_scale_values(values.reset_index(drop=True), scaler_row)
    raise ValueError(f"Unsupported forecast_output_scale '{output_scale}'.")


def inverse_scale_values(values: pd.Series, scaler: dict[str, Any]) -> pd.Series:
    if str(scaler["scaler"]) == "standard":
        return values * float(scaler["train_std"]) + float(scaler["train_mean"])
    if str(scaler["scaler"]) == "minmax":
        return values * float(scaler["train_scale"]) + float(scaler["train_min"])
    raise ValueError(f"Unsupported scaler '{scaler['scaler']}'.")


def inverse_scale_value(value: float, scaler: dict[str, Any]) -> float:
    return float(inverse_scale_values(pd.Series([value]), scaler).iloc[0])


def scale_original_values(values: pd.Series, scaler: dict[str, Any]) -> pd.Series:
    if str(scaler["scaler"]) == "standard":
        return (values - float(scaler["train_mean"])) / float(scaler["train_std"])
    if str(scaler["scaler"]) == "minmax":
        return (values - float(scaler["train_min"])) / float(scaler["train_scale"])
    raise ValueError(f"Unsupported scaler '{scaler['scaler']}'.")


def scale_original_value(value: float, scaler: dict[str, Any]) -> float:
    return float(scale_original_values(pd.Series([value]), scaler).iloc[0])


def select_hidden_units(lookback_steps: int, mlp_config: dict[str, Any]) -> int:
    strategy = str(mlp_config.get("hidden_units_strategy", "match_lookback"))
    if strategy == "match_lookback":
        return int(lookback_steps)
    return int(mlp_config.get("hidden_units", lookback_steps))


def select_hidden_layer_sizes(lookback_steps: int, mlp_config: dict[str, Any]) -> tuple[int, ...]:
    first_hidden_units = select_hidden_units(lookback_steps, mlp_config)
    return tuple(first_hidden_units for _ in range(max(1, int(mlp_config.get("num_layers", 1)))))


def format_learning_rate_alias(learning_rate_init: float) -> str:
    return f"{learning_rate_init:g}".replace("-", "m").replace(".", "p")


def limit_sequence(values: list[Any], configured_limit: Any) -> list[Any]:
    if configured_limit is None:
        return values
    limit = int(configured_limit)
    if limit <= 0:
        return values
    return values[:limit]


def format_duration(seconds: float) -> str:
    return str(timedelta(seconds=round(seconds)))


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
    seasonal_order_values = list(cast(list[int] | tuple[int, ...], arima_config.get("seasonal_order", [0, 0, 0])))
    if len(seasonal_order_values) != 3:
        raise ValueError("arima.seasonal_order must contain exactly three values: [P, D, Q].")
    seasonal_order: tuple[int, int, int] = (
        int(seasonal_order_values[0]),
        int(seasonal_order_values[1]),
        int(seasonal_order_values[2]),
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
        forecast = cast(pd.DataFrame, forecast_engine.forecast(df=train_frame, h=len(test_frame.index)))

    forecast = forecast[["ds", model_alias]].rename(columns={model_alias: "forecast"})
    evaluated = test_frame[["ds", "y"]].merge(forecast, on="ds", how="inner")
    if len(evaluated.index) != len(test_frame.index):
        raise ValueError(
            f"Forecast/test timestamp alignment failed for {model_alias}: "
            f"{len(evaluated.index)} aligned rows out of {len(test_frame.index)} test rows."
        )
    evaluated.attrs["warning_count"] = len(caught_warnings)
    evaluated.attrs["warnings"] = sorted({str(item.message) for item in caught_warnings})
    return evaluated


def forecast_exponential_smoothing(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    frequency: str,
    model_alias: str,
    smoothing_config: dict[str, Any],
    variant: dict[str, Any],
) -> pd.DataFrame:
    """Fit one exponential-smoothing variant and align predictions to holdout timestamps."""
    model_kind = str(variant.get("model", "simple"))
    if model_kind == "simple":
        model = SimpleExponentialSmoothingOptimized(alias=model_alias)
    elif model_kind == "holt":
        model = Holt(
            season_length=int(smoothing_config.get("season_length", 1)),
            error_type=str(variant.get("error_type", "A")),
            alias=model_alias,
        )
    elif model_kind == "auto_ets":
        model = AutoETS(
            season_length=int(smoothing_config.get("season_length", 1)),
            model=str(variant.get("ets_model", "ZZN")),
            damped=variant.get("damped"),
            alias=model_alias,
        )
    else:
        raise ValueError(f"Unsupported exponential smoothing model '{model_kind}'.")

    forecast_engine = StatsForecast(models=[model], freq=frequency, n_jobs=1)
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        forecast = cast(pd.DataFrame, forecast_engine.forecast(df=train_frame, h=len(test_frame.index)))

    forecast = forecast[["ds", model_alias]].rename(columns={model_alias: "forecast"})
    evaluated = test_frame[["ds", "y"]].merge(forecast, on="ds", how="inner")
    if len(evaluated.index) != len(test_frame.index):
        raise ValueError(
            f"Forecast/test timestamp alignment failed for {model_alias}: "
            f"{len(evaluated.index)} aligned rows out of {len(test_frame.index)} test rows."
        )
    evaluated.attrs["warning_count"] = len(caught_warnings)
    evaluated.attrs["warnings"] = sorted({str(item.message) for item in caught_warnings})
    return evaluated


def parse_arima_order(arima_config: dict[str, Any]) -> tuple[int, int, int]:
    order = cast(list[int] | tuple[int, int, int], arima_config.get("order", [1, 0, 1]))
    if len(order) != 3:
        raise ValueError("arima.order must contain exactly three values: [p, d, q].")
    return int(order[0]), int(order[1]), int(order[2])


def arima_orders_for_granularity(
    arima_config: dict[str, Any],
    granularity: str,
    lookback_steps: int,
) -> list[tuple[int, int, int]]:
    p_q_by_granularity = arima_config.get("p_q_values_by_granularity", {})
    if isinstance(p_q_by_granularity, dict) and granularity in p_q_by_granularity:
        p_q_values = [int(value) for value in p_q_by_granularity[granularity]]
    elif "p_q_values" in arima_config:
        p_q_values = [int(value) for value in arima_config["p_q_values"]]
    else:
        p_value, d_value, q_value = parse_arima_order(arima_config)
        return [(p_value, d_value, q_value)]

    del lookback_steps
    p_q_values = sorted({max(0, value) for value in p_q_values})
    d_by_granularity = arima_config.get("d_values_by_granularity", {})
    if isinstance(d_by_granularity, dict) and granularity in d_by_granularity:
        d_values = [int(value) for value in d_by_granularity[granularity]]
    else:
        d_values = [int(value) for value in arima_config.get("d_grid", [0, 1, 2])]
    return [(p_q, d_value, p_q) for p_q in p_q_values for d_value in d_values]


def build_arima_fit_plan(
    target_granularities: list[str],
    granularity_options: dict[str, str | None],
    resampling_policy: dict[str, Any],
    protocol: dict[str, Any],
    arima_config: dict[str, Any],
    lookback_duration: str,
) -> list[tuple[str, int, int, int]]:
    plan = []
    for granularity in target_granularities:
        frequency = statsforecast_frequency(granularity, granularity_options, resampling_policy)
        lookback_steps = steps_for_duration(lookback_duration, frequency)
        for p_value, d_value, q_value in arima_orders_for_granularity(arima_config, granularity, lookback_steps):
            plan.append((granularity, p_value, d_value, q_value))
    return plan


def build_generic_forecast_metric_row(forecast_frame: pd.DataFrame) -> dict[str, Any]:
    """Build metrics for model families that do not have ARIMA p/d/q metadata."""
    row = {
        "id_key": forecast_frame["id_key"].iloc[0],
        "granularity": forecast_frame["granularity"].iloc[0],
        "candidate_label": forecast_frame["candidate_label"].iloc[0],
        "model": forecast_frame["model"].iloc[0],
        "train_rows": int(forecast_frame["train_rows"].iloc[0]),
        "test_rows": int(forecast_frame["test_rows"].iloc[0]),
        "warning_count": int(forecast_frame.attrs.get("warning_count", 0)),
        "warnings": "; ".join(forecast_frame.attrs.get("warnings", [])),
        **forecast_metrics(forecast_frame["y"], forecast_frame["forecast"]),
    }
    return row


def exponential_smoothing_variants(smoothing_config: dict[str, Any]) -> list[dict[str, Any]]:
    variants = smoothing_config.get("variants", [])
    if not isinstance(variants, list) or not variants:
        return [
            {"name": "ses_level", "model": "simple"},
            {"name": "holt_trend", "model": "holt"},
            {"name": "auto_ets_nonseasonal", "model": "auto_ets", "ets_model": "ZZN"},
        ]
    return [dict(variant) for variant in variants if isinstance(variant, dict)]


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
    candidate_label: str | None = None,
) -> pd.DataFrame:
    forecast_frame["id_key"] = id_key
    forecast_frame["granularity"] = granularity
    forecast_frame["model"] = model_alias
    forecast_frame["p"] = p_value
    forecast_frame["d"] = d_value
    forecast_frame["q"] = q_value
    forecast_frame["train_rows"] = train_rows
    forecast_frame["test_rows"] = test_rows
    if candidate_label is not None:
        forecast_frame["candidate_label"] = candidate_label
    return forecast_frame


def apply_direct_forecast_output_scale(
    forecast_frame: pd.DataFrame,
    train_frame: pd.DataFrame,
    output_scale: str,
    scaling_method: str,
) -> pd.DataFrame:
    """Scale direct target-level forecasts for reporting without changing model fitting."""
    if output_scale == "original":
        scaled = forecast_frame.copy()
        scaled.attrs = dict(forecast_frame.attrs)
        scaled["forecast_output_scale"] = output_scale
        return scaled
    if output_scale != "scaled":
        raise ValueError(f"Unsupported forecast_output_scale '{output_scale}'.")

    scaler = fit_linear_scaler(train_frame["y"], scaling_method)
    scaled = forecast_frame.copy()
    scaled.attrs = dict(forecast_frame.attrs)
    scaled["y"] = apply_linear_scaler(scaled["y"], scaler)
    scaled["forecast"] = apply_linear_scaler(scaled["forecast"], scaler)
    scaled["forecast_output_scale"] = output_scale
    scaled["forecast_output_scaler"] = scaling_method
    return scaled


def write_csv_output(frame: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_csv(temp_path, index=False)
    temp_path.replace(output_path)


def _frame_float(frame: pd.DataFrame, row_index: int, column: str) -> float:
    value = cast(Any, frame.at[row_index, column])
    if pd.isna(value):
        return float("nan")
    return float(value)
