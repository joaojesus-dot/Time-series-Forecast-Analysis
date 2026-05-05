from __future__ import annotations

import argparse
import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Spec-driven plotting
# ---------------------------------------------------------------------------
# Render the configured plot specifications into image files.

def write_plot_specs(plot_specs: list[dict[str, Any]], frames: dict[str, pd.DataFrame]) -> None:
    handlers = {
        "time_series_groups": _plot_time_series_groups,
        "stacked_time_series": _plot_stacked_time_series,
        "target_granularity_comparison": _plot_target_granularity_comparison,
        "heatmap": _plot_heatmap_from_spec,
        "target_transform_plots": _plot_target_transform_plots,
        "smoothing_std_summary": _plot_smoothing_std_summary,
        "scaling_std_summary": _plot_scaling_std_summary,
        "target_scaling_plots": _plot_target_scaling_plots,
    }

    for spec in plot_specs:
        kind = str(spec["kind"])
        handlers[kind](spec, frames)


# ---------------------------------------------------------------------------
# Time-series plots
# ---------------------------------------------------------------------------
# Draw the configured time-series panels and grouped sensor plots.

def _plot_time_series_groups(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    frame = frames[str(spec["frame"])]
    output_dir = Path(spec["output_dir"])
    file_prefix = str(spec["file_prefix"])
    timestamp_column = str(spec.get("timestamp_column", "date"))
    groups = spec["groups"]
    units = spec.get("units", {})

    for group_name, columns in groups.items():
        available_columns = [column for column in columns if column in frame.columns]
        if not available_columns:
            continue

        fig, axis = plt.subplots(figsize=(14, 5))
        for column in available_columns:
            axis.plot(frame[timestamp_column], frame[column], linewidth=1.0, label=column)
        axis.set_title(f"{file_prefix} - {str(group_name).replace('_', ' ').title()}")
        axis.set_xlabel(timestamp_column)
        axis.set_ylabel(units.get(group_name, "Sensor value"))
        axis.legend(loc="upper right", fontsize=8)
        _save_figure(fig, output_dir / f"{file_prefix}_{group_name}.png")


def _plot_stacked_time_series(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    frame = frames[str(spec["frame"])]
    panels = spec["panels"]
    timestamp_column = str(spec.get("timestamp_column", "date"))
    fig, axes = plt.subplots(len(panels), 1, figsize=spec.get("figsize", (14, 7)), sharex=True)
    axes = list(axes) if len(panels) > 1 else [axes]

    for axis, panel in zip(axes, panels):
        axis.plot(
            pd.to_datetime(frame[timestamp_column]),
            frame[panel["column"]],
            color=panel.get("color", "#3366aa"),
            linewidth=panel.get("linewidth", 1),
        )
        axis.set_title(panel["title"])
        axis.set_ylabel(panel["ylabel"])

    axes[-1].set_xlabel(timestamp_column)
    _save_figure(fig, Path(spec["output_path"]))


def _plot_target_granularity_comparison(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    dataset_keys = spec["dataset_keys"]
    target_column = str(spec["target_column"])
    timestamp_column = str(spec.get("timestamp_column", "date"))
    plot_count = len(dataset_keys)
    if plot_count == 0:
        return
    fig, axes = plt.subplots(plot_count, 1, figsize=(14, 3.5 * plot_count), sharey=True)
    axes = list(axes) if plot_count > 1 else [axes]

    for axis, frame_key in zip(axes, dataset_keys):
        frame = frames[frame_key]
        axis.plot(frame[timestamp_column], frame[target_column], linewidth=0.9)
        axis.set_title(frame_key.replace("subset_B_", ""))
        axis.set_xlabel(timestamp_column)
        axis.set_ylabel(spec.get("ylabel", "Target"))

    fig.suptitle(spec["title"])
    _save_figure(fig, Path(spec["output_path"]))


# ---------------------------------------------------------------------------
# Correlation and distribution plots
# ---------------------------------------------------------------------------
# Draw the exploratory correlation heatmap used in the pre-forecasting review.

def _plot_heatmap_from_spec(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    frame = frames[str(spec["frame"])]
    sample_step = int(spec.get("sample_step", 1))
    columns = spec.get("columns")

    if sample_step > 1:
        frame = frame.iloc[::sample_step]
    if columns:
        frame = frame[[column for column in columns if column in frame.columns]]
    frame = frame.drop(columns=spec.get("drop_columns", []), errors="ignore")

    fig, axis = plt.subplots(figsize=spec.get("figsize", (12, 10)))
    sns.heatmap(frame.corr(numeric_only=True), cmap="coolwarm", center=0, ax=axis)
    axis.set_title(spec["title"])
    _save_figure(fig, Path(spec["output_path"]))


# ---------------------------------------------------------------------------
# Target preprocessing plots
# ---------------------------------------------------------------------------
# Draw the cached target transforms used in the preparation review.

def _plot_target_transform_plots(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    del frames
    output_dir = Path(spec["output_dir"])
    transform_series = spec["transform_series"]
    by_dataset = {}

    for item in transform_series:
        by_dataset.setdefault(item["id_key"], []).append(item)

    for id_key, records in by_dataset.items():
        original = next(record for record in records if record["transform"] == "original")
        smoothed = [record for record in records if record["transform"] == "trailing_rolling_mean"]
        first_difference = next(record for record in records if record["transform"] == "first_difference")
        second_difference = next(record for record in records if record["transform"] == "second_difference")

        _plot_smoothing_comparison(
            id_key,
            original,
            smoothed,
            output_dir,
            str(spec.get("target_ylabel", "Target")),
            str(spec.get("zoom_duration", "6h")),
        )
        _plot_first_difference(id_key, first_difference, output_dir)
        _plot_difference_comparison(id_key, original, first_difference, second_difference, output_dir)


def _plot_smoothing_std_summary(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    frame = frames[str(spec["frame"])].copy()
    frame["transform_label"] = frame["transform"] + " (" + frame["window"] + ")"
    level_frame = frame[frame["transform"] != "first_difference"]
    differenced_frame = frame[frame["transform"].isin(["first_difference", "second_difference"])]

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    sns.barplot(data=level_frame, x="granularity", y="std", hue="transform_label", ax=axes[0])
    axes[0].set_title("Level Target Std After Aggregation And Smoothing")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Level std")
    axes[0].legend(loc="upper right", fontsize=8)

    sns.barplot(data=differenced_frame, x="granularity", y="std", hue="transform", ax=axes[1])
    axes[1].set_title("Difference Std By Granularity")
    axes[1].set_xlabel("Granularity")
    axes[1].set_ylabel("Delta std")
    axes[1].legend(loc="upper right", fontsize=8)
    _save_figure(fig, Path(spec["output_path"]))


def _plot_scaling_std_summary(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    frame = frames[str(spec["frame"])].copy()
    frame = frame[frame["transform"].isin(["scaled_level", "scaled_first_difference", "scaled_second_difference"])]
    if frame.empty:
        return

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for axis, transform in zip(axes, ["scaled_level", "scaled_first_difference", "scaled_second_difference"]):
        transform_frame = frame[frame["transform"] == transform]
        sns.barplot(data=transform_frame, x="granularity", y="std", hue="scaling_method", ax=axis)
        axis.set_title(transform.replace("_", " ").title())
        axis.set_xlabel("")
        axis.set_ylabel("Std")
        axis.legend(loc="upper right", fontsize=8, title="Scaler")
    axes[-1].set_xlabel("Granularity")
    _save_figure(fig, Path(spec["output_path"]))


def _plot_target_scaling_plots(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    del frames
    output_dir = Path(spec["output_dir"])
    scaling_series = spec["scaling_series"]
    by_dataset = {}

    for item in scaling_series:
        by_dataset.setdefault(item["id_key"], []).append(item)

    for id_key, records in by_dataset.items():
        level_records = [record for record in records if record["transform"] == "scaled_level"]
        first_records = [record for record in records if record["transform"] == "scaled_first_difference"]
        second_records = [record for record in records if record["transform"] == "scaled_second_difference"]
        if not level_records:
            continue

        fig, axes = plt.subplots(2, 2, figsize=(14, 8))
        for record in level_records:
            dates = pd.to_datetime(record["dates"]).reset_index(drop=True)
            values = pd.Series(record["values"]).reset_index(drop=True)
            mask = _first_duration_mask(dates, str(spec.get("zoom_duration", "6h")))
            axes[0, 0].plot(
                dates[mask],
                values[mask],
                linewidth=1.0,
                label=str(record["scaling_method"]),
            )
            axes[0, 1].hist(values.dropna(), bins=80, alpha=0.45, label=str(record["scaling_method"]))

        for record in first_records:
            values = pd.Series(record["values"]).dropna()
            axes[1, 0].hist(values, bins=80, alpha=0.45, label=str(record["scaling_method"]))

        for record in second_records:
            values = pd.Series(record["values"]).dropna()
            axes[1, 1].hist(values, bins=80, alpha=0.45, label=str(record["scaling_method"]))

        axes[0, 0].set_title("Scaled Target Zoom")
        axes[0, 0].set_xlabel("date")
        axes[0, 0].set_ylabel("Scaled target")
        axes[0, 1].set_title("Scaled Target Distribution")
        axes[0, 1].set_xlabel("Scaled target")
        axes[1, 0].set_title("Scaled First Difference Distribution")
        axes[1, 0].set_xlabel("Scaled delta 1")
        axes[1, 1].set_title("Scaled Second Difference Distribution")
        axes[1, 1].set_xlabel("Scaled delta 2")

        for axis in axes.ravel():
            axis.legend(loc="upper right", fontsize=8, title="Scaler")
        _save_figure(fig, output_dir / f"{id_key}_target_scaling_comparison.png")


def _first_duration_mask(dates: pd.Series, duration: str) -> pd.Series:
    if dates.empty:
        return pd.Series(dtype=bool)
    end_date = dates.min() + pd.Timedelta(duration)
    return dates <= end_date


def _plot_smoothing_comparison(
    id_key: str,
    original: dict[str, Any],
    smoothed: list[dict[str, Any]],
    output_dir: Path,
    ylabel: str,
    zoom_duration: str,
) -> None:
    original_dates = pd.to_datetime(original["dates"]).reset_index(drop=True)
    original_values = pd.Series(original["values"]).reset_index(drop=True)
    smoothed_values = [
        {
            **item,
            "dates": pd.to_datetime(item["dates"]).reset_index(drop=True),
            "values": pd.Series(item["values"]).reset_index(drop=True),
        }
        for item in smoothed
    ]
    zoom_mask = _build_smoothing_zoom_mask(original_dates, original_values, smoothed_values, zoom_duration)

    fig, axes = plt.subplots(2, 1, figsize=(14, 7), sharex=True)
    axes[0].plot(original_dates[zoom_mask], original_values[zoom_mask], linewidth=0.9, alpha=0.65, label="original")
    for item in smoothed:
        item_values = pd.Series(item["values"]).reset_index(drop=True)
        item_dates = pd.to_datetime(item["dates"]).reset_index(drop=True)
        axes[0].plot(
            item_dates[zoom_mask],
            item_values[zoom_mask],
            linewidth=1.2,
            label=f"rolling mean {item['window']}",
        )
        axes[1].plot(
            item_dates[zoom_mask],
            (original_values - item_values)[zoom_mask],
            linewidth=1.1,
            label=f"original - {item['window']}",
        )

    axes[0].set_title(f"{id_key} - Target Smoothing Zoom")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Smoothing Residual")
    axes[1].set_xlabel("date")
    axes[1].set_ylabel("Temp delta")
    if smoothed:
        axes[1].legend(loc="upper right", fontsize=8)
    else:
        axes[1].text(0.5, 0.5, "No smoothing window exceeds this cadence.", ha="center", va="center")
    _save_figure(fig, output_dir / f"{id_key}_target_smoothing_comparison.png")


def _plot_first_difference(id_key: str, first_difference: dict[str, Any], output_dir: Path) -> None:
    dates = pd.to_datetime(first_difference["dates"]).reset_index(drop=True)
    values = pd.Series(first_difference["values"]).reset_index(drop=True)
    clean_values = values.dropna()
    rolling_steps = _steps_for_duration(dates, "1h")
    rolling_abs = values.abs().rolling(window=rolling_steps, min_periods=max(1, rolling_steps // 4)).mean()

    fig, axes = plt.subplots(2, 1, figsize=(14, 7))
    axes[0].hist(clean_values, bins=80, color="#4c72b0", alpha=0.85)
    axes[0].axvline(0, color="#333333", linewidth=0.8)
    axes[0].set_title(f"{id_key} - Target First-Difference Distribution")
    axes[0].set_xlabel("Delta temp")
    axes[0].set_ylabel("Frequency")

    axes[1].plot(dates, rolling_abs, linewidth=1.1, color="#4c72b0")
    axes[1].set_title("Rolling 1h Mean Absolute First Difference")
    axes[1].set_xlabel("date")
    axes[1].set_ylabel("Mean abs delta")
    _save_figure(fig, output_dir / f"{id_key}_target_first_difference.png")


def _plot_difference_comparison(
    id_key: str,
    original: dict[str, Any],
    first_difference: dict[str, Any],
    second_difference: dict[str, Any],
    output_dir: Path,
) -> None:
    original_dates = pd.to_datetime(original["dates"]).reset_index(drop=True)
    original_values = pd.Series(original["values"]).reset_index(drop=True)
    first_values = pd.Series(first_difference["values"]).reset_index(drop=True)
    second_values = pd.Series(second_difference["values"]).reset_index(drop=True)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].plot(original_dates, original_values, linewidth=0.9, color="#1f4b99")
    axes[0].set_title(f"{id_key} - Original Target")
    axes[0].set_ylabel("Target")

    axes[1].plot(original_dates, first_values, linewidth=0.9, color="#4c72b0")
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("First Difference")
    axes[1].set_ylabel("Delta 1")

    axes[2].plot(original_dates, second_values, linewidth=0.9, color="#55a868")
    axes[2].axhline(0, color="#333333", linewidth=0.8)
    axes[2].set_title("Second Difference")
    axes[2].set_xlabel("date")
    axes[2].set_ylabel("Delta 2")
    _save_figure(fig, output_dir / f"{id_key}_target_difference_comparison.png")


def _build_smoothing_zoom_mask(
    dates: pd.Series,
    original_values: pd.Series,
    smoothed_values: list[dict[str, Any]],
    zoom_duration: str,
) -> pd.Series:
    duration = pd.Timedelta(zoom_duration)
    reference = next((item for item in reversed(smoothed_values) if item["values"].notna().any()), None)
    if reference is None or dates.empty:
        return pd.Series([True] * len(dates))

    residual = (original_values - reference["values"]).abs()
    center_position = int(residual.idxmax()) if residual.notna().any() else len(dates) // 2
    center_date = dates.iloc[center_position]
    start_date = max(dates.min(), center_date - duration / 2)
    end_date = start_date + duration
    if end_date > dates.max():
        end_date = dates.max()
        start_date = max(dates.min(), end_date - duration)
    return (dates >= start_date) & (dates <= end_date)


def _steps_for_duration(dates: pd.Series, duration: str) -> int:
    diffs = dates.diff().dropna().dt.total_seconds()
    if diffs.empty:
        return 1
    seconds_per_step = max(float(diffs.median()), 1.0)
    return max(1, round(pd.Timedelta(duration).total_seconds() / seconds_per_step))


# ---------------------------------------------------------------------------
# Forecasting plots
# ---------------------------------------------------------------------------
# Draw the forecasting comparison plots from the evaluated forecast frames.

def write_univariate_comparison_plots(
    arima_metrics: pd.DataFrame,
    mlp_test_comparison: pd.DataFrame,
    arima_forecasts: pd.DataFrame,
    mlp_forecasts: pd.DataFrame,
    output_dir: Path,
    exponential_smoothing_metrics: pd.DataFrame | None = None,
    exponential_smoothing_forecasts: pd.DataFrame | None = None,
) -> None:
    """Write human-facing comparison plots for univariate model review."""
    metric_frames = []
    if not arima_metrics.empty:
        metric_frames.append(arima_metrics.assign(model_family="ARIMA", split="test"))
    if exponential_smoothing_metrics is not None and not exponential_smoothing_metrics.empty:
        metric_frames.append(exponential_smoothing_metrics.assign(model_family="Exponential Smoothing", split="test"))
    if not mlp_test_comparison.empty and "split" in mlp_test_comparison.columns:
        test_mlp_metrics = mlp_test_comparison[mlp_test_comparison["split"] == "test"].copy()
        if not test_mlp_metrics.empty:
            metric_frames.append(test_mlp_metrics.assign(model_family="MLP"))
    if metric_frames:
        comparable_metrics = pd.concat(metric_frames, ignore_index=True)
        write_univariate_metric_comparison(comparable_metrics, output_dir / "univariate_metric_comparison.png")
    write_univariate_forecast_overlays(arima_forecasts, mlp_forecasts, output_dir)
    del exponential_smoothing_forecasts


def write_univariate_metric_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    if metrics.empty or "r2" not in metrics.columns:
        return
    plot_frame = metrics.copy()
    plot_frame = plot_frame[pd.notna(plot_frame["r2"])]
    if plot_frame.empty:
        return
    if "candidate_label" not in plot_frame.columns:
        plot_frame["candidate_label"] = plot_frame.get("granularity", "candidate").astype(str)
    if "model_variant" not in plot_frame.columns:
        plot_frame["model_variant"] = plot_frame.get("model", plot_frame["model_family"]).astype(str)
    elif "model" in plot_frame.columns:
        plot_frame["model_variant"] = plot_frame["model_variant"].where(
            plot_frame["model_variant"].notna(),
            plot_frame["model"],
        )
    plot_frame["comparison_label"] = (
        plot_frame["model_family"].astype(str) + " | " + plot_frame["model_variant"].astype(str)
    )
    plot_frame = plot_frame.sort_values(["candidate_label", "comparison_label"])

    height = max(5, 0.38 * plot_frame["candidate_label"].nunique() + 2)
    fig, axis = plt.subplots(figsize=(15, height))
    sns.barplot(
        data=plot_frame,
        y="candidate_label",
        x="r2",
        hue="comparison_label",
        ax=axis,
        orient="h",
    )
    axis.set_title("Univariate Test R2 By Candidate")
    axis.set_xlabel("R2")
    axis.set_ylabel("Candidate")
    axis.legend(loc="best", fontsize=8, title="Model")
    _save_figure(fig, output_path)


def write_univariate_forecast_overlays(
    arima_forecasts: pd.DataFrame,
    mlp_forecasts: pd.DataFrame,
    output_dir: Path,
    max_points: int = 600,
) -> None:
    if mlp_forecasts.empty or "granularity" not in mlp_forecasts.columns or "split" not in mlp_forecasts.columns:
        return
    if arima_forecasts.empty or "granularity" not in arima_forecasts.columns:
        return

    for granularity in sorted(set(mlp_forecasts["granularity"].dropna())):
        mlp_test = mlp_forecasts[
            (mlp_forecasts["granularity"] == granularity) & (mlp_forecasts["split"] == "test")
        ].copy()
        arima_test = arima_forecasts[arima_forecasts["granularity"] == granularity].copy()
        if mlp_test.empty or arima_test.empty:
            continue

        plot_dates = mlp_test["ds"].head(max_points)
        arima_zoom = arima_test[arima_test["ds"].isin(plot_dates)]
        mlp_zoom = mlp_test.head(max_points)
        output_scale = first_forecast_output_scale(mlp_zoom, arima_zoom)
        y_label = "Scaled target" if output_scale == "scaled" else "Target"

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(mlp_zoom["ds"], mlp_zoom["y"], color="#222222", linewidth=1.1, label="actual")
        axes[0].plot(arima_zoom["ds"], arima_zoom["forecast"], linewidth=1.0, label="ARIMA")
        axes[0].plot(mlp_zoom["ds"], mlp_zoom["forecast"], linewidth=1.0, label="MLP")
        axes[0].set_title(f"{granularity} - Test Forecast Comparison")
        axes[0].set_ylabel(y_label)
        axes[0].legend(loc="upper right", fontsize=8)

        if not arima_zoom.empty:
            axes[1].plot(arima_zoom["ds"], arima_zoom["forecast"] - arima_zoom["y"], linewidth=1.0, label="ARIMA")
        axes[1].plot(mlp_zoom["ds"], mlp_zoom["forecast"] - mlp_zoom["y"], linewidth=1.0, label="MLP")
        axes[1].axhline(0, color="#333333", linewidth=0.8)
        axes[1].set_title("Forecast Error")
        axes[1].set_xlabel("date")
        axes[1].set_ylabel(f"Forecast - actual ({output_scale})")
        axes[1].legend(loc="upper right", fontsize=8)
        _save_figure(fig, output_dir / f"{granularity}_univariate_forecast_overlay.png")


def first_forecast_output_scale(*frames: pd.DataFrame) -> str:
    for frame in frames:
        if "forecast_output_scale" not in frame.columns:
            continue
        values = frame["forecast_output_scale"].dropna()
        if not values.empty:
            return str(values.iloc[0])
    return "original"


def write_historical_model_metric_plots(history_frame: pd.DataFrame, output_dir: Path) -> None:
    if history_frame.empty:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    test_history = history_frame[history_frame["split"] == "test"].copy()
    if test_history.empty:
        return

    write_historical_metric_leaderboard(test_history, output_dir / "historical_test_metric_leaderboard.png")
    for model_key, model_frame in test_history.groupby("model_key"):
        model_key_text = str(model_key)
        write_model_history_plot(model_key_text, model_frame, output_dir / f"{model_key_text}_test_history.png")


def write_historical_metric_leaderboard(history_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for axis, metric in zip(axes, ["mae", "rmse"]):
        sns.barplot(
            data=history_frame,
            x="run_label",
            y=metric,
            hue="model_label",
            ax=axis,
        )
        axis.set_title(f"Historical Test {metric.upper()} By Run")
        axis.set_xlabel("Run")
        axis.set_ylabel(metric.upper())
        axis.tick_params(axis="x", rotation=45)
        axis.legend(loc="upper right", fontsize=8)
    _save_figure(fig, output_path)


def write_target_profiling_plots(
    datasets: dict[str, pd.DataFrame],
    target_column: str,
    output_dir: Path,
    timestamp_column: str = "date",
    max_lags: int = 120,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for id_key, dataset in datasets.items():
        series = dataset[target_column].dropna().reset_index(drop=True)
        if len(series) < 20:
            continue
        dates = pd.to_datetime(dataset[timestamp_column], errors="coerce").reset_index(drop=True)
        rolling_window = max(5, min(len(series) // 10, 120))

        fig, axes = plt.subplots(2, 2, figsize=(14, 9))
        axes = axes.ravel()
        axes[0].plot(dates, series, linewidth=0.9, color="#1f4b99")
        axes[0].set_title(f"{id_key} - Target Over Time")
        axes[0].set_xlabel(timestamp_column)
        axes[0].set_ylabel(target_column)

        rolling_mean = series.rolling(window=rolling_window, min_periods=rolling_window).mean()
        rolling_std = series.rolling(window=rolling_window, min_periods=rolling_window).std()
        axes[1].plot(dates, series, linewidth=0.8, alpha=0.45, label="target")
        axes[1].plot(dates, rolling_mean, linewidth=1.1, label=f"rolling mean ({rolling_window})")
        axes[1].plot(dates, rolling_std, linewidth=1.1, label=f"rolling std ({rolling_window})")
        axes[1].set_title("Rolling Mean And Std")
        axes[1].set_xlabel(timestamp_column)
        axes[1].legend(loc="upper right", fontsize=8)

        plot_acf(series, lags=min(max_lags, len(series) - 2), ax=axes[2], title="Autocorrelation")
        plot_pacf(series, lags=min(max_lags, len(series) // 2 - 1, 60), ax=axes[3], title="Partial Autocorrelation", method="ywm")
        _save_figure(fig, output_dir / f"{id_key}_target_profile.png")


def write_target_correlation_barplots(
    correlation_summary: pd.DataFrame,
    output_dir: Path,
    top_n: int = 8,
) -> None:
    if correlation_summary.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    for granularity, frame in correlation_summary.groupby("granularity"):
        top_frame = frame.sort_values("abs_correlation", ascending=False).head(top_n)
        fig, axis = plt.subplots(figsize=(12, 5))
        sns.barplot(data=top_frame, x="variable", y="correlation", ax=axis, color="#1f4b99")
        axis.set_title(f"{granularity} - Top Target Correlations")
        axis.set_xlabel("Variable")
        axis.set_ylabel("Correlation With Target")
        axis.tick_params(axis="x", rotation=60)
        _save_figure(fig, output_dir / f"{granularity}_target_correlation_barplot.png")


def write_test_comparison_metric_plots(
    test_results: pd.DataFrame,
    output_dir: Path,
) -> None:
    if test_results.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    if "model_variant" in test_results.columns and test_results["model_variant"].nunique() > 1:
        ordered = test_results.sort_values(["granularity", "candidate_label", "model_variant"]).copy()
        ordered["variant_short"] = ordered["model_variant"].astype(str).str.replace("neuralforecast_2_hidden_", "", regex=False)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        for axis, metric in zip(axes, ["mae", "rmse", "r2"]):
            sns.barplot(data=ordered, x="granularity", y=metric, hue="variant_short", ax=axis)
            axis.set_title(f"Stage Model Variants - {metric.upper()}")
            axis.set_xlabel("Granularity")
            axis.set_ylabel(metric.upper())
            axis.legend(loc="best", fontsize=8, title="MLP variant")
        _save_figure(fig, output_dir / "mlp_variant_metric_comparison.png")
        return
    for granularity, frame in test_results.groupby("granularity"):
        ordered = frame.sort_values(["difference_order", "training_smoothing_window", "learning_rate_init"]).copy()
        ordered["candidate_short"] = ordered["candidate_label"].str.replace(f"{granularity}_", "", regex=False)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))
        for axis, metric in zip(axes, ["mae", "rmse", "r2"]):
            sns.barplot(data=ordered, x="candidate_short", y=metric, hue="learning_rate_init", ax=axis)
            axis.set_title(f"{granularity} - Test Comparison {metric.upper()}")
            axis.set_xlabel("Candidate")
            axis.set_ylabel(metric.upper())
            axis.tick_params(axis="x", rotation=70)
            axis.legend(loc="upper right", fontsize=8, title="LR")
        _save_figure(fig, output_dir / f"{granularity}_test_comparison_metrics.png")


def write_test_comparison_plots(
    candidate_forecasts: pd.DataFrame,
    output_dir: Path,
    max_points: int = 600,
) -> None:
    if candidate_forecasts.empty:
        return
    required_columns = {"granularity", "candidate_label", "ds", "y", "forecast"}
    if not required_columns.issubset(candidate_forecasts.columns):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    group_columns = ["granularity", "candidate_label"]
    if "model_variant" in candidate_forecasts.columns:
        group_columns.append("model_variant")
    if "learning_rate_init" in candidate_forecasts.columns:
        group_columns.append("learning_rate_init")

    for group_values, frame in candidate_forecasts.groupby(group_columns):
        group_data = dict(zip(group_columns, group_values if isinstance(group_values, tuple) else (group_values,)))
        granularity = group_data["granularity"]
        candidate_label = group_data["candidate_label"]
        model_variant = str(group_data.get("model_variant", ""))
        learning_rate_init = group_data.get("learning_rate_init")
        learning_rate_label = f"lr={learning_rate_init:g}" if learning_rate_init is not None else ""
        plot_frame = frame.sort_values("ds").head(max_points).copy()
        if plot_frame.empty:
            continue
        output_scale = (
            str(plot_frame["forecast_output_scale"].iloc[0])
            if "forecast_output_scale" in plot_frame.columns
            else "original"
        )
        y_label = "Scaled target" if output_scale == "scaled" else "Target"

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(plot_frame["ds"], plot_frame["y"], color="#222222", linewidth=1.1, label="actual")
        axes[0].plot(plot_frame["ds"], plot_frame["forecast"], color="#1f4b99", linewidth=1.0, label="forecast")
        title_parts = [part for part in [model_variant, learning_rate_label] if part]
        title_suffix = f" - {' - '.join(title_parts)}" if title_parts else ""
        axes[0].set_title(f"{granularity} - {candidate_label}{title_suffix} Test Comparison")
        axes[0].set_ylabel(y_label)
        axes[0].legend(loc="upper right", fontsize=8)

        axes[1].plot(
            plot_frame["ds"],
            plot_frame["forecast"] - plot_frame["y"],
            color="#b24d2b",
            linewidth=1.0,
            label="forecast - actual",
        )
        axes[1].axhline(0, color="#333333", linewidth=0.8)
        axes[1].set_title("Test Comparison Error")
        axes[1].set_xlabel("date")
        axes[1].set_ylabel(f"Forecast - actual ({output_scale})")
        axes[1].legend(loc="upper right", fontsize=8)

        safe_label = str(candidate_label).replace("/", "-").replace("\\", "-")
        safe_variant = _safe_filename(model_variant) if model_variant else ""
        safe_lr = learning_rate_label.replace("=", "_").replace(".", "p")
        filename_parts = [safe_label, safe_variant, safe_lr, "test_comparison"]
        filename = "_".join(part for part in filename_parts if part) + ".png"
        _save_figure(fig, output_dir / str(granularity) / filename)


def write_mlp_training_diagnostic_plots(
    training_history: pd.DataFrame,
    test_comparison: pd.DataFrame,
    output_dir: Path,
    max_points: int = 1200,
) -> None:
    if training_history.empty:
        return
    required_columns = {"granularity", "candidate_label", "learning_rate_init", "step", "train_loss"}
    if not required_columns.issubset(training_history.columns):
        return

    write_mlp_grouped_training_curves(training_history, output_dir / "loss_curves", "loss", max_points=max_points)
    write_mlp_grouped_training_curves(training_history, output_dir / "learning_curves", "learning", max_points=max_points)
    write_mlp_convergence_line_plots(training_history, output_dir / "convergence_by_candidate", max_points=max_points)
    write_mlp_training_summary_plot(training_history, test_comparison, output_dir / "training_summary.png")


def write_mlp_grouped_training_curves(
    training_history: pd.DataFrame,
    output_dir: Path,
    curve_type: str,
    max_points: int = 1200,
) -> None:
    required_columns = {"granularity", "candidate_label", "learning_rate_init", "step", "train_loss"}
    if training_history.empty or not required_columns.issubset(training_history.columns):
        return
    output_dir.mkdir(parents=True, exist_ok=True)

    group_columns = ["granularity", "candidate_label"]
    if "model_variant" in training_history.columns:
        group_columns.append("model_variant")
    group_columns.append("learning_rate_init")

    plot_frame = training_history.copy()
    if "model_variant" not in plot_frame.columns:
        plot_frame["model_variant"] = "mlp"
    if curve_type == "learning":
        if "best_train_loss" in plot_frame.columns:
            plot_frame["curve_value"] = plot_frame["best_train_loss"]
        else:
            plot_frame["curve_value"] = plot_frame.sort_values("step").groupby(group_columns)["train_loss"].cummin()
        y_label = "Best loss so far"
        title_suffix = "Learning Curves"
        filename_suffix = "learning_curves"
    else:
        plot_frame["curve_value"] = plot_frame.get("train_loss_rolling", plot_frame["train_loss"])
        y_label = "Rolling train loss"
        title_suffix = "Loss Curves"
        filename_suffix = "loss_curves"

    plot_frame["curve_label"] = (
        plot_frame["model_variant"].astype(str)
        + " | lr="
        + plot_frame["learning_rate_init"].map(lambda value: f"{float(value):g}")
    )
    for granularity, frame in plot_frame.groupby("granularity"):
        curve_frames = [
            _downsample_by_step(curve_frame.sort_values("step"), max_points)
            for _curve_label, curve_frame in frame.groupby(["candidate_label", "curve_label"])
        ]
        granularity_frame = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
        if granularity_frame.empty:
            continue
        fig, axis = plt.subplots(figsize=(14, 6))
        sns.lineplot(
            data=granularity_frame,
            x="step",
            y="curve_value",
            hue="curve_label",
            style="candidate_label",
            ax=axis,
            linewidth=1.2,
        )
        axis.set_title(f"{granularity} MLP {title_suffix}")
        axis.set_xlabel("Training step")
        axis.set_ylabel(y_label)
        axis.legend(loc="best", fontsize=8, title="Configuration")
        _save_figure(fig, output_dir / f"{granularity}_{filename_suffix}.png")


def write_mlp_convergence_line_plots(
    training_history: pd.DataFrame,
    output_dir: Path,
    max_points: int = 1200,
) -> None:
    if training_history.empty:
        return
    required_columns = {"granularity", "candidate_label", "learning_rate_init", "step", "train_loss"}
    if not required_columns.issubset(training_history.columns):
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    plot_frame = training_history.copy()
    plot_frame["loss_for_plot"] = plot_frame.get("train_loss_rolling", plot_frame["train_loss"])
    if "model_variant" not in plot_frame.columns:
        plot_frame["model_variant"] = "mlp"
    plot_frame["curve_label"] = (
        "layers="
        + plot_frame.get("num_layers", plot_frame["model_variant"]).astype(str)
        + " | lr="
        + plot_frame["learning_rate_init"].map(lambda value: f"{float(value):g}")
    )

    groups = list(plot_frame.groupby(["granularity", "candidate_label"]))
    if not groups:
        return
    fig, axes = plt.subplots(len(groups), 1, figsize=(14, max(4, 3.8 * len(groups))), sharex=False)
    if len(groups) == 1:
        axes = [axes]

    for axis, ((granularity, candidate_label), frame) in zip(axes, groups):
        curve_frames = [
            _downsample_by_step(curve_frame.sort_values("step"), max_points)
            for _curve_label, curve_frame in frame.groupby("curve_label")
        ]
        candidate_frame = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
        if candidate_frame.empty:
            continue

        sns.lineplot(
            data=candidate_frame,
            x="step",
            y="loss_for_plot",
            hue="curve_label",
            ax=axis,
            linewidth=1.2,
        )
        axis.set_title(f"{granularity} - {candidate_label} Training Convergence")
        axis.set_xlabel("Training step")
        axis.set_ylabel("Rolling train loss")
        axis.legend(loc="best", fontsize=8, title="Configuration")
    _save_figure(fig, output_dir / "all_candidates_convergence.png")


def write_mlp_training_summary_plot(
    training_history: pd.DataFrame,
    test_comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    if training_history.empty:
        return
    group_columns = ["granularity", "candidate_label"]
    if "model_variant" in training_history.columns:
        group_columns.append("model_variant")
    group_columns.append("learning_rate_init")
    final_loss = (
        training_history.sort_values("step")
        .groupby(group_columns, as_index=False)
        .tail(1)[group_columns + ["train_loss", "best_train_loss"]]
        .rename(columns={"train_loss": "final_train_loss"})
    )
    if final_loss.empty:
        return

    summary = final_loss.copy()
    if not test_comparison.empty and set(group_columns + ["mae"]).issubset(test_comparison.columns):
        metric_columns = [column for column in ["mae", "rmse", "r2"] if column in test_comparison.columns]
        metrics = test_comparison[group_columns + metric_columns].copy()
        summary = summary.merge(metrics, on=group_columns, how="left")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    sns.scatterplot(
        data=summary,
        x="final_train_loss",
        y="best_train_loss",
        hue="granularity",
        style="learning_rate_init",
        ax=axes[0],
    )
    axes[0].set_title("Final Loss vs Best Loss")
    axes[0].set_xlabel("Final train loss")
    axes[0].set_ylabel("Best train loss")
    axes[0].legend(loc="best", fontsize=8)

    if "mae" in summary.columns and summary["mae"].notna().any():
        sns.scatterplot(
            data=summary,
            x="best_train_loss",
            y="mae",
            hue="granularity",
            style="learning_rate_init",
            ax=axes[1],
        )
        axes[1].set_title("Best Train Loss vs Test MAE")
        axes[1].set_xlabel("Best train loss")
        axes[1].set_ylabel("Test MAE")
        axes[1].legend(loc="best", fontsize=8)
    else:
        sns.scatterplot(
            data=summary,
            x="learning_rate_init",
            y="best_train_loss",
            hue="granularity",
            ax=axes[1],
        )
        axes[1].set_title("Best Train Loss By Learning Rate")
        axes[1].set_xlabel("Learning rate")
        axes[1].set_ylabel("Best train loss")
        axes[1].legend(loc="best", fontsize=8)
    _save_figure(fig, output_path)


def _downsample_by_step(frame: pd.DataFrame, max_points: int) -> pd.DataFrame:
    if len(frame.index) <= max_points:
        return frame
    stride = max(1, len(frame.index) // max_points)
    sampled = frame.iloc[::stride].copy()
    if sampled.iloc[-1]["step"] != frame.iloc[-1]["step"]:
        sampled = pd.concat([sampled, frame.tail(1)], ignore_index=True)
    return sampled


def _safe_filename(value: str) -> str:
    return (
        value.replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace(" ", "_")
        .replace("=", "_")
    )


def write_model_history_plot(model_key: str, model_frame: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ordered = model_frame.sort_values(["granularity", "run_created_at", "run_id"])
    for axis, metric in zip(axes, ["mae", "rmse"]):
        sns.barplot(data=ordered, x="run_label", y=metric, hue="granularity", ax=axis)
        axis.set_title(f"{model_key} Test {metric.upper()} Across Runs")
        axis.set_xlabel("Run")
        axis.set_ylabel(metric.upper())
        axis.tick_params(axis="x", rotation=45)
        axis.legend(loc="upper right", fontsize=8)
    _save_figure(fig, output_path)


def compare_archived_mlp_stage_runs(
    stage_a_run: Path,
    stage_a_label: str,
    stage_b_run: Path,
    stage_b_label: str,
    output_dir: Path,
) -> None:
    stage_a_metrics = _load_archived_mlp_metrics(stage_a_run, stage_a_label)
    stage_b_metrics = _load_archived_mlp_metrics(stage_b_run, stage_b_label)
    metrics = pd.concat([stage_a_metrics, stage_b_metrics], ignore_index=True)
    selected_candidates = sorted(set(stage_a_metrics["candidate_label"]) & set(stage_b_metrics["candidate_label"]))
    if selected_candidates:
        metrics = metrics[metrics["candidate_label"].isin(selected_candidates)].copy()
    _write_archived_stage_metric_comparisons(metrics, output_dir / "metrics")

    stage_a_forecasts = _load_archived_mlp_forecasts(stage_a_run, stage_a_label)
    stage_b_forecasts = _load_archived_mlp_forecasts(stage_b_run, stage_b_label)
    forecasts = pd.concat([stage_a_forecasts, stage_b_forecasts], ignore_index=True)
    _write_archived_stage_forecast_comparisons(forecasts, metrics, output_dir / "forecast_windows")
    _write_archived_stage_error_distribution(forecasts, metrics, output_dir / "error_distribution")


def compare_stage3_univariate_mlp_to_baseline(
    baseline_dir: Path,
    advanced_dir: Path,
    output_dir: Path,
) -> None:
    baseline_metrics = _load_stage_mlp_metrics(baseline_dir, "Stage 2 baseline")
    advanced_metrics = _load_stage_mlp_metrics(advanced_dir, "Stage 3 advanced MLP")
    baseline_metrics["comparison_family"] = "Baseline"
    advanced_metrics["comparison_family"] = "Advanced MLP"
    metrics = pd.concat([baseline_metrics, advanced_metrics], ignore_index=True)
    _write_stage3_metric_comparison(metrics, output_dir / "metrics")

    baseline_forecasts = _load_stage_mlp_forecasts(baseline_dir, "Stage 2 baseline")
    advanced_forecasts = _load_stage_mlp_forecasts(advanced_dir, "Stage 3 advanced MLP")
    forecasts = pd.concat([baseline_forecasts, advanced_forecasts], ignore_index=True)
    _write_stage3_forecast_comparison(forecasts, metrics, output_dir / "forecast_windows")
    _write_stage3_error_distribution(forecasts, metrics, output_dir / "error_distribution")
    _write_stage3_mlp_report(metrics, output_dir / "stage3_univariate_mlp_comparison.md")


def _load_stage_mlp_metrics(stage_dir: Path, stage_label: str) -> pd.DataFrame:
    path = stage_dir / "tables" / "forecasting_summary" / "univariate_mlp_test_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing MLP metrics table: {path}")
    frame = pd.read_csv(path)
    frame = frame[frame.get("split", "test") == "test"].copy()
    for column in ["mae", "rmse", "r2", "learning_rate_init", "training_seconds"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["stage"] = stage_label
    return frame


def _load_stage_mlp_forecasts(stage_dir: Path, stage_label: str) -> pd.DataFrame:
    candidates = [
        stage_dir / "forecasting" / "mlp_forecasts.csv",
        stage_dir / "forecasting" / "univariate" / "mlp_forecasts.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        print(f"[stage3-comparison] No mlp_forecasts.csv found for {stage_label}: {stage_dir}", flush=True)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"ds", "y", "forecast", "granularity", "candidate_label", "model_variant", "learning_rate_init"}
    if not required.issubset(frame.columns):
        print(f"[stage3-comparison] Forecast file is missing required columns: {path}", flush=True)
        return pd.DataFrame()
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    frame["stage"] = stage_label
    for column in ["y", "forecast", "learning_rate_init"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "split" in frame.columns:
        frame = frame[frame["split"] == "test"].copy()
    return frame.dropna(subset=["ds", "y", "forecast"])


def _write_stage3_metric_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "stage3_all_mlp_metrics.csv", index=False)
    best_stage3 = _best_archived_rows(
        metrics[metrics["comparison_family"] == "Advanced MLP"].copy(),
        ["granularity", "candidate_label"],
        "r2",
    )
    baseline = metrics[metrics["comparison_family"] == "Baseline"].copy()
    summary = baseline.merge(
        best_stage3,
        on=["granularity", "candidate_label"],
        suffixes=("_baseline", "_stage3_best"),
    )
    if summary.empty:
        return
    summary["r2_delta"] = summary["r2_stage3_best"] - summary["r2_baseline"]
    summary["mae_delta"] = summary["mae_stage3_best"] - summary["mae_baseline"]
    summary["rmse_delta"] = summary["rmse_stage3_best"] - summary["rmse_baseline"]
    summary["mae_change_pct"] = 100.0 * summary["mae_delta"] / summary["mae_baseline"]
    summary["rmse_change_pct"] = 100.0 * summary["rmse_delta"] / summary["rmse_baseline"]
    summary.to_csv(output_dir / "stage3_best_vs_baseline_summary.csv", index=False)

    clustered = pd.concat([baseline, metrics[metrics["comparison_family"] == "Advanced MLP"]], ignore_index=True)
    clustered["plot_model"] = clustered.apply(_stage3_metric_plot_label, axis=1)
    for metric in ["r2", "mae", "rmse"]:
        fig, axis = plt.subplots(figsize=(12, 5))
        sns.barplot(data=clustered, x="granularity", y=metric, hue="plot_model", ax=axis)
        axis.set_title(f"Stage 3 Advanced MLP vs Stage 2 Baseline - {metric.upper()}")
        axis.set_xlabel("Granularity")
        axis.set_ylabel(metric.upper())
        axis.legend(loc="best", fontsize=8, title="Model")
        _save_figure(fig, output_dir / f"stage3_vs_baseline_{metric}.png")

    delta_long = summary.melt(
        id_vars=["granularity", "candidate_label", "model_variant_stage3_best"],
        value_vars=["r2_delta", "mae_delta", "rmse_delta"],
        var_name="metric_delta",
        value_name="delta",
    )
    fig, axis = plt.subplots(figsize=(12, 5))
    sns.barplot(data=delta_long, x="granularity", y="delta", hue="metric_delta", ax=axis)
    axis.axhline(0.0, color="#333333", linewidth=0.9)
    axis.set_title("Best Stage 3 MLP Change Relative To Stage 2 Baseline")
    axis.set_xlabel("Granularity")
    axis.set_ylabel("Stage 3 best - baseline")
    axis.legend(loc="best", fontsize=8, title="Metric delta")
    _save_figure(fig, output_dir / "stage3_best_delta_vs_baseline.png")


def _stage3_metric_plot_label(row: pd.Series) -> str:
    if row.get("comparison_family") == "Baseline" or row.get("stage") == "Stage 2 baseline":
        return "Baseline 1-hidden"
    variant = str(row.get("model_variant", "advanced"))
    if "fixed_16" in variant:
        return "Advanced 2-hidden fixed 16"
    if "match_lookback" in variant:
        return "Advanced 2-hidden match lookback"
    return variant


def _write_stage3_forecast_comparison(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    if forecasts.empty or metrics.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    best_stage3 = _best_archived_rows(
        metrics[metrics["comparison_family"] == "Advanced MLP"].copy(),
        ["granularity", "candidate_label"],
        "r2",
    )
    baseline = metrics[metrics["comparison_family"] == "Baseline"].copy()
    keep_rows = pd.concat([baseline, best_stage3], ignore_index=True)
    merge_keys = ["stage", "granularity", "candidate_label", "model_variant", "learning_rate_init"]
    selected = forecasts.merge(keep_rows[merge_keys].drop_duplicates(), on=merge_keys, how="inner")
    if selected.empty:
        return
    selected["plot_model"] = selected.apply(_stage3_metric_plot_label, axis=1)
    for (granularity, candidate_label), frame in selected.groupby(["granularity", "candidate_label"]):
        plot_frame = frame.sort_values("ds").groupby("plot_model", group_keys=False).head(300)
        if plot_frame["plot_model"].nunique() < 2:
            continue
        output_scale = _first_forecast_output_scale(plot_frame)
        y_label = "Scaled target" if output_scale == "scaled" else "Target"
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        actual = plot_frame.sort_values("ds").drop_duplicates("ds")
        axes[0].plot(actual["ds"], actual["y"], color="#222222", linewidth=1.2, label="Actual")
        for model_label, model_frame in plot_frame.groupby("plot_model"):
            model_frame = model_frame.sort_values("ds")
            axes[0].plot(model_frame["ds"], model_frame["forecast"], linewidth=1.0, label=model_label)
            axes[1].plot(
                model_frame["ds"],
                (model_frame["forecast"] - model_frame["y"]).abs(),
                linewidth=1.0,
                label=model_label,
            )
        axes[0].set_title(f"{granularity} Forecast Window: Stage 2 Baseline vs Best Stage 3 MLP")
        axes[0].set_ylabel(y_label)
        axes[0].legend(loc="best", fontsize=8)
        axes[1].set_title(f"{candidate_label} absolute forecast error")
        axes[1].set_xlabel("date")
        axes[1].set_ylabel(f"Absolute error ({output_scale})")
        axes[1].legend(loc="best", fontsize=8)
        _save_figure(fig, output_dir / f"{granularity}_{_safe_filename(str(candidate_label))}_baseline_vs_stage3_forecast.png")


def _write_stage3_error_distribution(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    if forecasts.empty or metrics.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    best_stage3 = _best_archived_rows(
        metrics[metrics["comparison_family"] == "Advanced MLP"].copy(),
        ["granularity", "candidate_label"],
        "r2",
    )
    baseline = metrics[metrics["comparison_family"] == "Baseline"].copy()
    keep_rows = pd.concat([baseline, best_stage3], ignore_index=True)
    merge_keys = ["stage", "granularity", "candidate_label", "model_variant", "learning_rate_init"]
    selected = forecasts.merge(keep_rows[merge_keys].drop_duplicates(), on=merge_keys, how="inner")
    if selected.empty:
        return
    selected["plot_model"] = selected.apply(_stage3_metric_plot_label, axis=1)
    selected["absolute_error"] = (selected["forecast"] - selected["y"]).abs()
    selected.to_csv(output_dir / "stage3_best_vs_baseline_absolute_errors.csv", index=False)
    fig, axis = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=selected, x="granularity", y="absolute_error", hue="plot_model", showfliers=False, ax=axis)
    output_scale = _first_forecast_output_scale(selected)
    axis.set_title("Absolute Error Distribution: Stage 2 Baseline vs Best Stage 3 MLP")
    axis.set_xlabel("Granularity")
    axis.set_ylabel(f"Absolute error ({output_scale})")
    axis.legend(loc="best", fontsize=8, title="Model")
    _save_figure(fig, output_dir / "stage3_best_vs_baseline_absolute_error_distribution.png")


def _write_stage3_mlp_report(metrics: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    best_stage3 = _best_archived_rows(
        metrics[metrics["comparison_family"] == "Advanced MLP"].copy(),
        ["granularity", "candidate_label"],
        "r2",
    )
    baseline = metrics[metrics["comparison_family"] == "Baseline"].copy()
    summary = baseline.merge(
        best_stage3,
        on=["granularity", "candidate_label"],
        suffixes=("_baseline", "_stage3_best"),
    )
    if summary.empty:
        output_path.write_text("# Stage 3 Univariate MLP Comparison\n\nNo comparable metrics were available.\n", encoding="utf-8")
        return
    summary["r2_delta"] = summary["r2_stage3_best"] - summary["r2_baseline"]
    summary["mae_delta"] = summary["mae_stage3_best"] - summary["mae_baseline"]
    improved_rows = summary[summary["r2_delta"] > 0]
    verdict = (
        "Stage 3 MLP improves the baseline for at least one granularity, but the gains are small and not consistent."
        if not improved_rows.empty
        else "Stage 3 MLP does not materially improve the Stage 2 3-minute baseline."
    )
    lines = [
        "# Stage 3 Univariate MLP Comparison",
        "",
        "## Question",
        "Does the advanced two-hidden-layer MLP work better than the Stage 2 one-hidden-layer 3-minute baseline?",
        "",
        "## Short Answer",
        verdict,
        "",
        "## Best Stage 3 Result Per Granularity",
        "",
        "| Granularity | Baseline R2 | Best Stage 3 R2 | R2 Delta | Baseline MAE | Best Stage 3 MAE | MAE Delta | Best Stage 3 Variant |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summary.sort_values("granularity").to_dict("records"):
        lines.append(
            "| {granularity} | {r2_base:.4f} | {r2_adv:.4f} | {r2_delta:+.4f} | "
            "{mae_base:.4f} | {mae_adv:.4f} | {mae_delta:+.4f} | {variant} |".format(
                granularity=row["granularity"],
                r2_base=float(row["r2_baseline"]),
                r2_adv=float(row["r2_stage3_best"]),
                r2_delta=float(row["r2_delta"]),
                mae_base=float(row["mae_baseline"]),
                mae_adv=float(row["mae_stage3_best"]),
                mae_delta=float(row["mae_delta"]),
                variant=row["model_variant_stage3_best"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "- The comparison uses the same top preprocessing candidate per granularity and the same 3-minute forecast horizon.",
            "- Stage 3 changes the MLP architecture, so the relevant comparison is against the Stage 2 3-minute baseline, not against one-step results.",
            "- A useful improvement should increase R2 and reduce MAE/RMSE consistently. Small mixed changes should be treated as inconclusive.",
            "",
            "## Supporting Artifacts",
            "- `metrics/stage3_vs_baseline_r2.png`",
            "- `metrics/stage3_vs_baseline_mae.png`",
            "- `metrics/stage3_vs_baseline_rmse.png`",
            "- `metrics/stage3_best_delta_vs_baseline.png`",
            "- `error_distribution/stage3_best_vs_baseline_absolute_error_distribution.png`",
            "- `forecast_windows/*_baseline_vs_stage3_forecast.png`",
            "",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def _load_archived_mlp_metrics(run_dir: Path, stage_label: str) -> pd.DataFrame:
    path = run_dir / "tables" / "forecasting_summary" / "univariate_mlp_test_comparison.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing MLP metrics table: {path}")
    frame = pd.read_csv(path)
    frame = frame[frame.get("split", "test") == "test"].copy()
    for column in ["mae", "rmse", "r2", "learning_rate_init"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame["stage"] = stage_label
    return _best_archived_rows(frame, ["stage", "granularity", "candidate_label"], "r2")


def _load_archived_mlp_forecasts(run_dir: Path, stage_label: str) -> pd.DataFrame:
    candidates = [
        run_dir / "forecasting" / "univariate" / "mlp_forecasts.csv",
        run_dir / "forecasting" / "mlp_forecasts.csv",
    ]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        print(f"[stage-comparison] No archived mlp_forecasts.csv found for {stage_label}: {run_dir}", flush=True)
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required = {"ds", "y", "forecast", "granularity", "candidate_label"}
    if not required.issubset(frame.columns):
        print(f"[stage-comparison] Forecast file is missing required columns: {path}", flush=True)
        return pd.DataFrame()
    frame["ds"] = pd.to_datetime(frame["ds"], errors="coerce")
    frame["stage"] = stage_label
    for column in ["y", "forecast", "learning_rate_init"]:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    if "split" in frame.columns:
        frame = frame[frame["split"] == "test"].copy()
    return frame.dropna(subset=["ds", "y", "forecast"])


def _write_archived_stage_metric_comparisons(metrics: pd.DataFrame, output_dir: Path) -> None:
    if metrics.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "stage_mlp_candidate_metrics.csv", index=False)
    wide = _build_stage_metric_delta_frame(metrics)
    if wide.empty:
        return
    wide.to_csv(output_dir / "horizon_impact_summary.csv", index=False)

    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=wide.sort_values("granularity"), x="granularity", y="r2_drop", hue="candidate_label", dodge=False, ax=axis)
    axis.set_title("R2 Decrease From One-Step To 3-Minute Forecast Horizon")
    axis.set_xlabel("Granularity")
    axis.set_ylabel("R2 drop: 1step - 3min")
    axis.legend(loc="best", fontsize=8, title="Candidate")
    _save_figure(fig, output_dir / "defense_r2_drop_by_granularity.png")

    long_increase = wide.melt(
        id_vars=["granularity", "candidate_label"],
        value_vars=["mae_increase_factor", "rmse_increase_factor"],
        var_name="metric",
        value_name="increase_factor",
    )
    long_increase["metric"] = long_increase["metric"].map(
        {"mae_increase_factor": "MAE", "rmse_increase_factor": "RMSE"}
    )
    fig, axis = plt.subplots(figsize=(10, 5))
    sns.barplot(data=long_increase, x="granularity", y="increase_factor", hue="metric", ax=axis)
    axis.axhline(1.0, color="#333333", linewidth=0.9)
    axis.set_title("Relative Error Increase From One-Step To 3-Minute Forecast Horizon")
    axis.set_xlabel("Granularity")
    axis.set_ylabel("3min error / 1step error")
    axis.legend(loc="best", fontsize=8, title="Metric")
    _save_figure(fig, output_dir / "defense_error_increase_factor.png")


def _build_stage_metric_delta_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"stage", "granularity", "candidate_label", "mae", "rmse", "r2"}
    if not required.issubset(metrics.columns) or metrics["stage"].nunique() != 2:
        return pd.DataFrame()
    stages = list(metrics["stage"].drop_duplicates())
    stage_a, stage_b = stages[0], stages[1]
    wide = metrics.pivot_table(
        index=["granularity", "candidate_label"],
        columns="stage",
        values=["mae", "rmse", "r2"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{stage}" for metric, stage in wide.columns]
    wide = wide.reset_index()
    wide["stage_a"] = stage_a
    wide["stage_b"] = stage_b
    wide["r2_drop"] = wide[f"r2_{stage_a}"] - wide[f"r2_{stage_b}"]
    wide["mae_increase_factor"] = wide[f"mae_{stage_b}"] / wide[f"mae_{stage_a}"]
    wide["rmse_increase_factor"] = wide[f"rmse_{stage_b}"] / wide[f"rmse_{stage_a}"]
    return wide


def _write_archived_stage_forecast_comparisons(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    if forecasts.empty or metrics.empty:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    metric_keys = [
        column
        for column in ["stage", "granularity", "candidate_label", "model_variant", "learning_rate_init"]
        if column in metrics.columns
    ]
    selected = forecasts.merge(metrics[metric_keys].drop_duplicates(), on=metric_keys, how="inner")
    if selected.empty:
        selected = forecasts[forecasts["candidate_label"].isin(metrics["candidate_label"].unique())].copy()

    for (granularity, candidate_label), frame in selected.groupby(["granularity", "candidate_label"]):
        plot_frame = frame.sort_values("ds").groupby("stage", group_keys=False).head(300)
        if plot_frame.empty or plot_frame["stage"].nunique() < 2:
            continue
        output_scale = _first_forecast_output_scale(plot_frame)
        y_label = "Scaled target" if output_scale == "scaled" else "Target"
        fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
        first_stage = plot_frame["stage"].iloc[0]
        actual = plot_frame[plot_frame["stage"] == first_stage].sort_values("ds")
        axes[0].plot(actual["ds"], actual["y"], color="#222222", linewidth=1.2, label="Actual")
        for stage, stage_frame in plot_frame.groupby("stage"):
            stage_frame = stage_frame.sort_values("ds")
            axes[0].plot(stage_frame["ds"], stage_frame["forecast"], linewidth=1.0, label=stage)
            axes[1].plot(stage_frame["ds"], (stage_frame["forecast"] - stage_frame["y"]).abs(), linewidth=1.0, label=stage)
        axes[0].set_title(f"{granularity} Baseline MLP Forecasts: One-Step vs 3-Minute Horizon")
        axes[0].set_ylabel(y_label)
        axes[0].legend(loc="best", fontsize=8)
        axes[1].set_title(f"{candidate_label} absolute forecast error")
        axes[1].set_xlabel("date")
        axes[1].set_ylabel(f"Absolute error ({output_scale})")
        axes[1].legend(loc="best", fontsize=8)
        _save_figure(fig, output_dir / f"defense_{granularity}_{_safe_filename(str(candidate_label))}_forecast_window.png")


def _write_archived_stage_error_distribution(
    forecasts: pd.DataFrame,
    metrics: pd.DataFrame,
    output_dir: Path,
) -> None:
    if forecasts.empty or metrics.empty:
        return
    required = {"stage", "granularity", "candidate_label", "y", "forecast"}
    if not required.issubset(forecasts.columns):
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    selected = forecasts[forecasts["candidate_label"].isin(metrics["candidate_label"].unique())].copy()
    if selected.empty:
        return
    selected["absolute_error"] = (selected["forecast"] - selected["y"]).abs()
    selected.to_csv(output_dir / "stage_forecast_absolute_errors.csv", index=False)
    fig, axis = plt.subplots(figsize=(11, 5))
    sns.boxplot(
        data=selected,
        x="granularity",
        y="absolute_error",
        hue="stage",
        showfliers=False,
        ax=axis,
    )
    output_scale = _first_forecast_output_scale(selected)
    axis.set_title("Absolute Forecast Error Distribution By Horizon")
    axis.set_xlabel("Granularity")
    axis.set_ylabel(f"Absolute error ({output_scale})")
    axis.legend(loc="best", fontsize=8)
    _save_figure(fig, output_dir / "defense_absolute_error_distribution.png")


def _best_archived_rows(frame: pd.DataFrame, group_columns: list[str], metric: str) -> pd.DataFrame:
    ascending = metric != "r2"
    tie_columns = [column for column in ["rmse", "mae"] if column in frame.columns and column != metric]
    sort_columns = group_columns + [metric] + tie_columns
    sort_ascending = [True] * len(group_columns) + [ascending] + [True] * len(tie_columns)
    return frame.sort_values(sort_columns, ascending=sort_ascending).groupby(group_columns, as_index=False).head(1)


def _first_forecast_output_scale(frame: pd.DataFrame) -> str:
    if "forecast_output_scale" not in frame.columns:
        return "original"
    values = frame["forecast_output_scale"].dropna()
    if values.empty:
        return "original"
    return str(values.iloc[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot archived forecasting results.")
    parser.add_argument("--compare-archived-stages", action="store_true")
    parser.add_argument("--compare-stage3-mlp", action="store_true")
    parser.add_argument("--stage-a-run", type=Path)
    parser.add_argument("--stage-a-label", type=str, default="Stage 2 - 1step baseline MLP")
    parser.add_argument("--stage-b-run", type=Path)
    parser.add_argument("--stage-b-label", type=str, default="Stage 2 - 3min baseline MLP")
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("outputs/chinese_boiler_dataset/current/stage_2_horizon_impact/stage_2_baseline_3min_top3"),
    )
    parser.add_argument(
        "--advanced-dir",
        type=Path,
        default=Path("outputs/chinese_boiler_dataset/current/stage_3_advanced_models/univariate_mlp"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    if args.compare_archived_stages:
        if args.stage_a_run is None or args.stage_b_run is None:
            parser.error("--stage-a-run and --stage-b-run are required with --compare-archived-stages.")
        output_dir = args.output_dir or Path("outputs/chinese_boiler_dataset/current/stage_2_horizon_impact/comparison")
        compare_archived_mlp_stage_runs(
            args.stage_a_run,
            args.stage_a_label,
            args.stage_b_run,
            args.stage_b_label,
            output_dir,
        )
        return 0
    if args.compare_stage3_mlp:
        output_dir = args.output_dir or Path("outputs/chinese_boiler_dataset/current/stage_3_advanced_models/univariate_mlp/plots/comparison")
        compare_stage3_univariate_mlp_to_baseline(args.baseline_dir, args.advanced_dir, output_dir)
        return 0
    parser.error("No plotting action selected. Use --compare-archived-stages or --compare-stage3-mlp.")


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied.*", category=UserWarning)
        fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
