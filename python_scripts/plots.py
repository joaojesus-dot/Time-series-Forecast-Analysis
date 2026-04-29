from __future__ import annotations

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
) -> None:
    """Write human-facing comparison plots for univariate model review."""
    metric_frames = []
    if not arima_metrics.empty:
        metric_frames.append(arima_metrics.assign(model_family="ARIMA", split="test"))
    if not mlp_test_comparison.empty and "split" in mlp_test_comparison.columns:
        test_mlp_metrics = mlp_test_comparison[mlp_test_comparison["split"] == "test"].copy()
        if not test_mlp_metrics.empty:
            metric_frames.append(test_mlp_metrics.assign(model_family="MLP"))
    if metric_frames:
        comparable_metrics = pd.concat(metric_frames, ignore_index=True)
        write_univariate_metric_comparison(comparable_metrics, output_dir / "univariate_metric_comparison.png")
    write_univariate_forecast_overlays(arima_forecasts, mlp_forecasts, output_dir)


def write_univariate_metric_comparison(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, metric in zip(axes, ["mae", "rmse"]):
        sns.barplot(data=metrics, x="granularity", y=metric, hue="model_family", ax=axis)
        axis.set_title(metric.upper())
        axis.set_xlabel("Granularity")
        axis.set_ylabel(metric.upper())
        axis.legend(loc="upper right", fontsize=8)
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

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(mlp_zoom["ds"], mlp_zoom["y"], color="#222222", linewidth=1.1, label="actual")
        axes[0].plot(arima_zoom["ds"], arima_zoom["forecast"], linewidth=1.0, label="ARIMA")
        axes[0].plot(mlp_zoom["ds"], mlp_zoom["forecast"], linewidth=1.0, label="MLP")
        axes[0].set_title(f"{granularity} - Test Forecast Comparison")
        axes[0].set_ylabel("Target")
        axes[0].legend(loc="upper right", fontsize=8)

        if not arima_zoom.empty:
            axes[1].plot(arima_zoom["ds"], arima_zoom["forecast"] - arima_zoom["y"], linewidth=1.0, label="ARIMA")
        axes[1].plot(mlp_zoom["ds"], mlp_zoom["forecast"] - mlp_zoom["y"], linewidth=1.0, label="MLP")
        axes[1].axhline(0, color="#333333", linewidth=0.8)
        axes[1].set_title("Forecast Error")
        axes[1].set_xlabel("date")
        axes[1].set_ylabel("Forecast - actual")
        axes[1].legend(loc="upper right", fontsize=8)
        _save_figure(fig, output_dir / f"{granularity}_univariate_forecast_overlay.png")


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
    if "learning_rate_init" in candidate_forecasts.columns:
        group_columns.append("learning_rate_init")

    for group_values, frame in candidate_forecasts.groupby(group_columns):
        if len(group_columns) == 3:
            granularity, candidate_label, learning_rate_init = group_values
            learning_rate_label = f"lr={learning_rate_init:g}"
        else:
            granularity, candidate_label = group_values
            learning_rate_label = ""
        plot_frame = frame.sort_values("ds").head(max_points).copy()
        if plot_frame.empty:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
        axes[0].plot(plot_frame["ds"], plot_frame["y"], color="#222222", linewidth=1.1, label="actual")
        axes[0].plot(plot_frame["ds"], plot_frame["forecast"], color="#1f4b99", linewidth=1.0, label="forecast")
        title_suffix = f" - {learning_rate_label}" if learning_rate_label else ""
        axes[0].set_title(f"{granularity} - {candidate_label}{title_suffix} Test Comparison")
        axes[0].set_ylabel("Target")
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
        axes[1].set_ylabel("Forecast - actual")
        axes[1].legend(loc="upper right", fontsize=8)

        safe_label = str(candidate_label).replace("/", "-").replace("\\", "-")
        safe_lr = learning_rate_label.replace("=", "_").replace(".", "p")
        filename = f"{safe_label}_{safe_lr}_test_comparison.png" if safe_lr else f"{safe_label}_test_comparison.png"
        _save_figure(fig, output_dir / str(granularity) / filename)


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


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied.*", category=UserWarning)
        fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
