from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


sns.set_theme(style="whitegrid")


# ---------------------------------------------------------------------------
# Spec-driven plotting
# ---------------------------------------------------------------------------
# Plot specs describe what should be drawn. This module owns how each plot type
# is rendered. Keeping plot intent in JSON makes it easier to change the
# presentation layer without touching analysis code.

def write_plot_specs(plot_specs: list[dict[str, Any]], frames: dict[str, pd.DataFrame]) -> None:
    handlers = {
        "time_series_groups": _plot_time_series_groups,
        "stacked_time_series": _plot_stacked_time_series,
        "target_granularity_comparison": _plot_target_granularity_comparison,
        "heatmap": _plot_heatmap_from_spec,
        "target_transform_plots": _plot_target_transform_plots,
        "smoothing_std_summary": _plot_smoothing_std_summary,
    }

    for spec in plot_specs:
        kind = str(spec["kind"])
        handlers[kind](spec, frames)


# ---------------------------------------------------------------------------
# Time-series plots
# ---------------------------------------------------------------------------
# These functions visualize selected process signals over time. Grouped plots
# are used only when the configuration says the variables share a meaningful
# physical family or comparable interpretation.

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
    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharey=True)
    axes = axes.ravel()

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
# Heatmaps and standardized boxplots are descriptive tools. Standardization is
# applied only for cross-variable distribution comparison because raw units are
# not comparable across pressure, temperature, oxygen, flow, and fan signals.

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


def _plot_hist_grid_by_dataset(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    dataset_keys = spec["dataset_keys"]
    target_column = str(spec["target_column"])
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for axis, frame_key in zip(axes, dataset_keys):
        axis.hist(frames[frame_key][target_column].dropna(), bins=50, color="#2f6f8f", alpha=0.85)
        axis.set_title(frame_key.replace("subset_B_", ""))
        axis.set_xlabel(spec.get("xlabel", "Value"))
        axis.set_ylabel("Frequency")

    fig.suptitle(spec["title"])
    _save_figure(fig, Path(spec["output_path"]))


def _plot_standardized_boxplots(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    output_dir = Path(spec["output_dir"])
    timestamp_column = str(spec.get("timestamp_column", "date"))

    for frame_key in spec["dataset_keys"]:
        frame = frames[frame_key].drop(columns=timestamp_column)
        standardized = (frame - frame.mean()) / frame.std()
        fig, axis = plt.subplots(figsize=(14, 6))
        standardized.boxplot(ax=axis, showfliers=False, rot=75)
        axis.set_title(f"{frame_key} - Standardized Variable Distribution")
        axis.set_ylabel("Z-score")
        _save_figure(fig, output_dir / f"{frame_key}_standardized_boxplot.png")


# ---------------------------------------------------------------------------
# Target preprocessing plots
# ---------------------------------------------------------------------------
# Smoothing and first-difference plots are generated from cached transform
# series. This keeps the plotted values aligned with the statistical summaries.

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

        _plot_smoothing_comparison(
            id_key,
            original,
            smoothed,
            output_dir,
            str(spec.get("target_ylabel", "Target")),
            str(spec.get("zoom_duration", "6h")),
        )
        _plot_first_difference(id_key, first_difference, output_dir)


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
# Forecasting modules pass already-evaluated forecast frames here. This keeps
# model fitting separate from visual diagnostics.

def write_forecast_plot(
    id_key: str,
    granularity: str,
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    forecast_frames: list[pd.DataFrame],
    output_dir: Path,
    model_family: str,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    train_tail = train_frame.tail(min(len(train_frame), len(test_frame)))

    axes[0].plot(train_tail["ds"], train_tail["y"], linewidth=0.9, color="#888888", label="train tail")
    axes[0].plot(test_frame["ds"], test_frame["y"], linewidth=1.0, color="#222222", label="test actual")
    for forecast_frame in forecast_frames:
        axes[0].plot(
            forecast_frame["ds"],
            forecast_frame["forecast"],
            linewidth=1.0,
            label=str(forecast_frame["model"].iloc[0]),
        )
    axes[0].axvline(test_frame["ds"].iloc[0], color="#333333", linewidth=0.9, linestyle="--")
    axes[0].set_title(f"{id_key} - 80/20 {model_family} Forecast")
    axes[0].set_ylabel("Target")
    axes[0].legend(loc="upper right", fontsize=8)

    for forecast_frame in forecast_frames:
        axes[1].plot(
            forecast_frame["ds"],
            forecast_frame["forecast"] - forecast_frame["y"],
            linewidth=1.0,
            label=str(forecast_frame["model"].iloc[0]),
        )
    axes[1].axhline(0, color="#333333", linewidth=0.8)
    axes[1].set_title("Forecast Error")
    axes[1].set_xlabel("date")
    axes[1].set_ylabel("Forecast - actual")
    axes[1].legend(loc="upper right", fontsize=8)
    _save_figure(fig, output_dir / f"{id_key}_{model_family.lower()}_forecast.png")


def write_forecast_metrics_plot(metrics: pd.DataFrame, output_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for axis, metric in zip(axes, ["mae", "rmse"]):
        pivot = metrics.pivot(index="granularity", columns="model", values=metric)
        pivot.plot(kind="bar", ax=axis)
        axis.set_title(metric.upper())
        axis.set_xlabel("Granularity")
        axis.set_ylabel(metric.upper())
        axis.legend(loc="upper right", fontsize=8)
    _save_figure(fig, output_path)


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Tight layout not applied.*", category=UserWarning)
        fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
