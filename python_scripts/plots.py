from __future__ import annotations

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
        "hist_grid_by_dataset": _plot_hist_grid_by_dataset,
        "standardized_boxplots": _plot_standardized_boxplots,
        "barh": _plot_barh,
        "window_bar": _plot_window_bar,
        "simultaneous_counts": _plot_simultaneous_counts,
        "outlier_timeline": _plot_outlier_timeline,
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
# Outlier plots
# ---------------------------------------------------------------------------
# Outlier visuals are intentionally compact: counts by variable, dense windows,
# simultaneous events, and a timeline. Large timestamp-level CSVs are avoided
# unless there is a corresponding plot that makes them reviewable.

def _plot_barh(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    frame = frames[str(spec["frame"])]
    if frame.empty:
        return

    sort_by = str(spec.get("sort_by", spec["x"]))
    fig, axis = plt.subplots(figsize=spec.get("figsize", (10, 6)))
    frame.sort_values(sort_by).plot(
        x=spec["y"],
        y=spec["x"],
        kind="barh",
        ax=axis,
        color=spec.get("color", "#ba4e00"),
    )
    axis.set_title(spec["title"])
    axis.set_xlabel(spec["xlabel"])
    axis.set_ylabel(spec["ylabel"])
    legend = axis.get_legend()
    if legend is not None:
        legend.remove()
    _save_figure(fig, Path(spec["output_path"]))


def _plot_window_bar(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    frame = frames[str(spec["frame"])].copy()
    if frame.empty:
        return

    frame["label"] = frame[str(spec.get("label_column", "window_start"))].astype(str)
    fig, axis = plt.subplots(figsize=spec.get("figsize", (12, 7)))
    sns.barplot(data=frame, y="label", x=spec["x"], ax=axis, color=spec.get("color", "#b85c38"))
    axis.set_title(spec["title"])
    axis.set_xlabel(spec["xlabel"])
    axis.set_ylabel(spec["ylabel"])
    _save_figure(fig, Path(spec["output_path"]))


def _plot_simultaneous_counts(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    frame = frames[str(spec["frame"])]
    if frame.empty:
        return

    counts = frame["variable_count"].value_counts().sort_index().reset_index()
    counts.columns = ["simultaneous_variable_count", "timestamp_count"]
    fig, axis = plt.subplots(figsize=(8, 5))
    sns.barplot(data=counts, x="simultaneous_variable_count", y="timestamp_count", ax=axis, color="#11875d")
    axis.set_title(spec["title"])
    axis.set_xlabel("Variables outlying at the same timestamp")
    axis.set_ylabel("Timestamp count")
    _save_figure(fig, Path(spec["output_path"]))


def _plot_outlier_timeline(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    frame = frames[str(spec["frame"])]
    if frame.empty:
        return

    timeline = frame.copy()
    timeline["window_start"] = timeline["date"].dt.floor(str(spec.get("window", "5min")))
    timeline = timeline.groupby("window_start").size().reset_index(name="event_count")
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(timeline["window_start"], timeline["event_count"], linewidth=1.2, color="#8a3ffc")
    axis.set_title(spec["title"])
    axis.set_xlabel("date")
    axis.set_ylabel("Outlier event count")
    _save_figure(fig, Path(spec["output_path"]))


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

        _plot_smoothing_comparison(id_key, original, smoothed, output_dir, str(spec.get("target_ylabel", "Target")))
        _plot_first_difference(id_key, first_difference, output_dir)


def _plot_smoothing_std_summary(spec: dict[str, Any], frames: dict[str, pd.DataFrame]) -> None:
    frame = frames[str(spec["frame"])].copy()
    frame["transform_label"] = frame["transform"] + " (" + frame["window"] + ")"
    fig, axis = plt.subplots(figsize=(12, 6))
    sns.barplot(data=frame, x="granularity", y="std", hue="transform_label", ax=axis)
    axis.set_title(spec["title"])
    axis.set_xlabel("Granularity")
    axis.set_ylabel("Std")
    axis.legend(loc="upper right", fontsize=8)
    _save_figure(fig, Path(spec["output_path"]))


def _plot_smoothing_comparison(
    id_key: str,
    original: dict[str, Any],
    smoothed: list[dict[str, Any]],
    output_dir: Path,
    ylabel: str,
) -> None:
    fig, axis = plt.subplots(figsize=(14, 5))
    axis.plot(original["dates"], original["values"], linewidth=0.9, alpha=0.55, label="original")
    for item in smoothed:
        axis.plot(item["dates"], item["values"], linewidth=1.2, label=f"rolling mean {item['window']}")
    axis.set_title(f"{id_key} - Target Smoothing Comparison")
    axis.set_xlabel("date")
    axis.set_ylabel(ylabel)
    axis.legend(loc="upper right", fontsize=8)
    _save_figure(fig, output_dir / f"{id_key}_target_smoothing_comparison.png")


def _plot_first_difference(id_key: str, first_difference: dict[str, Any], output_dir: Path) -> None:
    fig, axis = plt.subplots(figsize=(14, 4))
    axis.plot(first_difference["dates"], first_difference["values"], linewidth=0.9, color="#4c72b0")
    axis.axhline(0, color="#333333", linewidth=0.8)
    axis.set_title(f"{id_key} - Target First Difference")
    axis.set_xlabel("date")
    axis.set_ylabel("Delta temp")
    _save_figure(fig, output_dir / f"{id_key}_target_first_difference.png")


def _save_figure(fig: plt.Figure, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
